/*!
 * Copyright 2018 XGBoost contributors
 */

#include <xgboost/logging.h>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <utility>
#include <vector>
#include <memory>
#include <mutex>

#include "hist_util.h"
#include "xgboost/host_device_vector.h"
#include "device_helpers.cuh"
#include "quantile.h"
#include "../tree/param.h"
#include "../data/adapter.h"
#include "math.h"
#include "../data/device_adapter.cuh"

namespace xgboost {
namespace common {

using WQSketch = DenseCuts::WQSketch;

__global__ void FindCutsK(WQSketch::Entry* __restrict__ cuts,
                          const bst_float* __restrict__ data,
                          const float* __restrict__ cum_weights,
                          int nsamples,
                          int ncuts) {
  // ncuts < nsamples
  int icut = threadIdx.x + blockIdx.x * blockDim.x;
  if (icut >= ncuts) {
    return;
  }
  int isample = 0;
  if (icut == 0) {
    isample = 0;
  } else if (icut == ncuts - 1) {
    isample = nsamples - 1;
  } else {
    bst_float rank = cum_weights[nsamples - 1] / static_cast<float>(ncuts - 1)
        * static_cast<float>(icut);
    // -1 is used because cum_weights is an inclusive sum
    isample = dh::UpperBound(cum_weights, nsamples, rank);
    isample = max(0, min(isample, nsamples - 1));
  }
  // repeated values will be filtered out on the CPU
  bst_float rmin = isample > 0 ? cum_weights[isample - 1] : 0;
  bst_float rmax = cum_weights[isample];
  cuts[icut] = WQSketch::Entry(rmin, rmax, rmax - rmin, data[isample]);
}

// predictate for thrust filtering that returns true if the element is not a NaN
struct IsNotNaN {
  __device__ bool operator()(float a) const { return !isnan(a); }
};

__global__ void UnpackFeaturesK(float* __restrict__ fvalues,
                                float* __restrict__ feature_weights,
                                const size_t* __restrict__ row_ptrs,
                                const float* __restrict__ weights,
                                Entry* entries,
                                size_t nrows_array,
                                size_t row_begin_ptr,
                                size_t nrows) {
  size_t irow = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
  if (irow >= nrows) {
    return;
  }
  size_t row_length = row_ptrs[irow + 1] - row_ptrs[irow];
  int icol = threadIdx.y + blockIdx.y * blockDim.y;
  if (icol >= row_length) {
    return;
  }
  Entry entry = entries[row_ptrs[irow] - row_begin_ptr + icol];
  size_t ind = entry.index * nrows_array + irow;
  // if weights are present, ensure that a non-NaN value is written to weights
  // if and only if it is also written to features
  if (!isnan(entry.fvalue) && (weights == nullptr || !isnan(weights[irow]))) {
    fvalues[ind] = entry.fvalue;
    if (feature_weights != nullptr && weights != nullptr) {
      feature_weights[ind] = weights[irow];
    }
  }
}

  using SketchEntry = WQuantileSketch<bst_float, bst_float>::Entry;
/*!
 * \brief A container that holds the device sketches across all
 *  sparse page batches which are distributed to different devices.
 *  As sketches are aggregated by column, the mutex guards
 *  multiple devices pushing sketch summary for the same column
 *  across distinct rows.
 */
struct SketchContainer {
  std::vector<DenseCuts::WQSketch> sketches_;  // NOLINT
  static constexpr int kOmpNumColsParallelizeLimit = 1000;

  SketchContainer(int max_bin,size_t num_columns,size_t num_rows )  {
    // Initialize Sketches for this dmatrix
    sketches_.resize(num_columns);
#pragma omp parallel for default(none) shared(max_bin) schedule(static) \
if (num_columns> kOmpNumColsParallelizeLimit)  // NOLINT
    for (int icol = 0; icol < num_columns; ++icol) {  // NOLINT
      sketches_[icol].Init(num_rows, 1.0 / (8 * max_bin));
    }
  }

  /**
   * \brief Pushes cuts to the sketches.
   *
   * \param entries_per_column  The entries per column.
   * \param entries             Vector of cuts from all columns, length entries_per_column * num_columns.
   * \param column_scan         Exclusive scan of column sizes. Used to detect cases where there are
   *                            fewer entries than we have storage for.
   */
  void Push(size_t entries_per_column,
            const thrust::host_vector<SketchEntry>& entries,
            const thrust::host_vector<size_t>& column_scan) {
#pragma omp parallel for default(none) schedule( \
    static) if (sketches_.size() >               \
                SketchContainer::kOmpNumColsParallelizeLimit)  // NOLINT
    for (int icol = 0; icol < sketches_.size(); ++icol) {
      size_t column_size = column_scan[icol + 1] - column_scan[icol];
      if (column_size == 0) continue;
      WQuantileSketch<bst_float, bst_float>::SummaryContainer summary;
      size_t num_available_cuts =
          std::min(size_t(entries_per_column), column_size);
      summary.Reserve(num_available_cuts);
      summary.MakeFromSorted(&entries[entries_per_column * icol],
                             num_available_cuts);

      sketches_[icol].PushSummary(summary);
    }
  }

  // Prevent copying/assigning/moving this as its internals can't be
  // assigned/copied/moved
  SketchContainer(const SketchContainer&) = delete;
  SketchContainer(const SketchContainer &&) = delete;
  SketchContainer &operator=(const SketchContainer &) = delete;
  SketchContainer &operator=(const SketchContainer &&) = delete;
};

// finds quantiles on the GPU
class GPUSketcher {
 public:
  GPUSketcher(int device, int max_bin, int gpu_nrows)
      : device_(device), max_bin_(max_bin), gpu_batch_nrows_(gpu_nrows), row_stride_(0) {}

  ~GPUSketcher() {  // NOLINT
    dh::safe_cuda(cudaSetDevice(device_));
  }

  void SketchBatch(const SparsePage &batch, const MetaInfo &info) {
    n_rows_ = batch.Size();

    Init(batch, info, gpu_batch_nrows_);
    Sketch(batch, info);
    ComputeRowStride();
  }

  /* Builds the sketches on the GPU for the dmatrix and returns the row stride
   * for the entire dataset */
  size_t Sketch(DMatrix *dmat, DenseCuts *hmat) {
    const MetaInfo& info = dmat->Info();

    row_stride_ = 0;
    sketch_container_.reset(new SketchContainer(max_bin_, dmat->Info().num_col_,dmat->Info().num_row_));
    for (const auto& batch : dmat->GetBatches<SparsePage>()) {
      this->SketchBatch(batch, info);
    }

    hmat->Init(&sketch_container_->sketches_, max_bin_, info.num_row_);
    return row_stride_;
  }

  // This needs to be public because of the __device__ lambda.
  void ComputeRowStride() {
    // Find the row stride for this batch
    auto row_iter = row_ptrs_.begin();
    // Functor for finding the maximum row size for this batch
    auto get_size = [=] __device__(size_t row) {
      return row_iter[row + 1] - row_iter[row];
    }; // NOLINT

    auto counting = thrust::make_counting_iterator(size_t(0));
    using TransformT = thrust::transform_iterator<decltype(get_size), decltype(counting), size_t>;
    TransformT row_size_iter = TransformT(counting, get_size);
    size_t batch_row_stride =
        thrust::reduce(row_size_iter, row_size_iter + n_rows_, 0, thrust::maximum<size_t>());
    row_stride_ = std::max(row_stride_, batch_row_stride);
  }

  // This needs to be public because of the __device__ lambda.
  void FindColumnCuts(size_t batch_nrows, size_t icol) {
    size_t tmp_size = tmp_storage_.size();
    // filter out NaNs in feature values
    auto fvalues_begin = fvalues_.data() + icol * gpu_batch_nrows_;
    cub::DeviceSelect::If(tmp_storage_.data().get(),
                          tmp_size,
                          fvalues_begin,
                          fvalues_cur_.data(),
                          num_elements_.begin(),
                          batch_nrows,
                          IsNotNaN());
    size_t nfvalues_cur = 0;
    thrust::copy_n(num_elements_.begin(), 1, &nfvalues_cur);

    // compute cumulative weights using a prefix scan
    if (has_weights_) {
      // filter out NaNs in weights;
      // since cub::DeviceSelect::If performs stable filtering,
      // the weights are stored in the correct positions
      auto feature_weights_begin = feature_weights_.data() + icol * gpu_batch_nrows_;
      cub::DeviceSelect::If(tmp_storage_.data().get(),
                            tmp_size,
                            feature_weights_begin,
                            weights_.data().get(),
                            num_elements_.begin(),
                            batch_nrows,
                            IsNotNaN());

      // sort the values and weights
      cub::DeviceRadixSort::SortPairs(tmp_storage_.data().get(),
                                      tmp_size,
                                      fvalues_cur_.data().get(),
                                      fvalues_begin.get(),
                                      weights_.data().get(),
                                      weights2_.data().get(),
                                      nfvalues_cur);

      // sum the weights to get cumulative weight values
      cub::DeviceScan::InclusiveSum(tmp_storage_.data().get(),
                                    tmp_size,
                                    weights2_.begin(),
                                    weights_.begin(),
                                    nfvalues_cur);
    } else {
      // sort the batch values
      cub::DeviceRadixSort::SortKeys(tmp_storage_.data().get(),
                                     tmp_size,
                                     fvalues_cur_.data().get(),
                                     fvalues_begin.get(),
                                     nfvalues_cur);

      // fill in cumulative weights with counting iterator
      thrust::copy_n(thrust::make_counting_iterator(1), nfvalues_cur, weights_.begin());
    }

    // remove repeated items and sum the weights across them;
    // non-negative weights are assumed
    cub::DeviceReduce::ReduceByKey(tmp_storage_.data().get(),
                                   tmp_size,
                                   fvalues_begin,
                                   fvalues_cur_.begin(),
                                   weights_.begin(),
                                   weights2_.begin(),
                                   num_elements_.begin(),
                                   thrust::maximum<bst_float>(),
                                   nfvalues_cur);
    size_t n_unique = 0;
    thrust::copy_n(num_elements_.begin(), 1, &n_unique);

    // extract cuts
    n_cuts_cur_[icol] = std::min(n_cuts_, n_unique);
    // if less elements than cuts: copy all elements with their weights
    if (n_cuts_ > n_unique) {
      float* weights2_ptr = weights2_.data().get();
      float* fvalues_ptr = fvalues_cur_.data().get();
      WQSketch::Entry* cuts_ptr = cuts_d_.data().get() + icol * n_cuts_;
      dh::LaunchN(device_, n_unique, [=]__device__(size_t i) {
        bst_float rmax = weights2_ptr[i];
        bst_float rmin = i > 0 ? weights2_ptr[i - 1] : 0;
        cuts_ptr[i] = WQSketch::Entry(rmin, rmax, rmax - rmin, fvalues_ptr[i]);
      });
    } else if (n_cuts_cur_[icol] > 0) {
      // if more elements than cuts: use binary search on cumulative weights
      uint32_t constexpr kBlockThreads = 256;
      uint32_t const kGrids = common::DivRoundUp(n_cuts_cur_[icol], kBlockThreads);
      dh::LaunchKernel {kGrids, kBlockThreads} (
          FindCutsK,
          cuts_d_.data().get() + icol * n_cuts_,
          fvalues_cur_.data().get(),
          weights2_.data().get(),
          n_unique,
          n_cuts_cur_[icol]);
      dh::safe_cuda(cudaGetLastError());  // NOLINT
    }
  }

 private:
  void Init(const SparsePage& row_batch, const MetaInfo& info, int gpu_batch_nrows) {
    num_cols_ = info.num_col_;
    has_weights_ = info.weights_.Size() > 0;

    // find the batch size
    if (gpu_batch_nrows == 0) {
      // By default, use no more than 1/16th of GPU memory
      gpu_batch_nrows_ = dh::TotalMemory(device_) / (16 * num_cols_ * sizeof(Entry));
    } else if (gpu_batch_nrows == -1) {
      gpu_batch_nrows_ = n_rows_;
    } else {
      gpu_batch_nrows_ = gpu_batch_nrows;
    }
    if (gpu_batch_nrows_ > n_rows_) {
      gpu_batch_nrows_ = n_rows_;
    }

    constexpr int kFactor = 8;
    double eps = 1.0 / (kFactor * max_bin_);
    size_t dummy_nlevel;
    WQSketch::LimitSizeLevel(gpu_batch_nrows_, eps, &dummy_nlevel, &n_cuts_);

    // allocate necessary GPU buffers
    dh::safe_cuda(cudaSetDevice(device_));

    entries_.resize(gpu_batch_nrows_ * num_cols_);
    fvalues_.resize(gpu_batch_nrows_ * num_cols_);
    fvalues_cur_.resize(gpu_batch_nrows_);
    cuts_d_.resize(n_cuts_ * num_cols_);
    cuts_h_.resize(n_cuts_ * num_cols_);
    weights_.resize(gpu_batch_nrows_);
    weights2_.resize(gpu_batch_nrows_);
    num_elements_.resize(1);

    if (has_weights_) {
      feature_weights_.resize(gpu_batch_nrows_ * num_cols_);
    }
    n_cuts_cur_.resize(num_cols_);

    // allocate storage for CUB algorithms; the size is the maximum of the sizes
    // required for various algorithm
    size_t tmp_size = 0, cur_tmp_size = 0;
    // size for sorting
    if (has_weights_) {
      cub::DeviceRadixSort::SortPairs(nullptr,
                                      cur_tmp_size,
                                      fvalues_cur_.data().get(),
                                      fvalues_.data().get(),
                                      weights_.data().get(),
                                      weights2_.data().get(),
                                      gpu_batch_nrows_);
    } else {
      cub::DeviceRadixSort::SortKeys(nullptr,
                                     cur_tmp_size,
                                     fvalues_cur_.data().get(),
                                     fvalues_.data().get(),
                                     gpu_batch_nrows_);
    }
    tmp_size = std::max(tmp_size, cur_tmp_size);
    // size for inclusive scan
    if (has_weights_) {
      cub::DeviceScan::InclusiveSum(nullptr,
                                    cur_tmp_size,
                                    weights2_.begin(),
                                    weights_.begin(),
                                    gpu_batch_nrows_);
      tmp_size = std::max(tmp_size, cur_tmp_size);
    }
    // size for reduction by key
    cub::DeviceReduce::ReduceByKey(nullptr,
                                   cur_tmp_size,
                                   fvalues_.begin(),
                                   fvalues_cur_.begin(),
                                   weights_.begin(),
                                   weights2_.begin(),
                                   num_elements_.begin(),
                                   thrust::maximum<bst_float>(),
                                   gpu_batch_nrows_);
    tmp_size = std::max(tmp_size, cur_tmp_size);
    // size for filtering
    cub::DeviceSelect::If(nullptr,
                          cur_tmp_size,
                          fvalues_.begin(),
                          fvalues_cur_.begin(),
                          num_elements_.begin(),
                          gpu_batch_nrows_,
                          IsNotNaN());
    tmp_size = std::max(tmp_size, cur_tmp_size);

    tmp_storage_.resize(tmp_size);
  }

  void Sketch(const SparsePage& row_batch, const MetaInfo& info) {
    // copy rows to the device
    dh::safe_cuda(cudaSetDevice(device_));
    const auto& offset_vec = row_batch.offset.HostVector();
    row_ptrs_.resize(n_rows_ + 1);
    thrust::copy(offset_vec.data(), offset_vec.data() + n_rows_ + 1, row_ptrs_.begin());
    size_t gpu_nbatches = common::DivRoundUp(n_rows_, gpu_batch_nrows_);
    for (size_t gpu_batch = 0; gpu_batch < gpu_nbatches; ++gpu_batch) {
      SketchBatch(row_batch, info, gpu_batch);
    }
  }

  void SketchBatch(const SparsePage& row_batch, const MetaInfo& info, size_t gpu_batch) {
    // compute start and end indices
    size_t batch_row_begin = gpu_batch * gpu_batch_nrows_;
    size_t batch_row_end = std::min((gpu_batch + 1) * gpu_batch_nrows_,
                                    static_cast<size_t>(n_rows_));
    size_t batch_nrows = batch_row_end - batch_row_begin;

    const auto& offset_vec = row_batch.offset.HostVector();
    const auto& data_vec = row_batch.data.HostVector();

    size_t n_entries = offset_vec[batch_row_end] - offset_vec[batch_row_begin];
    // copy the batch to the GPU
    dh::safe_cuda(cudaMemcpyAsync(entries_.data().get(),
                                  data_vec.data() + offset_vec[batch_row_begin],
                                  n_entries * sizeof(Entry),
                                  cudaMemcpyDefault));
    // copy the weights if necessary
    if (has_weights_) {
      const auto& weights_vec = info.weights_.HostVector();
      dh::safe_cuda(cudaMemcpyAsync(weights_.data().get(),
                                    weights_vec.data() + batch_row_begin,
                                    batch_nrows * sizeof(bst_float),
                                    cudaMemcpyDefault));
    }

    // unpack the features; also unpack weights if present
    thrust::fill(fvalues_.begin(), fvalues_.end(), NAN);
    if (has_weights_) {
      thrust::fill(feature_weights_.begin(), feature_weights_.end(), NAN);
    }

    dim3 block3(16, 64, 1);
    // NOTE: This will typically support ~ 4M features - 64K*64
    dim3 grid3(common::DivRoundUp(batch_nrows, block3.x),
               common::DivRoundUp(num_cols_, block3.y), 1);
    dh::LaunchKernel {grid3, block3} (
        UnpackFeaturesK,
        fvalues_.data().get(),
        has_weights_ ? feature_weights_.data().get() : nullptr,
        row_ptrs_.data().get() + batch_row_begin,
        has_weights_ ? weights_.data().get() : nullptr, entries_.data().get(),
        gpu_batch_nrows_,
        offset_vec[batch_row_begin],
        batch_nrows);

    for (int icol = 0; icol < num_cols_; ++icol) {
      FindColumnCuts(batch_nrows, icol);
    }

    // add cuts into sketches
    thrust::copy(cuts_d_.begin(), cuts_d_.end(), cuts_h_.begin());
#pragma omp parallel for default(none) schedule(static) \
if (num_cols_ > SketchContainer::kOmpNumColsParallelizeLimit) // NOLINT
    for (int icol = 0; icol < num_cols_; ++icol) {
      WQSketch::SummaryContainer summary;
      summary.Reserve(n_cuts_);
      summary.MakeFromSorted(&cuts_h_[n_cuts_ * icol], n_cuts_cur_[icol]);

      sketch_container_->sketches_[icol].PushSummary(summary);

    }
  }

  const int device_;
  const int max_bin_;
  int gpu_batch_nrows_;
  size_t row_stride_;
  std::unique_ptr<SketchContainer> sketch_container_;

  bst_uint n_rows_{};
  int num_cols_{0};
  size_t n_cuts_{0};
  bool has_weights_{false};

  dh::device_vector<size_t> row_ptrs_{};
  dh::device_vector<Entry> entries_{};
  dh::device_vector<bst_float> fvalues_{};
  dh::device_vector<bst_float> feature_weights_{};
  dh::device_vector<bst_float> fvalues_cur_{};
  dh::device_vector<WQSketch::Entry> cuts_d_{};
  thrust::host_vector<WQSketch::Entry> cuts_h_{};
  dh::device_vector<bst_float> weights_{};
  dh::device_vector<bst_float> weights2_{};
  std::vector<size_t> n_cuts_cur_{};
  dh::device_vector<size_t> num_elements_{};
  dh::device_vector<char> tmp_storage_{};
};

size_t DeviceSketchOld(int device,
                    int max_bin,
                    int gpu_batch_nrows,
                    DMatrix* dmat,
                    HistogramCuts* hmat) {
  GPUSketcher sketcher(device, max_bin, gpu_batch_nrows);
  // We only need to return the result in HistogramCuts container, so it is safe to
  // use a pointer of local HistogramCutsDense
  DenseCuts dense_cuts(hmat);
  auto res = sketcher.Sketch(dmat, &dense_cuts);
  return res;
}

struct EntryCompareOp {
  __device__ bool operator()(const Entry& a, const Entry& b) {
    if (a.index == b.index) {
      return a.fvalue < b.fvalue;
    }
    return a.index < b.index;
  }
};

/**
 * \brief Extracts the cuts from sorted data.
 *
 * \param device                The device.
 * \param cuts                  Output cuts
 * \param num_cuts_per_feature  Number of cuts per feature.
 * \param sorted_data           Sorted entries in segments of columns
 * \param column_sizes_scan     Describes the boundaries of column segments in sorted data
 */
void ExtractCuts(int device, Span<SketchEntry> cuts,
                 size_t num_cuts_per_feature, Span<Entry> sorted_data,
                 Span<size_t> column_sizes_scan) {
  dh::LaunchN(device, cuts.size(), [=] __device__(size_t idx) {
    // Each thread is responsible for obtaining one cut from the sorted input
    size_t column_idx = idx / num_cuts_per_feature;
    size_t column_size =
        column_sizes_scan[column_idx + 1] - column_sizes_scan[column_idx];
    size_t num_available_cuts =
        std::min(size_t(num_cuts_per_feature), column_size);
    size_t cut_idx = idx % num_cuts_per_feature;
    if (cut_idx >= num_available_cuts) return;

    Span<Entry> column_entries =
        sorted_data.subspan(column_sizes_scan[column_idx], column_size);

    size_t rank = (column_entries.size() * cut_idx) / num_available_cuts;
    auto value = column_entries[rank].fvalue;
    cuts[idx] = SketchEntry(rank, rank + 1, 1, value);
  });
}

void ProcessBatch(int device, const SparsePage& page, size_t begin, size_t end,
  SketchContainer *sketch_container, int num_cuts, size_t num_columns) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  const auto & host_data = page.data.ConstHostVector();
  dh::device_vector<Entry> sorted_entries( host_data.begin()+begin,host_data.begin()+end);
  thrust::sort(thrust::cuda::par(alloc), sorted_entries.begin(),
    sorted_entries.end(), EntryCompareOp());
  dh::caching_device_vector<size_t> column_sizes_scan(num_columns + 1,
                                                      0);
  auto d_column_sizes_scan = column_sizes_scan.data().get();
  auto d_sorted_entries = sorted_entries.data().get();
  dh::LaunchN(device, sorted_entries.size(), [=] __device__(size_t idx)
  {
    auto& e = d_sorted_entries[idx];
      atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
        &d_column_sizes_scan[e.index]),
        static_cast<unsigned long long>(1));  // NOLINT
  });
  thrust::exclusive_scan(thrust::cuda::par(alloc), column_sizes_scan.begin(),
                         column_sizes_scan.end(), column_sizes_scan.begin());
  thrust::host_vector<size_t > host_column_sizes_scan(column_sizes_scan);

  dh::caching_device_vector<SketchEntry> cuts(num_columns * num_cuts);
  ExtractCuts(device, {cuts.data().get(), cuts.size()}, num_cuts,
              {sorted_entries.data().get(), sorted_entries.size()},
              {column_sizes_scan.data().get(), column_sizes_scan.size()});

  // add cuts into sketches
  thrust::host_vector<SketchEntry> host_cuts(cuts);
  sketch_container->Push(num_cuts, host_cuts, host_column_sizes_scan);
}

HistogramCuts DeviceSketch(int device, DMatrix* dmat, int max_bins,
                           size_t sketch_batch_num_elements) {

  HistogramCuts cuts;
  DenseCuts dense_cuts(&cuts);
  SketchContainer sketch_container(max_bins, dmat->Info().num_col_,
                                   dmat->Info().num_row_);

  constexpr int kFactor = 8;
  double eps = 1.0 / (kFactor * max_bins);
  size_t dummy_nlevel;
  size_t num_cuts;
  WQuantileSketch<bst_float, bst_float>::LimitSizeLevel(
       dmat->Info().num_row_, eps, &dummy_nlevel, &num_cuts);
  num_cuts = std::min(num_cuts, dmat->Info().num_row_);
  if (sketch_batch_num_elements == 0) {
    sketch_batch_num_elements = dmat->Info().num_nonzero_;
  }
  for (const auto &batch : dmat->GetBatches<SparsePage>()) {
    size_t batch_nnz = batch.data.ConstHostVector().size();
    for (auto begin = 0ull; begin < batch_nnz; begin += sketch_batch_num_elements) {
      size_t end = std::min(batch_nnz, begin + sketch_batch_num_elements);
      ProcessBatch(device, batch, begin, end, &sketch_container, num_cuts,
                   dmat->Info().num_col_);
    }
  }

  dense_cuts.Init(&sketch_container.sketches_,max_bins, dmat->Info().num_row_);
  return cuts;
}

struct IsValidFunctor : public thrust::unary_function<Entry, bool> {
  explicit IsValidFunctor(float missing) : missing(missing) {}

  float missing;
  __device__ bool operator()(const data::COOTuple& e) const {
    if (common::CheckNAN(e.value) || e.value == missing) {
      return false;
    }
    return true;
  }
  __device__ bool operator()(const Entry& e) const {
    if (common::CheckNAN(e.fvalue) || e.fvalue == missing) {
      return false;
    }
    return true;
  }
};

template <typename ReturnT, typename IterT, typename FuncT>
thrust::transform_iterator<FuncT, IterT, ReturnT> MakeTransformIterator(
    IterT iter, FuncT func) {
  return thrust::transform_iterator<FuncT, IterT, ReturnT>(iter, func);
}

template <typename AdapterT>
void ProcessBatch(AdapterT *adapter, size_t begin, size_t end, float missing,
  SketchContainer *sketch_container, int num_cuts) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  adapter->BeforeFirst();
  adapter->Next();
  auto& batch = adapter->Value();
  // Enforce single batch
  CHECK(!adapter->Next());

  auto batch_iter = MakeTransformIterator<data::COOTuple>(thrust::make_counting_iterator(0llu), [=] __device__(size_t idx) {
    return batch.GetElement(idx);
  });
  auto entry_iter = MakeTransformIterator<Entry>(
    thrust::make_counting_iterator(0llu), [=] __device__(size_t idx) {
    return Entry(batch.GetElement(idx).column_idx,
      batch.GetElement(idx).value);
  });

  // Work out how many valid entries we have in each column
  dh::caching_device_vector<size_t> column_sizes_scan(adapter->NumColumns() + 1,
    0);
  auto d_column_sizes_scan = column_sizes_scan.data().get();
  IsValidFunctor is_valid(missing);
  dh::LaunchN(adapter->DeviceIdx(), end - begin, [=] __device__(size_t idx)
  {
    auto &e = batch_iter[begin + idx];
    if (is_valid(e)) {
      atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
        &d_column_sizes_scan[e.column_idx]),
        static_cast<unsigned long long>(1));  // NOLINT
    }
  });
  thrust::exclusive_scan(thrust::cuda::par(alloc), column_sizes_scan.begin(),
    column_sizes_scan.end(), column_sizes_scan.begin());
  thrust::host_vector<size_t > host_column_sizes_scan(column_sizes_scan);
  size_t num_valid = host_column_sizes_scan.back();

  // Copy current subset of valid elements into temporary storage and sort
  thrust::device_vector<Entry> sorted_entries(num_valid);
  thrust::copy_if(thrust::cuda::par(alloc), entry_iter + begin, entry_iter + end, sorted_entries.begin(), is_valid);
  thrust::sort(thrust::cuda::par(alloc), sorted_entries.begin(),
    sorted_entries.end(), EntryCompareOp());

  // Extract the cuts from all columns concurrently
  dh::caching_device_vector<SketchEntry> cuts(adapter->NumColumns() * num_cuts);
  ExtractCuts(adapter->DeviceIdx(), {cuts.data().get(), cuts.size()}, num_cuts,
    {sorted_entries.data().get(), sorted_entries.size()},
    {column_sizes_scan.data().get(), column_sizes_scan.size()});

  // Push cuts into sketches stored in host memory
  thrust::host_vector<SketchEntry> host_cuts(cuts);
  sketch_container->Push(num_cuts, host_cuts, host_column_sizes_scan);
}


template <typename AdapterT>
HistogramCuts AdapterDeviceSketch(AdapterT *adapter, int num_bins, float missing, size_t sketch_batch_num_elements) {

  CHECK(adapter->NumRows() != data::kAdapterUnknownSize);
  CHECK(adapter->NumColumns() != data::kAdapterUnknownSize);

  adapter->BeforeFirst();
  adapter->Next();
  auto& batch = adapter->Value();

  // Enforce single batch
  CHECK(!adapter->Next());

  HistogramCuts cuts;
  DenseCuts dense_cuts(&cuts);
  SketchContainer sketch_container(num_bins, adapter->NumColumns(),
                                   adapter->NumRows());

  constexpr int kFactor = 8;
  double eps = 1.0 / (kFactor * num_bins);
  size_t dummy_nlevel;
  size_t num_cuts;
  WQuantileSketch<bst_float, bst_float>::LimitSizeLevel(
       adapter->NumRows(), eps, &dummy_nlevel, &num_cuts);
  num_cuts = std::min(num_cuts, adapter->NumRows());
  if (sketch_batch_num_elements == 0) {
    sketch_batch_num_elements = batch.Size();
  }
  for (auto begin = 0ull; begin < batch.Size(); begin += sketch_batch_num_elements) {
    size_t end = std::min(batch.Size(), begin + sketch_batch_num_elements);
    ProcessBatch(adapter, begin, end, missing, &sketch_container, num_cuts);
  }

  dense_cuts.Init(&sketch_container.sketches_, num_bins, adapter->NumRows());
  return cuts;
}

template HistogramCuts AdapterDeviceSketch(data::CudfAdapter* adapter,
                                           int num_bins, float missing,
                                           size_t sketch_batch_size);
template HistogramCuts AdapterDeviceSketch(data::CupyAdapter* adapter,
                                           int num_bins, float missing,
                                           size_t sketch_batch_size);
}  // namespace common
}  // namespace xgboost
