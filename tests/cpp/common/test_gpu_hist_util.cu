#include <dmlc/filesystem.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>


#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include "xgboost/c_api.h"

#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/hist_util.h"

#include "../helpers.h"
#include <xgboost/data.h>
#include "../../../src/data/device_adapter.cuh"
#include "../data/test_array_interface.h"
#include "../../../src/common/math.h"

namespace xgboost {
namespace common {

void TestDeviceSketch(bool use_external_memory) {
  // create the data
  int nrows = 10001;
  std::shared_ptr<xgboost::DMatrix> *dmat = nullptr;

  size_t num_cols = 1;
  dmlc::TemporaryDirectory tmpdir;
  std::string file = tmpdir.path + "/big.libsvm";
  if (use_external_memory) {
    auto sp_dmat = CreateSparsePageDMatrix(nrows * 3, 128UL, file); // 3 entries/row
    dmat = new std::shared_ptr<xgboost::DMatrix>(std::move(sp_dmat));
    num_cols = 5;
  } else {
     std::vector<float> test_data(nrows);
     auto count_iter = thrust::make_counting_iterator(0);
     // fill in reverse order
     std::copy(count_iter, count_iter + nrows, test_data.rbegin());

     // create the DMatrix
     DMatrixHandle dmat_handle;
     XGDMatrixCreateFromMat(test_data.data(), nrows, 1, -1,
                            &dmat_handle);
     dmat = static_cast<std::shared_ptr<xgboost::DMatrix> *>(dmat_handle);
  }

  int device{0};
  int max_bin{20};
  int gpu_batch_nrows{0};

  // find quantiles on the CPU
  HistogramCuts hmat_cpu;
  hmat_cpu.Build((*dmat).get(), max_bin);

  // find the cuts on the GPU
  HistogramCuts hmat_gpu;
  size_t row_stride = DeviceSketch(device, max_bin, gpu_batch_nrows, dmat->get(), &hmat_gpu);

  // compare the row stride with the one obtained from the dmatrix
  bst_row_t expected_row_stride = 0;
  for (const auto &batch : dmat->get()->GetBatches<xgboost::SparsePage>()) {
    const auto &offset_vec = batch.offset.ConstHostVector();
    for (int i = 1; i <= offset_vec.size() -1; ++i) {
      expected_row_stride = std::max(expected_row_stride, offset_vec[i] - offset_vec[i-1]);
    }
  }

  ASSERT_EQ(expected_row_stride, row_stride);

  // compare the cuts
  double eps = 1e-2;
  ASSERT_EQ(hmat_gpu.MinValues().size(), num_cols);
  ASSERT_EQ(hmat_gpu.Ptrs().size(), num_cols + 1);
  ASSERT_EQ(hmat_gpu.Values().size(), hmat_cpu.Values().size());
  ASSERT_LT(fabs(hmat_cpu.MinValues()[0] - hmat_gpu.MinValues()[0]), eps * nrows);
  for (int i = 0; i < hmat_gpu.Values().size(); ++i) {
    ASSERT_LT(fabs(hmat_cpu.Values()[i] - hmat_gpu.Values()[i]), eps * nrows);
  }

  delete dmat;
}

TEST(gpu_hist_util, DeviceSketch) {
  TestDeviceSketch(false);
}

TEST(gpu_hist_util, DeviceSketch_ExternalMemory) {
  TestDeviceSketch(true); }

struct SketchContainer {
  std::vector<DenseCuts::WXQSketch> sketches_;  // NOLINT
  std::vector<std::mutex> col_locks_; // NOLINT
  static constexpr int kOmpNumColsParallelizeLimit = 1000;

  SketchContainer(int max_bin,size_t num_columns,size_t num_rows ) : col_locks_(num_columns) {
    // Initialize Sketches for this dmatrix
    sketches_.resize(num_columns);
#pragma omp parallel for default(none) shared(max_bin) schedule(static) \
if (num_columns> kOmpNumColsParallelizeLimit)  // NOLINT
    for (int icol = 0; icol < num_columns; ++icol) {  // NOLINT
      sketches_[icol].Init(num_rows, 1.0 / (8 * max_bin));
    }
  }

  // Prevent copying/assigning/moving this as its internals can't be assigned/copied/moved
  SketchContainer(const SketchContainer &) = delete;
  SketchContainer(const SketchContainer &&) = delete;
  SketchContainer &operator=(const SketchContainer &) = delete;
  SketchContainer &operator=(const SketchContainer &&) = delete;
};


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
void AdapterDeviceSketch(AdapterT *adapter, int num_bins, float missing) {
  CHECK(adapter->NumRows() != data::kAdapterUnknownSize);
  CHECK(adapter->NumColumns() != data::kAdapterUnknownSize);

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
  size_t valid_elements = thrust::count_if(
      batch_iter, batch_iter + batch.Size(), IsValidFunctor(missing));
  thrust::device_vector<Entry> tmp(valid_elements);

  thrust::copy_if(entry_iter, entry_iter + batch.Size(), tmp.begin(),
                  IsValidFunctor(missing));
  thrust::device_vector<size_t> column_sizes_scan(adapter->NumColumns() + 1);
  auto d_column_sizes_scan = column_sizes_scan.data().get();
  auto d_tmp = tmp.data().get();
  dh::LaunchN(adapter->DeviceIdx(),tmp.size(),[=] __device__ (size_t idx)
  {
    auto &e = d_tmp[idx];
    atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
      &d_column_sizes_scan[e.index]),
      static_cast<unsigned long long>(1));  // NOLINT
  });
  thrust::exclusive_scan(column_sizes_scan.begin(), column_sizes_scan.end(),column_sizes_scan.begin());

  thrust::sort(tmp.begin(), tmp.end(),
               [=] __device__(const Entry &a, const Entry &b) {
                 if (a.index == b.index) {
                   return a.fvalue < b.fvalue;
                 }
                 return a.index < b.index;
               });

  SketchContainer sketch_container(num_bins, adapter->NumColumns(),
                                   adapter->NumRows());
}

TEST(gpu_hist_util, AdapterDeviceSketch)
{
  int rows = 50;
  int cols = 10;
  float missing = 0.0;
  thrust::device_vector< float> data(rows*cols);
  auto json_array_interface = Generate2dArrayInterface(rows, cols, "<f4", &data);
  std::stringstream ss;
  Json::Dump(json_array_interface, &ss);
  std::string str = ss.str();
  data::CupyAdapter adapter(str);

  AdapterDeviceSketch(&adapter, 4, missing);
}
}  // namespace common
}  // namespace xgboost
