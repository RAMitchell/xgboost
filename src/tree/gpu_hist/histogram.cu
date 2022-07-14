/*!
 * Copyright 2020-2021 by XGBoost Contributors
 */
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <ctgmath>
#include <limits>

#include "xgboost/base.h"
#include "row_partitioner.cuh"

#include "histogram.cuh"

#include "../../data/ellpack_page.cuh"
#include "../../common/device_helpers.cuh"

namespace xgboost {
namespace tree {
// Following 2 functions are slightly modified version of fbcuda.

/* \brief Constructs a rounding factor used to truncate elements in a sum such that the
   sum of the truncated elements is the same no matter what the order of the sum is.

 * Algorithm 5: Reproducible Sequential Sum in 'Fast Reproducible Floating-Point
 * Summation' by Demmel and Nguyen

 * In algorithm 5 the bound is calculated as $max(|v_i|) * n$.  Here we use the bound
 *
 * \begin{equation}
 *   max( fl(\sum^{V}_{v_i>0}{v_i}), fl(\sum^{V}_{v_i<0}|v_i|) )
 * \end{equation}
 *
 * to avoid outliers, as the full reduction is reproducible on GPU with reduction tree.
 */
template <typename T>
T CreateRoundingFactor(T max_abs, int n) {
  T delta = max_abs / (static_cast<T>(1.0) - 2 * n * std::numeric_limits<T>::epsilon());

  // Calculate ceil(log_2(delta)).
  // frexpf() calculates exp and returns `x` such that
  // delta = x * 2^exp, where `x` in (-1.0, -0.5] U [0.5, 1).
  // Because |x| < 1, exp is exactly ceil(log_2(delta)).
  int exp;
  std::frexp(delta, &exp);

  // return M = 2 ^ ceil(log_2(delta))
  return std::ldexp(static_cast<T>(1.0), exp);
}

namespace {
struct Pair {
  GradientPair first;
  GradientPair second;
};
__host__ XGBOOST_DEV_INLINE Pair operator+(Pair const& lhs, Pair const& rhs) {
  return {lhs.first + rhs.first, lhs.second + rhs.second};
}
}  // anonymous namespace

struct Clip : public thrust::unary_function<GradientPair, Pair> {
  static XGBOOST_DEV_INLINE float Pclip(float v) {
    return v > 0 ? v : 0;
  }
  static XGBOOST_DEV_INLINE float Nclip(float v) {
    return v < 0 ? abs(v) : 0;
  }

  XGBOOST_DEV_INLINE Pair operator()(GradientPair x) const {
    auto pg = Pclip(x.GetGrad());
    auto ph = Pclip(x.GetHess());

    auto ng = Nclip(x.GetGrad());
    auto nh = Nclip(x.GetHess());

    return { GradientPair{ pg, ph }, GradientPair{ ng, nh } };
  }
};

template <typename GradientSumT>
HistRounding<GradientSumT> CreateRoundingFactor(common::Span<GradientPair const> gpair) {
  using T = typename GradientSumT::ValueT;
  dh::XGBCachingDeviceAllocator<char> alloc;

  thrust::device_ptr<GradientPair const> gpair_beg {gpair.data()};
  thrust::device_ptr<GradientPair const> gpair_end {gpair.data() + gpair.size()};
  auto beg = thrust::make_transform_iterator(gpair_beg, Clip());
  auto end = thrust::make_transform_iterator(gpair_end, Clip());
  Pair p = dh::Reduce(thrust::cuda::par(alloc), beg, end, Pair{}, thrust::plus<Pair>{});
  GradientPair positive_sum {p.first}, negative_sum {p.second};

  auto histogram_rounding = GradientSumT {
    CreateRoundingFactor<T>(std::max(positive_sum.GetGrad(), negative_sum.GetGrad()),
                            gpair.size()),
    CreateRoundingFactor<T>(std::max(positive_sum.GetHess(), negative_sum.GetHess()),
                            gpair.size()) };

  using IntT = typename HistRounding<GradientSumT>::SharedSumT::ValueT;

  /**
   * Factor for converting gradients from fixed-point to floating-point.
   */
  GradientSumT to_floating_point =
      histogram_rounding /
      T(IntT(1) << (sizeof(typename GradientSumT::ValueT) * 8 -
                    2));  // keep 1 for sign bit
  /**
   * Factor for converting gradients from floating-point to fixed-point. For
   * f64:
   *
   *   Precision = 64 - 1 - log2(rounding)
   *
   * rounding is calcuated as exp(m), see the rounding factor calcuation for
   * details.
   */
  GradientSumT to_fixed_point = GradientSumT(
      T(1) / to_floating_point.GetGrad(), T(1) / to_floating_point.GetHess());

  return {histogram_rounding, to_fixed_point, to_floating_point};
}

template HistRounding<GradientPairPrecise>
CreateRoundingFactor(common::Span<GradientPair const> gpair);
template HistRounding<GradientPair>
CreateRoundingFactor(common::Span<GradientPair const> gpair);
// Atomic add function for gradients
template <typename OutputGradientT, typename InputGradientT>
XGBOOST_DEV_INLINE void AddGpair(OutputGradientT* dest,
                                       const InputGradientT& gpair) {
  auto dst_ptr = reinterpret_cast<typename OutputGradientT::ValueT*>(dest);

  *dst_ptr += static_cast<typename OutputGradientT::ValueT>(gpair.GetGrad());
  *(dst_ptr + 1) += static_cast<typename OutputGradientT::ValueT>(gpair.GetHess());
}

template <typename GradientSumT, int kBlockThreads>
class HistogramAgent {
  using SharedSumT = typename HistRounding<GradientSumT>::SharedSumT;
  SharedSumT* smem_arr;
  GradientSumT* d_node_hist;
  dh::LDGIterator<const RowPartitioner::RowIndexT> d_ridx;const GradientPair* d_gpair;
  const FeatureGroup group;
  const EllpackDeviceAccessor& matrix;
  int feature_stride;
  size_t n_elements;
  const int32_t* d_gidx;
  const HistRounding<GradientSumT>& rounding;

 public:
  __device__ HistogramAgent(SharedSumT* smem_arr, GradientSumT* __restrict__ d_node_hist,
                            const FeatureGroup& group, const EllpackDeviceAccessor& matrix,
                            common::Span<const RowPartitioner::RowIndexT> d_ridx,
                            const int32_t* __restrict__ d_gidx,
                            const HistRounding<GradientSumT>& rounding, const GradientPair* d_gpair)
      : smem_arr(smem_arr),
        d_node_hist(d_node_hist),
        d_ridx(d_ridx.data()),
        group(group),
        matrix(matrix),
        feature_stride(matrix.is_dense ? group.num_features : matrix.row_stride),
        n_elements(feature_stride * d_ridx.size()),
        d_gidx(d_gidx),
        rounding(rounding),
        d_gpair(d_gpair) {}
  template <int kItemsPerTile>
  __device__ void ProcessTileShared(std::size_t offset) {
    std::size_t idx = offset + threadIdx.x;
    if (idx >= n_elements) return;
    int ridx = d_ridx[idx / feature_stride];
    int gidx = d_gidx[ridx * matrix.row_stride + group.start_feature + idx % feature_stride];
    if (matrix.is_dense || gidx != matrix.NumBins()) {
      // If we are not using shared memory, accumulate the values directly into
      // global memory
      gidx = gidx - group.start_bin;
      auto adjusted = rounding.ToFixedPoint(d_gpair[ridx]);
      dh::AtomicAddGpair(smem_arr + gidx, adjusted);
      // AddGpair(smem_arr + gidx, adjusted);
    }
  }
  __device__ void BuildHistogramWithShared() {
    dh::BlockFill(smem_arr, group.num_bins, SharedSumT());
    __syncthreads();

    constexpr int kItemsPerTile = 1;
    constexpr int kTileSize = kItemsPerTile * kBlockThreads;
    std::size_t offset = blockIdx.x * kTileSize;
    while (offset < n_elements) {
      ProcessTileShared<kItemsPerTile>(offset);
      offset += kTileSize * gridDim.x;
    }

    // Write shared memory back to global memory
    __syncthreads();
    for (auto i : dh::BlockStrideRange(0, group.num_bins)) {
      auto truncated = rounding.ToFloatingPoint(smem_arr[i]);
      dh::AtomicAddGpair(d_node_hist + group.start_bin + i, truncated);
    }
  }

  __device__ void BuildHistogramWithGlobal() {
    for (auto idx : dh::GridStrideRange(static_cast<size_t>(0), n_elements)) {
      int ridx = d_ridx[idx / feature_stride];
      int gidx = d_gidx[ridx * matrix.row_stride + group.start_feature + idx % feature_stride];
      // int gidx = matrix.gidx_iter[ridx * matrix.row_stride + group.start_feature +
      // idx % feature_stride];
      if (matrix.is_dense || gidx != matrix.NumBins()) {
        // If we are not using shared memory, accumulate the values directly into
        // global memory
          GradientSumT truncated{
              TruncateWithRoundingFactor<GradientSumT::ValueT>(rounding.rounding.GetGrad(), d_gpair[ridx].GetGrad()),
              TruncateWithRoundingFactor<GradientSumT::ValueT>(rounding.rounding.GetHess(), d_gpair[ridx].GetHess()),
          };
          dh::AtomicAddGpair(d_node_hist + gidx, truncated);
      }
    }
  }
};

  template <typename GradientSumT, bool use_shared_memory_histograms,int kBlockThreads>
  __global__ void __launch_bounds__(kBlockThreads) SharedMemHistKernel(EllpackDeviceAccessor matrix,
                                      FeatureGroupsAccessor feature_groups,
                                      common::Span<const RowPartitioner::RowIndexT> d_ridx,
                                      GradientSumT* __restrict__ d_node_hist,
                                      const GradientPair* __restrict__ d_gpair,
                                      HistRounding<GradientSumT> const rounding,
                                      const int32_t* __restrict__ d_gidx) {
    using SharedSumT = typename HistRounding<GradientSumT>::SharedSumT;
    using T = typename GradientSumT::ValueT;

    extern __shared__ char smem[];
    const FeatureGroup group = feature_groups[blockIdx.y];
    SharedSumT* smem_arr = reinterpret_cast<SharedSumT*>(smem);
    auto agent=HistogramAgent<GradientSumT,kBlockThreads>(smem_arr, d_node_hist, group, matrix, d_ridx,d_gidx, rounding,d_gpair);
    if (use_shared_memory_histograms) {
      agent.BuildHistogramWithShared();
    } else {
      agent.BuildHistogramWithGlobal();
    }
  }

  template <typename GradientSumT>
  void BuildGradientHistogram(EllpackDeviceAccessor const& matrix,
                              FeatureGroupsAccessor const& feature_groups,
                              common::Span<GradientPair const> gpair,
                              common::Span<const uint32_t> d_ridx,
                              common::Span<GradientSumT> histogram,
                              HistRounding<GradientSumT> rounding, bool force_global_memory) {
    thrust::device_vector<int32_t> gidx(matrix.row_stride * matrix.n_rows);
    auto d_gidx = gidx.data().get();

    dh::LaunchN(gidx.size(), [=] __device__(size_t i) { d_gidx[i] = matrix.gidx_iter[i]; });

    // decide whether to use shared memory
    int device = 0;
    dh::safe_cuda(cudaGetDevice(&device));
    // opt into maximum shared memory for the kernel if necessary
    size_t max_shared_memory = dh::MaxSharedMemoryOptin(device);

    size_t smem_size =
        sizeof(typename HistRounding<GradientSumT>::SharedSumT) * feature_groups.max_group_bins;
    bool shared = !force_global_memory && smem_size <= max_shared_memory;
    smem_size = shared ? smem_size : 0;

    constexpr int kBlockThreads = 1024;
    auto runit = [&](auto kernel) {
      if (shared) {
        dh::safe_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           max_shared_memory));
      }

      // determine the launch configuration

      int num_groups = feature_groups.NumGroups();
      int n_mps = 0;
      dh::safe_cuda(cudaDeviceGetAttribute(&n_mps, cudaDevAttrMultiProcessorCount, device));
      int n_blocks_per_mp = 0;
      dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_blocks_per_mp, kernel,
                                                                  kBlockThreads , smem_size));
      unsigned grid_size = n_blocks_per_mp * n_mps;

      // TODO(canonizer): This is really a hack, find a better way to distribute
      // the data among thread blocks. The intention is to generate enough thread
      // blocks to fill the GPU, but avoid having too many thread blocks, as this
      // is less efficient when the number of rows is low. At least one thread
      // block per feature group is required. The number of thread blocks:
      // - for num_groups <= num_groups_threshold, around  grid_size * num_groups
      // - for num_groups_threshold <= num_groups <= num_groups_threshold *
      // grid_size,
      //     around grid_size * num_groups_threshold
      // - for num_groups_threshold * grid_size <= num_groups, around num_groups
      int num_groups_threshold = 4;
      grid_size =
          common::DivRoundUp(grid_size, common::DivRoundUp(num_groups, num_groups_threshold));

      using T = typename GradientSumT::ValueT;
      dh::LaunchKernel{dim3(grid_size, num_groups), static_cast<uint32_t>(kBlockThreads ),
                       smem_size}(kernel, matrix, feature_groups, d_ridx, histogram.data(),
                                  gpair.data(), rounding, d_gidx);
    };

    if (shared) {
      runit(SharedMemHistKernel<GradientSumT, true,kBlockThreads>);
    } else {
      runit(SharedMemHistKernel<GradientSumT, false,kBlockThreads>);
    }

    dh::safe_cuda(cudaDeviceSynchronize());
    dh::safe_cuda(cudaGetLastError());
  }

  template void BuildGradientHistogram<GradientPairPrecise>(
      EllpackDeviceAccessor const& matrix, FeatureGroupsAccessor const& feature_groups,
      common::Span<GradientPair const> gpair, common::Span<const uint32_t> ridx,
      common::Span<GradientPairPrecise> histogram, HistRounding<GradientPairPrecise> rounding,
      bool force_global_memory);

}  // namespace tree
}  // namespace xgboost
