
/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <xgboost/base.h>
#include "../../../src/common/device_helpers.cuh"
#include "../helpers.h"
#include "gtest/gtest.h"


using xgboost::common::Span;

void CreateTestData(xgboost::bst_uint num_rows, int max_row_size,
                    thrust::host_vector<int> *row_ptr,
                    thrust::host_vector<xgboost::bst_uint> *rows) {
  row_ptr->resize(num_rows + 1);
  int sum = 0;
  for (xgboost::bst_uint i = 0; i <= num_rows; i++) {
    (*row_ptr)[i] = sum;
    sum += rand() % max_row_size;  // NOLINT

    if (i < num_rows) {
      for (int j = (*row_ptr)[i]; j < sum; j++) {
        (*rows).push_back(i);
      }
    }
  }
}

void TestLbs() {
  srand(17);
  dh::CubMemory temp_memory;

  std::vector<int> test_rows = {4, 100, 1000};
  std::vector<int> test_max_row_sizes = {4, 100, 1300};

  for (auto num_rows : test_rows) {
    for (auto max_row_size : test_max_row_sizes) {
      thrust::host_vector<int> h_row_ptr;
      thrust::host_vector<xgboost::bst_uint> h_rows;
      CreateTestData(num_rows, max_row_size, &h_row_ptr, &h_rows);
      thrust::device_vector<size_t> row_ptr = h_row_ptr;
      thrust::device_vector<int> output_row(h_rows.size());
      auto d_output_row = output_row.data();

      dh::TransformLbs(0, &temp_memory, h_rows.size(), dh::Raw(row_ptr),
                       row_ptr.size() - 1, false,
                       [=] __device__(size_t idx, size_t ridx) {
                         d_output_row[idx] = ridx;
                       });

      dh::safe_cuda(cudaDeviceSynchronize());
      ASSERT_TRUE(h_rows == output_row);
    }
  }
}

TEST(cub_lbs, Test) {
  TestLbs();
}

TEST(sumReduce, Test) {
  thrust::device_vector<float> data(100, 1.0f);
  dh::CubMemory temp;
  auto sum = dh::SumReduction(temp, dh::Raw(data), data.size());
  ASSERT_NEAR(sum, 100.0f, 1e-5);
}

void TestAllocator() {
  int n = 10;
  Span<float> a;
  Span<int> b;
  Span<size_t> c;
  dh::BulkAllocator ba;
  ba.Allocate(0, &a, n, &b, n, &c, n);

  // Should be no illegal memory accesses
  dh::LaunchN(0, n, [=] __device__(size_t idx) { c[idx] = a[idx] + b[idx]; });

  dh::safe_cuda(cudaDeviceSynchronize());
}

// Define the test in a function so we can use device lambda
TEST(bulkAllocator, Test) { TestAllocator(); }

template <typename PointerT, typename ComparisonOpT>
__device__ void CompareExchange(PointerT a, PointerT b, ComparisonOpT op) {
  using ValueT = typename std::iterator_traits<PointerT>::value_type;
  if (op(*a, *b)) {
    ValueT temp = *a;
    *a = *b;
    *b = temp;
  }
}

template <typename PointerT, typename ComparisonOpT>
__device__ void CompareExchange(PointerT a, PointerT b, bool a_valid,
                                bool b_valid, ComparisonOpT op) {
  if (a_valid && b_valid) {
    CompareExchange(a, b, op);
  }
}

template <typename T, typename ValueT=std::iterator_traits<T>::value_type>
class BitonicSequence {
 public:
  XGBOOST_DEVICE BitonicSequence(const T &iter, size_t n,  int phase,int step)
      : iter(iter), n(n),  phase(phase),step(step) {}
  // Number of independent GPU threads needed at this step
  size_t NumThreadsRequired(int multistep_degree) {
    size_t n_next_power = std::pow(2, std::ceil(std::log2(n)));
    size_t elements_per_thread = 1 << multistep_degree;
    return n_next_power / elements_per_thread;
  }
  // Perform compare/exchange over two elements for a given thread
  template <typename ComparisonOpT>
  XGBOOST_DEVICE void CompareExchangeThread(size_t tidx, ComparisonOpT op) {
    size_t sorted_subsequence_length = SortedSequenceLength();
    size_t bitonic_subsequence_length = BitonicSequenceLength();
    size_t b_idx = 0;
    size_t a_idx =
        (tidx % sorted_subsequence_length) +
        sorted_subsequence_length * ((tidx / sorted_subsequence_length) * 2);
    if (step == phase) {
      // The second index is mirrored in the first step
      // This deals with inputs not an exact power of two
      b_idx = a_idx ^ (bitonic_subsequence_length - 1);
    } else {
      b_idx = a_idx + sorted_subsequence_length;
    }
    if (a_idx < n && b_idx < n) {
      CompareExchange(&iter[a_idx], &iter[b_idx], op);
    }
  }

  template <typename ComparisonOpT>
  XGBOOST_DEVICE void CompareExchangeThread3(size_t tidx, ComparisonOpT op) {
    constexpr size_t degree = 3;
    size_t stride = 1 << (step - degree);
    constexpr size_t elements_per_thread = 1 << degree;
    size_t sub_block_idx = tidx / stride;
    size_t thread_start_position =
        (sub_block_idx * stride * elements_per_thread) + tidx % stride;

    // Read in values
    ValueT values[elements_per_thread];
    bool valid[elements_per_thread];
    for (auto i = 0ull; i < elements_per_thread; i++) {
      size_t src_idx = thread_start_position + stride * i;
      if (src_idx < n) {
        values[i] = iter[src_idx];
        valid[i] = true;
      } else {
        valid[i] = false;
      }
    }
    CompareExchange(&values[0], &values[4], valid[0], valid[4], op);
    CompareExchange(&values[1], &values[5], valid[1], valid[5], op);
    CompareExchange(&values[2], &values[6], valid[2], valid[6], op);
    CompareExchange(&values[3], &values[7], valid[3], valid[7], op);

    CompareExchange(&values[0], &values[2], valid[0], valid[2], op);
    CompareExchange(&values[1], &values[3], valid[1], valid[3], op);
    CompareExchange(&values[4], &values[6], valid[4], valid[6], op);
    CompareExchange(&values[5], &values[7], valid[5], valid[7], op);

    CompareExchange(&values[0], &values[1], valid[0], valid[1], op);
    CompareExchange(&values[2], &values[3], valid[2], valid[3], op);
    CompareExchange(&values[4], &values[5], valid[4], valid[5], op);
    CompareExchange(&values[6], &values[7], valid[6], valid[7], op);

    // Write back results
    for (auto i = 0ull; i < elements_per_thread; i++) {
      size_t dst_idx = thread_start_position + stride * i;
      if (dst_idx < n) {
        iter[dst_idx] = values[i];
      }
    }
  }

  // The size of sorted subsequences at this step
  XGBOOST_DEVICE size_t SortedSequenceLength() { return 1 << (step - 1); }
  // The size of bitonic (ascending/descending) subsequences at this step
  XGBOOST_DEVICE size_t BitonicSequenceLength() { return 1 << step; }

 private:
  T iter;
  size_t n;
  int phase;
  int step;
};

template <int TILE_SIZE, typename InputIteratorT, typename OutputIteratorT,
          typename ComparisonOpT>
  __global__ void BitonicSortSharedKernel(InputIteratorT input_begin,
    InputIteratorT input_end,
    OutputIteratorT output_begin,
    ComparisonOpT op,int begin_phase,int begin_step) {
  size_t n = input_end - input_begin;
  using ValueT = typename std::iterator_traits<OutputIteratorT>::value_type;
  __shared__ ValueT s_values[TILE_SIZE];
  size_t block_base_offset = blockIdx.x *  TILE_SIZE;
  // Copy values in
  for (auto i : dh::BlockStrideRange(size_t(0), size_t(TILE_SIZE))) {
    if (block_base_offset + i < n) {
      s_values[i] = input_begin[block_base_offset + i];
    }
  }
  __syncthreads();
  const int kMaxPhases = std::ceil(std::log2(double(n)));
  bool finished = false; // Indicates we cannot go further in this kernel
  for (int phase = begin_phase; phase <= kMaxPhases; phase++) {
    for (int step = phase == begin_phase ? begin_step : phase; step > 0;
         step--) {
      BitonicSequence<ValueT *> bitonic(s_values, n - block_base_offset, phase,
                                        step);

      if (bitonic.BitonicSequenceLength() > TILE_SIZE) {
        // Subsequence length to large
        finished = true;
        break;
      }

      bitonic.CompareExchangeThread(threadIdx.x, op);

      __syncthreads();
    }

    if (finished) {
      break;
    }
  }
  // Write values back
  for (auto i : dh::BlockStrideRange(size_t(0), size_t(TILE_SIZE))) {
    if (block_base_offset + i < n) {
      output_begin[block_base_offset + i] = s_values[i];
    }
  }
}

// Subsequences are small enough to be processed in shared memory
template <int TILE_SIZE, typename InputIteratorT, typename OutputIteratorT,
  typename ComparisonOpT>
void BitonicShared(InputIteratorT input_begin, InputIteratorT input_end,
    OutputIteratorT output_begin, ComparisonOpT op,int phase,int step)
{
  size_t n = input_end - input_begin;
  constexpr int kBlockSize = TILE_SIZE / 2;
  auto grid_size = xgboost::common::DivRoundUp(n, TILE_SIZE);
  BitonicSortSharedKernel<TILE_SIZE>
      <<<grid_size, kBlockSize>>>(input_begin, input_end, output_begin, op,phase,step);
}

// Subsequences must be processed in global memory
template < typename InputIteratorT, typename OutputIteratorT,
          typename ComparisonOpT>
void BitonicGlobal(InputIteratorT input_begin, InputIteratorT input_end,
                   OutputIteratorT output_begin, ComparisonOpT op, int phase,
                   int step) {
  BitonicSequence<OutputIteratorT> bitonic(
      output_begin, input_end - input_begin, phase, step);
  dh::LaunchN(0, bitonic.NumThreadsRequired(1), [=] __device__(size_t idx) {
    BitonicSequence<OutputIteratorT> d_bitonic = bitonic;
    d_bitonic.CompareExchangeThread(idx, op);
  });
}

template <typename InputIteratorT, typename OutputIteratorT,
  typename ComparisonOpT>
  void BitonicSortShared(InputIteratorT input_begin, InputIteratorT input_end,
    OutputIteratorT output_begin, ComparisonOpT op) {
  using ValueT = typename std::iterator_traits<OutputIteratorT>::value_type;

  size_t n = input_end - input_begin;
  int num_phases = std::ceil(std::log2(n));
  constexpr int kSharedTileSize = 1024;  // Each thread handles two elements
  BitonicShared<kSharedTileSize>(input_begin, input_end, output_begin, op, 1, 1);
  int num_phases_shared = std::ceil(std::log2(kSharedTileSize));

  for (int phase = num_phases_shared + 1; phase <= num_phases; phase++) {
    for (int step = phase; step > 0; step--) {
      size_t bitonic_subsequence_length = std::pow(2, step);
      if (bitonic_subsequence_length > kSharedTileSize) {
        BitonicGlobal(input_begin, input_end, output_begin, op, phase, step);
      } else {
        BitonicShared<kSharedTileSize>(input_begin, input_end, output_begin, op,
                                     phase, step);
        break;
      }
    }
  }
}

template <typename InputIteratorT, typename OutputIteratorT,
          typename ComparisonOpT>
void BitonicSortNaive(InputIteratorT input_begin, InputIteratorT input_end,
                 OutputIteratorT output_begin, ComparisonOpT op) {
  using ValueT = typename std::iterator_traits<OutputIteratorT>::value_type;

  size_t n = input_end - input_begin;
  thrust::copy_n(input_begin, n, output_begin);
  int num_phases = std::ceil(std::log2(n));
  for (int phase = 1; phase <= num_phases; phase++) {
    for (int step = phase; step > 0; step--) {
        BitonicGlobal(input_begin, input_end, output_begin, op, phase, step);
    }
  }
}

template < typename InputIteratorT, typename OutputIteratorT,
          typename ComparisonOpT>
__global__ void BitonicSortMultiStep3Kernel(InputIteratorT input_begin,
                                            InputIteratorT input_end,
                                            OutputIteratorT output_begin,
                                            ComparisonOpT op, int phase,
                                            int step) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
  BitonicSequence<OutputIteratorT> bitonic(
      output_begin, input_end - input_begin, phase, step);
  bitonic.CompareExchangeThread3(tidx, op);
}

template < typename InputIteratorT, typename OutputIteratorT,
          typename ComparisonOpT>
void BitonicGlobalMultiStep3(InputIteratorT input_begin, InputIteratorT input_end,
                   OutputIteratorT output_begin, ComparisonOpT op, int phase, int step) {
  BitonicSequence<OutputIteratorT> bitonic(output_begin, input_end - input_begin, phase, step);
  constexpr int kBlockSize = 512;
  auto grid_size = xgboost::common::DivRoundUp(bitonic.NumThreadsRequired(3), kBlockSize);
  BitonicSortMultiStep3Kernel<<<grid_size, kBlockSize>>>(input_begin, input_end, output_begin, op, phase,
                                step);
}

template <typename InputIteratorT, typename OutputIteratorT,
  typename ComparisonOpT>
  void BitonicSortMultiStep(InputIteratorT input_begin, InputIteratorT input_end,
    OutputIteratorT output_begin, ComparisonOpT op) {
  using ValueT = typename std::iterator_traits<OutputIteratorT>::value_type;

  size_t n = input_end - input_begin;
  int num_phases = std::ceil(std::log2(n));
  constexpr int kSharedTileSize = 1024;  // Each thread handles two elements
  BitonicShared<kSharedTileSize>(input_begin, input_end, output_begin, op, 1, 1);
  int num_phases_shared = std::ceil(std::log2(kSharedTileSize));
  for (int phase = num_phases_shared; phase <= num_phases; phase++) {
    for (int step = phase; step > 0; step--) {
      size_t bitonic_subsequence_length = std::pow(2, step);
      if (bitonic_subsequence_length > kSharedTileSize) {
      if (step >= 3 && step != phase) {
        BitonicGlobalMultiStep3(input_begin, input_end, output_begin, op, phase,
                                step);
        step -= 2;
      } else {
        BitonicGlobal(input_begin, input_end, output_begin, op, phase, step);
      }
      } else {
        BitonicShared<kSharedTileSize>(input_begin, input_end, output_begin, op,
          phase, step);
        break;
      }
    }
  }
}

void TestBitonicSortNaive()
{
  size_t sizes[] = {5, 16, 1000,  5876};
  int iterations_per_size = 10;
  for (auto size : sizes) {
    for (auto i = 0ull; i < iterations_per_size; i++) {
      thrust::device_vector<int> x(size);
      auto counting = thrust::make_counting_iterator(0ll);
      thrust::transform(counting, counting + size, x.begin(),[=]__device__ (size_t idx) {
        thrust::default_random_engine rng(i);
        thrust::uniform_int_distribution<int> dist;
        rng.discard(idx);
        return dist(rng);
      });

      BitonicSortNaive(x.begin(), x.end(), x.begin(), thrust::greater<int>());

      EXPECT_TRUE(thrust::is_sorted(x.begin(), x.end()));
    }
  }
  
}
TEST(DeviceHelpers, BitonicSortNaive) {
  TestBitonicSortNaive();
}
void TestBitonicSortShared()
{
  size_t sizes[] = {5, 16, 1000,  5876, 20000};
  int iterations_per_size = 10;
  for (auto size : sizes) {
    for (auto i = 0ull; i < iterations_per_size; i++) {
      thrust::device_vector<int> x(size);
      auto counting = thrust::make_counting_iterator(0ll);
      thrust::transform(counting, counting + size, x.begin(),[=]__device__ (size_t idx) {
        thrust::default_random_engine rng(i);
        thrust::uniform_int_distribution<int> dist;
        rng.discard(idx);
        return dist(rng);
      });

      thrust::device_vector<int> x_copy(x);
      BitonicSortShared(x_copy.begin(), x_copy.end(), x_copy.begin(), thrust::greater<int>());
      thrust::sort(x.begin(), x.end());

      EXPECT_EQ(x, x_copy);
    }
  }

}
TEST(DeviceHelpers, BitonicSortShared) {
  TestBitonicSortShared();
}

void TestBitonicSortMultiStep()

{
  size_t sizes[] = {5, 16, 1000,  5876, 20000};
  //size_t sizes[] = {5};
  int iterations_per_size = 10;
  for (auto size : sizes) {
    for (auto i = 0ull; i < iterations_per_size; i++) {
      thrust::device_vector<int> x(size);
      auto counting = thrust::make_counting_iterator(0ll);
      thrust::transform(counting, counting + size, x.begin(),[=]__device__ (size_t idx) {
        thrust::default_random_engine rng(i);
        thrust::uniform_int_distribution<int> dist;
        rng.discard(idx);
        return dist(rng);
      });

      thrust::device_vector<int> x_copy(x);
      BitonicSortMultiStep(x_copy.begin(), x_copy.end(), x_copy.begin(), thrust::greater<int>());
      thrust::sort(x.begin(), x.end());

      EXPECT_EQ(x, x_copy);
    }
  }

}
TEST(DeviceHelpers, BitonicSortMultiStep) {
  TestBitonicSortMultiStep();
}
TEST(DeviceHelpers, BitonicSortFloat) {
  thrust::device_vector<float> x;
  x.push_back(5);
  x.push_back(1);
  BitonicSortNaive(x.begin(), x.end(), x.begin(), thrust::greater<int>());

  EXPECT_TRUE(thrust::is_sorted(x.begin(), x.end()));
}

void TestBitonicSortPerformance()
{
  size_t sizes[] = {1<<15,1<<20,1<<25,1<<28};
  int iterations_per_size = 10;
  for (auto size : sizes) {
    thrust::device_vector<int> x(size);
    auto counting = thrust::make_counting_iterator(0ll);
    thrust::transform(counting, counting + size, x.begin(),
                      [=] __device__(size_t idx) {
                        thrust::default_random_engine rng;
                        thrust::uniform_int_distribution<int> dist;
                        rng.discard(idx);
                        return dist(rng);
                      });
    double bitonic_time = 0;
    double thrust_time = 0;
    for (auto i = 0ull; i < iterations_per_size; i++) {
      auto x_copy = x;
      xgboost::common::Timer t;
      BitonicSortMultiStep(x_copy.begin(), x_copy.end(), x_copy.begin(), thrust::greater<int>());
      dh::safe_cuda(cudaDeviceSynchronize());
      t.Stop();
      bitonic_time  += t.ElapsedSeconds();
      t.Reset();
      thrust::sort(x.begin(), x.end(), [=] __device__(int a, int b) { return a > b; });
      dh::safe_cuda(cudaDeviceSynchronize());
      t.Stop();
      thrust_time  += t.ElapsedSeconds();
    }

      printf("n: %llu bitonic: %f thrust: %f\n", size, bitonic_time/iterations_per_size, thrust_time/iterations_per_size);
  }
}
TEST(DeviceHelpers, BitonicSortPerformance) { TestBitonicSortPerformance(); }

void BitonicSortProfile() {
  size_t size = 1 << 25;
  thrust::device_vector<int> x(size);
  auto counting = thrust::make_counting_iterator(0ll);
  thrust::transform(counting, counting + size, x.begin(),
                    [=] __device__(size_t idx) {
                      thrust::default_random_engine rng;
                      thrust::uniform_int_distribution<int> dist;
                      rng.discard(idx);
                      return dist(rng);
                    });
  BitonicSortMultiStep(x.begin(), x.end(), x.begin(), thrust::greater<int>());
}
TEST(DeviceHelpers, BitonicSortProfile) { BitonicSortProfile(); }
