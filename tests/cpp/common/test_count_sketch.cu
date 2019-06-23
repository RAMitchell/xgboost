
#include <gtest/gtest.h>
#include "../../../src/common/count_sketch.cuh"


TEST(CountSketch, SingleElement) {
  CountSketch<int, int > sketch(1 << 8, 4);
  // Insert a single element
  thrust::device_vector<int> keys = std::vector<int>{5};
  thrust::device_vector<int> counts = std::vector<int>{7};
  sketch.Add(keys.begin(),keys.end(),counts.begin());

  thrust::device_vector<int> counts_out(counts.size());
  sketch.PointQuery(keys.begin(), keys.end(), counts_out.begin());
  EXPECT_EQ(counts.front(), counts_out.front());
}

TEST(CountSketch, MultipleElement) {
  CountSketch<int, int > sketch(1 << 8, 4);
  thrust::device_vector<int> keys = std::vector<int>{5,5,5,5,5};
  sketch.Add(keys.begin(),keys.end());

  thrust::device_vector<int> counts_out(1);
  sketch.PointQuery(keys.begin(), keys.end(), counts_out.begin());
  EXPECT_EQ(keys.size(), counts_out.front());
}
