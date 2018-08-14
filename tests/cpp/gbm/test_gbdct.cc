// Copyright by Contributors
#include <gtest/gtest.h>
#include "../../src/common/dct.h"
#include "../helpers.h"

namespace xgboost {
TEST(dct, Test) {
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto X = common::ForwardDCT(x);
  auto x_recovered = common::InverseDCT(X);

  // Test basic transform and inverse transform
  for (auto i = 0ull; i < x.size(); i++) {
    EXPECT_FLOAT_EQ(x[i], x_recovered[i]);
  }

  // Test if we can zero high frequency components and approximately recover the
  // series
  auto sparse_coefficients = 2;
  X.resize(sparse_coefficients);
  X.resize(x.size(), 0.0f);
  x_recovered = common::InverseDCT(X);
  for (auto i = 0ull; i < x.size(); i++) {
    EXPECT_NEAR(x[i], x_recovered[i], 0.5f);
  }
}
}  // namespace xgboost
