// Copyright by Contributors
#include <gtest/gtest.h>
#include "../../src/gbm/gbdct.cc"
#include "../helpers.h"

namespace xgboost {
TEST(dct, Test) {
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto DCT = gbm::Matrix<float>::ForwardDCT(x.size(), x.size());
  auto inverse_DCT = DCT.Transpose();
  auto X = DCT * x;
  auto x_recovered = inverse_DCT * X;
  // Test basic transform and inverse transform
  for (auto i = 0ull; i < x.size(); i++) {
    EXPECT_NEAR(x[i], x_recovered[i], 1e-5);
  }

  // Test if we can remove high frequency components and approximately recover
  // the series
  auto sparse_coefficients = 2;
  X.resize(sparse_coefficients);
  auto truncated_inverse =
      gbm::Matrix<float>::ForwardDCT(sparse_coefficients, x.size()).Transpose();
  x_recovered = truncated_inverse * X;
  for (auto i = 0ull; i < x.size(); i++) {
    EXPECT_NEAR(x[i], x_recovered[i], 0.5f);
  }
}

TEST(square_matrix_multiply, Test) {
  int n = 16;
  auto DCT = gbm::Matrix<float>::ForwardDCT(n, n);
  auto I = DCT * DCT.Transpose();
  for (auto i = 0ull; i < n; i++) {
    EXPECT_FLOAT_EQ(I.Data()[i * n + i], 1.0f);
  }
}

TEST(rectangular_matrix_multiply, Test) {
  int n = 16;
  int m = 9;
  auto DCT = gbm::Matrix<float>::ForwardDCT(m, n);
  auto I = DCT * DCT.Transpose();
  for (auto i = 0ull; i < m; i++) {
     EXPECT_FLOAT_EQ(I.Data()[i * m + i], 1.0f);
  }
}
}  // namespace xgboost
