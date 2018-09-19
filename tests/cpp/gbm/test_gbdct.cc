// Copyright by Contributors
#include <gtest/gtest.h>
#include "../../src/gbm/gbdct.cc"
#include "../helpers.h"

namespace xgboost {
namespace gbm {

void TestFullTransform(const Matrix<float>& T) {
  EXPECT_EQ(T.Rows(), T.Columns());
  std::vector<float> x(T.Rows());
  std::iota(x.begin(), x.end(), 0.0);
  auto Tt = T.Transpose();
  auto b = Tt * x;
  auto x_recovered = T * b;
  // Test basic transform and inverse transform
  for (auto i = 0ull; i < x.size(); i++) {
    EXPECT_NEAR(x[i], x_recovered[i], 1e-5);
  }

  auto TTt = T * Tt;
  auto TtT = Tt * T;
  auto I = Matrix<float>::Identity(T.Rows(), T.Columns());
  // Check orthogonal
  for (auto i = 0ull; i < T.Rows(); i++) {
    for (auto j = 0ull; j < T.Columns(); j++) {
      EXPECT_NEAR(I(i, j), TTt(i, j), 1e-5);
      EXPECT_NEAR(I(i, j), TtT(i, j), 1e-5);
    }
  }
}

void TestTruncatedTransform(const Matrix<float>& T) {
  EXPECT_LT(T.Columns(), T.Rows());
  std::vector<float> x(T.Rows());
  std::iota(x.begin(), x.end(), 0.0);
  auto Tt = T.Transpose();
  auto b = Tt * x;
  auto x_recovered = T * b;
  // Test basic transform and inverse transform
  for (auto i = 0ull; i < x.size(); i++) {
    EXPECT_NEAR(x[i], x_recovered[i], 2.0f);
  }

  auto TtT = Tt * T;
  auto I = Matrix<float>::Identity(T.Columns(), T.Columns());
  // Check semi-orthogonal
  for (auto i = 0ull; i < T.Columns(); i++) {
    for (auto j = 0ull; j < T.Columns(); j++) {
      EXPECT_NEAR(I(i, j), TtT(i, j), 1e-5);
    }
  }
}

TEST(dct, Test) {
  TestFullTransform(Matrix<float>::InverseDCT(16, 16));
  TestTruncatedTransform(Matrix<float>::InverseDCT(16, 7));
}

TEST(haar, Test) {
  TestFullTransform(Matrix<float>::InverseHaar(16, 16));
  TestTruncatedTransform(Matrix<float>::InverseHaar(16, 7));
}

TEST(linear_solve, Test) {
  int n = 8;
  int m = 5;
  auto T = Matrix<float>::InverseDCT(m, n);
  auto A = T.Transpose() *
           Matrix<float>::Diagonal({4, 5, 4, 5, 4, 3, 2, 7}) * T;
  std::vector<float> x = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  auto b = A * x;
  auto result = Matrix<float>::CholeskySolve(A, b);
  for (auto i = 0ull; i < m; i++) {
    EXPECT_NEAR(result[i], x[i], 1e-5f);
  }
}
}  // namespace gbm
}  // namespace xgboost
