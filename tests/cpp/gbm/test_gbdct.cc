// Copyright by Contributors
#include <gtest/gtest.h>
#include "../../src/gbm/gbdct.cc"
#include "../helpers.h"

namespace xgboost {
namespace gbm {

void TestFullTransform(const Matrix<double>& T) {
  EXPECT_EQ(T.Rows(), T.Columns());
  std::vector<double> x(T.Rows());
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
  auto I = Matrix<double>::Identity(T.Rows(), T.Columns());
  // Check orthogonal
  for (auto i = 0ull; i < T.Rows(); i++) {
    for (auto j = 0ull; j < T.Columns(); j++) {
      EXPECT_NEAR(I(i, j), TTt(i, j), 1e-5);
      EXPECT_NEAR(I(i, j), TtT(i, j), 1e-5);
    }
  }
}

void TestTruncatedTransform(const Matrix<double>& T, bool test_near) {
  EXPECT_LT(T.Columns(), T.Rows());
  auto Tt = T.Transpose();

  if (test_near) {
    std::vector<double> x(T.Rows());
    std::iota(x.begin(), x.end(), 0.0);
    auto b = Tt * x;
    auto x_recovered = T * b;
    // Test basic transform and inverse transform
    for (auto i = 0ull; i < x.size(); i++) {
      EXPECT_NEAR(x[i], x_recovered[i], 2.0f);
    }
  }

  auto TtT = Tt * T;
  auto I = Matrix<double>::Identity(T.Columns(), T.Columns());
  // Check semi-orthogonal
  for (auto i = 0ull; i < T.Columns(); i++) {
    for (auto j = 0ull; j < T.Columns(); j++) {
      EXPECT_NEAR(I(i, j), TtT(i, j), 1e-5);
    }
  }
}

TEST(dct, Test) {
  TestFullTransform(Matrix<double>::InverseDCT(16, 16));
  TestTruncatedTransform(Matrix<double>::InverseDCT(16, 7), true);
}

TEST(haar, Test) {
  TestFullTransform(Matrix<double>::InverseHaar(16, 16));
  TestTruncatedTransform(Matrix<double>::InverseHaar(16, 7), true);
}

TEST(rp, Test) {
  int seeds[] = {0, 177, 20000};
  for (auto seed : seeds) {
    TestFullTransform(Matrix<double>::RandomProjection(16, 16, seed));
    TestTruncatedTransform(Matrix<double>::RandomProjection(16, 7, seed),
                           false);
  }
}

TEST(linear_solve, Test) {
  int n = 8;
  int m = 5;
  auto T = Matrix<double>::InverseDCT(m, n);
  auto A =
      T.Transpose() * Matrix<double>::Diagonal({4, 5, 4, 5, 4, 3, 2, 7}) * T;
  std::vector<double> x = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  auto b = A * x;
  auto result = Matrix<double>::CholeskySolve(A, b);
  for (auto i = 0ull; i < m; i++) {
    EXPECT_NEAR(result[i], x[i], 1e-5f);
  }
}
}  // namespace gbm
}  // namespace xgboost
