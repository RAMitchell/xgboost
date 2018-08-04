/*!
 * Copyright 2018 by Contributors
 * \file dct.h
 * \brief Discrete cosine transform utilities
 * \author Rory Mitchell
 */
#ifndef XGBOOST_COMMON_DCT_H_
#define XGBOOST_COMMON_DCT_H_
#include <vector>
#include <xgboost/logging.h>

namespace xgboost {
namespace common {

constexpr float kPi = 3.14159265358979f;
using DCTCoefficient = float;

inline std::vector<DCTCoefficient> ForwardDCT(const std::vector<float> &x, int N,
                                       int K) {
  CHECK_EQ(x.size(), N);
  CHECK_EQ(x.size(), K);
  std::vector<DCTCoefficient> X(K);
  for (auto k = 0; k < K; k++) {
    double dct_k = 0;
    for (auto n = 0; n < N; n++) {
      dct_k += x[n] * cos((kPi / N) * (n + 0.5f) * k);
    }
    X[k] = dct_k;
  }
  return X;
}

inline float InverseDCTSingle(const std::vector<DCTCoefficient> &x, int N, int k) {
  CHECK_EQ(x.size(), N);
  CHECK_LE(k, x.size());
  double dct_k = 0.5 * x[0];
  for (auto n = 1; n < N; n++) {
    auto x_n = x[n];
    dct_k += x_n * cos((kPi / N) * n * (k + 0.5f));
  }
  return dct_k * (2.0f / N);
}

inline std::vector<float> InverseDCT(const std::vector<DCTCoefficient> &x, int N,
                              int K) {
  CHECK_EQ(x.size(), N);
  CHECK_EQ(x.size(), K);
  std::vector<float> X(K);
  for (auto k = 0; k < K; k++) {
    X[k] = InverseDCTSingle(x, N, k);
  }
  return X;
}

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_DCT_H_
