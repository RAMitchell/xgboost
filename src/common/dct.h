/*!
 * Copyright 2018 by Contributors
 * \file dct.h
 * \brief Discrete cosine transform utilities
 * \author Rory Mitchell
 */
#ifndef XGBOOST_COMMON_DCT_H_
#define XGBOOST_COMMON_DCT_H_
#include <xgboost/logging.h>
#include <vector>

namespace xgboost {
namespace common {

constexpr float kPi = 3.14159265358979f;
using DCTCoefficient = float;

inline std::vector<DCTCoefficient> ForwardDCT(const std::vector<float> &x) {
  int N = x.size();
  int K = x.size();
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

inline std::vector<float> InverseDCT(const std::vector<DCTCoefficient> &x) {
  int N = x.size();
  int K = x.size();
  std::vector<float> X(K);
  for (auto k = 0; k < K; k++) {
    double dct_k = 0.5 * x[0];
    for (auto n = 1; n < N; n++) {
      auto x_n = x[n];
      dct_k += x_n * cos((kPi / N) * n * (k + 0.5f));
    }
    X[k] = dct_k * (2.0f / N);
  }
  return X;
}

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_DCT_H_
