/*!
 * Copyright 2018 by Contributors
 * \file dct.h
 * \brief Discrete cosine transform utilities
 * \author Rory Mitchell
 */
#ifndef XGBOOST_COMMON_DCT_H_
#define XGBOOST_COMMON_DCT_H_
#include <vector>

namespace xgboost {
namespace common {

constexpr float kPi = 3.14159265358979f;
using DCTCoefficient = float;

/**
 * \brief   Compute the type II discrete cosine transform. Uses naive n^2
 * algorithm. Not recommended for large input sizes.
 *
 * \author  Rory
 * \date    7/08/2018
 *
 * \param   x   A std::vector&lt;float&gt; to process.
 *
 * \return  A std::vector&lt;DCTCoefficient&gt of length x.size();
 */

inline std::vector<DCTCoefficient> ForwardDCT(const std::vector<float> &x) {
  int N = x.size();
  int K = x.size();
  std::vector<DCTCoefficient> X(K);
  for (auto k = 0; k < K; k++) {
    double dct_k = 0;
    for (auto n = 0; n < N; n++) {
      dct_k += x[n] * cos((kPi / N) * (n + 0.5f) * k);
    }
    // Coefficient to enforce orthogonality
    double a_k = k == 0 ? sqrt(1.0 / K) : sqrt(2.0 / K);
    X[k] = dct_k * a_k;
  }
  return X;
}

/**
 * \brief   Truncated version of forward dct. Sets all coefficients after
 * max_coefficients to zero.
 *
 * \author  Rory
 * \date    7/08/2018
 *
 * \param   x                   A std::vector&lt;float&gt; to process.
 * \param   max_coefficients    The maximum coefficients.
 *
 * \return  A std::vector&lt;DCTCoefficient&gt;
 */

inline std::vector<DCTCoefficient> TruncatedForwardDCT(
    const std::vector<float> &x, int max_coefficients) {
  auto X = ForwardDCT(x);
  X.resize(max_coefficients);
  X.resize(x.size(), 0.0f);
  return X;
}

/**
 * \brief   Inverse dct. Also known as the inverse to the type II dct, or type
 * III dct
 *
 * \author  Rory
 * \date    7/08/2018
 *
 * \param   x   A std::vector&lt;DCTCoefficient&gt; to process.
 *
 * \return  A std::vector&lt;float&gt;
 */

inline std::vector<float> InverseDCT(const std::vector<DCTCoefficient> &x) {
  int N = x.size();
  int K = x.size();
  std::vector<float> X(K);
  for (auto k = 0; k < K; k++) {
    double dct_k = 0;
    for (auto n = 0; n < N; n++) {
      auto x_n = x[n];
      double a_n = n == 0 ? sqrt(1.0 / N) : sqrt(2.0 / N);
      dct_k += a_n * x_n * cos((kPi / N) * n * (k + 0.5f));
    }
    X[k] = dct_k;
  }
  return X;
}

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_DCT_H_
