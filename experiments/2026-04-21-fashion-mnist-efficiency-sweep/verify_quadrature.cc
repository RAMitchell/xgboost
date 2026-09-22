// Check the implementation's advertised Gauss-Legendre polynomial exactness.
#include <cmath>
#include <iostream>
#include "src/predictor/interpretability/quadrature.h"

int main() {
  bool exact = true;
  for (std::size_t n : {4, 6, 8, 16}) {
    auto rule = xgboost::interpretability::detail::MakeEndpointQuadrature<16>(n, 1e-15);
    double max_error = 0;
    for (std::size_t degree = 0; degree < 2 * n; ++degree) {
      double integral = 0;
      for (std::size_t i = 0; i < n; ++i) {
        integral += rule.weights[i] * std::pow(rule.nodes[i], degree);
      }
      max_error = std::max(max_error, std::abs(integral - 1.0 / (degree + 1)));
    }
    std::cout << n << " points, degrees 0.." << 2 * n - 1
              << ": max absolute error = " << max_error << '\n';
    exact &= max_error < 1e-13;
  }
  return exact ? 0 : 1;
}
