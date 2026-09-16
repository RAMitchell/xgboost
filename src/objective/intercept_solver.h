/**
 * Copyright 2026, XGBoost Contributors
 * \brief Bracketed scalar intercept estimation without sorting or histograms.
 */
#ifndef XGBOOST_OBJECTIVE_INTERCEPT_SOLVER_H_
#define XGBOOST_OBJECTIVE_INTERCEPT_SOLVER_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../common/kernel.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/linalg.h"
#include "xgboost/span.h"

namespace xgboost::obj {
enum class InterceptLoss { kExpectile, kPseudoHuber };

struct InterceptPoint {
  double value, lower, upper;
};

struct InterceptStats {
  double gradient{0}, hessian{0}, curvature_bound{0};
  double minimum{std::numeric_limits<double>::infinity()};
  double maximum{-std::numeric_limits<double>::infinity()};
};

struct AddInterceptStats {
  XGBOOST_DEVICE InterceptStats operator()(InterceptStats a, InterceptStats b) const {
    return {a.gradient + b.gradient, a.hessian + b.hessian, a.curvature_bound + b.curvature_bound,
            std::min(a.minimum, b.minimum), std::max(a.maximum, b.maximum)};
  }
};

// The first pass gathers weighted moments and bounds. Subsequent passes gather the exact
// gradient, Hessian, and a lower bound on curvature throughout the current bracket.
XGBOOST_DEVICE inline InterceptStats InterceptRow(double label, double weight, InterceptLoss loss,
                                                  double parameter, InterceptPoint point,
                                                  bool initialize) {
  if (weight == 0.0) {
    return {};
  }
  if (initialize) {
    return {weight * label, weight, weight * label * label, label, label};
  }
  auto r = point.value - label;
  if (loss == InterceptLoss::kExpectile) {
    auto scale = r >= 0.0 ? 1.0 - parameter : parameter;
    return {weight * scale * r, weight * scale, weight * std::min(parameter, 1.0 - parameter)};
  }
  auto s = hypot(1.0, r / parameter);
  auto farthest = std::max(std::abs(point.lower - label), std::abs(point.upper - label));
  auto bound = hypot(1.0, farthest / parameter);
  return {weight * (r / s), weight / s / s / s, weight / bound / bound / bound};
}

struct InterceptStatsKernel {
  using Signature = void(Context const*, MetaInfo const&, InterceptLoss, common::Span<double const>,
                         common::Span<InterceptPoint const>, bool, std::vector<InterceptStats>*);
};

/**
 * Solve a scalar intercept problem by finding a zero of its aggregate gradient. The method
 * applies to continuous, monotone gradients with a finite root bracket. Row derivatives,
 * initialization, and curvature bounds depend on the loss; independent outputs are solved
 * separately, not as a coupled optimization problem.
 *
 * Currently implemented losses are expectile and pseudo-Huber. Their parameters are alphas
 * and slopes, respectively; expectiles share one label column, while pseudo-Huber uses one
 * per output. For these losses, the positive-weight label range provides the initial bracket.
 *
 * One initial pass supplies the mean and a bracket containing every positive-weight label.
 * Each subsequent pass sums gradient and Hessian in double precision. Newton steps outside
 * the bracket are replaced by bisection; stalled gradient progress also triggers bisection.
 * All workers, including empty ones, use identical globally reduced statistics and stopping
 * decisions. No row-sized gradient buffer is needed.
 *
 * Stop when the bracket or a curvature-based error bound certifies accuracy of 1e-5 times
 * the weighted label standard deviation, subject to float32 output resolution. A small Newton
 * step alone is not a convergence certificate. Returns the number of statistics passes.
 */
int FitInterceptRoot(Context const* ctx, MetaInfo const& info, InterceptLoss loss,
                     common::Span<double const> parameters, linalg::Vector<float>* out);
}  // namespace xgboost::obj
#endif  // XGBOOST_OBJECTIVE_INTERCEPT_SOLVER_H_
