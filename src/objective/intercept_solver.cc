/**
 * Copyright 2026, XGBoost Contributors
 */
#include "intercept_solver.h"

#include <dmlc/registry.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "../collective/aggregator.h"
#include "../common/optional_weight.h"
#include "init_estimation.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(intercept_solver);
namespace {
void InterceptStatsCpu(Context const* ctx, MetaInfo const& info, InterceptLoss loss,
                       common::Span<double const> parameters,
                       common::Span<InterceptPoint const> points, bool initialize,
                       std::vector<InterceptStats>* out) {
  auto labels = info.labels.HostView();
  auto weights = common::MakeOptionalWeights(DeviceOrd::CPU(), info.weights_);
  out->resize(parameters.size());
  for (std::size_t j = 0; j < parameters.size(); ++j) {
    double g{0}, h{0}, c{0};
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    auto column = loss == InterceptLoss::kExpectile ? 0 : j;
    auto point = initialize ? InterceptPoint{} : points[j];
#pragma omp parallel for num_threads(ctx->Threads()) reduction(+ : g, h, c) reduction(min : lo) \
    reduction(max : hi)
    for (bst_omp_uint i = 0; i < info.num_row_; ++i) {
      auto row =
          InterceptRow(labels(i, column), weights[i], loss, parameters[j], point, initialize);
      g += row.gradient;
      h += row.hessian;
      c += row.curvature_bound;
      lo = std::min(lo, row.minimum);
      hi = std::max(hi, row.maximum);
    }
    (*out)[j] = {g, h, c, lo, hi};
  }
}
auto const kRegisterInterceptStatsCpu =
    common::KernelRegistration<InterceptStatsKernel>{DeviceOrd::kCPU, &InterceptStatsCpu};
}  // namespace

int FitInterceptRoot(Context const* ctx, MetaInfo const& info, InterceptLoss loss,
                     common::Span<double const> parameters, linalg::Vector<float>* out) {
  CheckInitInputs(info);
  CHECK(!parameters.empty());
  for (auto p : parameters) {
    CHECK(std::isfinite(p));
    if (loss == InterceptLoss::kExpectile) {
      CHECK_GE(p, 0);
      CHECK_LE(p, 1);
    } else {
      CHECK_GT(p, 0);
    }
  }
  auto n = parameters.size();
  CHECK(info.labels.Shape(1) == (loss == InterceptLoss::kExpectile ? 1 : n) || info.num_row_ == 0);
  auto cpu = ctx->MakeCPU();
  std::vector<InterceptStats> stats;
  std::vector<InterceptPoint> points(n);
  std::vector<double> sums(n * 3), bounds(n * 2), tolerance(n);
  std::vector<double> previous_gradient(n, std::numeric_limits<double>::infinity());
  std::vector<bool> done(n, false);
  auto reduce = [&](bool initialize) {
    common::DispatchKernel<InterceptStatsKernel>(ctx, info, loss, parameters,
                                                 common::Span<InterceptPoint const>{points},
                                                 initialize, &stats);
    for (std::size_t j = 0; j < n; ++j) {
      sums[j * 3] = stats[j].gradient;
      sums[j * 3 + 1] = stats[j].hessian;
      sums[j * 3 + 2] = stats[j].curvature_bound;
      bounds[j * 2] = -stats[j].minimum;
      bounds[j * 2 + 1] = stats[j].maximum;
    }
    // Reduce sufficient statistics, not worker-local intercepts. Even an empty or zero-weight
    // worker participates in every pass; all bracket updates and stopping decisions below use
    // these global values. CUDA kernels return only small host-side statistics for this step.
    collective::SafeColl(collective::GlobalSum(&cpu, linalg::MakeVec(sums.data(), sums.size())));
    if (initialize) {
      collective::SafeColl(collective::Allreduce(
          &cpu, linalg::MakeVec(bounds.data(), bounds.size()), collective::Op::kMax));
    }
  };
  reduce(true);
  for (std::size_t j = 0; j < n; ++j) {
    auto weight = sums[j * 3 + 1];
    if (weight == 0.0) {
      points[j] = {0, 0, 0};
      done[j] = true;
      continue;
    }
    auto mean = sums[j * 3] / weight;
    auto lower = -bounds[j * 2], upper = bounds[j * 2 + 1];
    mean = std::clamp(mean, lower, upper);
    auto variance = std::max(0.0, sums[j * 3 + 2] / weight - mean * mean);
    tolerance[j] =
        std::max({1e-5 * std::sqrt(variance),
                  static_cast<double>(std::numeric_limits<float>::epsilon()) * std::abs(mean),
                  static_cast<double>(std::numeric_limits<float>::denorm_min())});
    points[j] = {mean, lower, upper};
    done[j] = lower == upper || (loss == InterceptLoss::kExpectile && parameters[j] == 0.5);
    if (loss == InterceptLoss::kExpectile && (parameters[j] == 0 || parameters[j] == 1)) {
      points[j].value = parameters[j] == 0 ? lower : upper;
      done[j] = true;
    }
  }
  int passes = 1;
  while (!std::all_of(done.cbegin(), done.cend(), [](bool v) { return v; })) {
    CHECK_LT(passes, 1024) << "Intercept root solver did not converge.";
    reduce(false);
    ++passes;
    for (std::size_t j = 0; j < n; ++j) {
      if (done[j]) {
        continue;
      }
      auto& p = points[j];
      auto g = sums[j * 3], h = sums[j * 3 + 1], lower_h = sums[j * 3 + 2];
      auto error_bound =
          lower_h > 0 ? std::abs(g) / lower_h : std::numeric_limits<double>::infinity();
      if (loss == InterceptLoss::kPseudoHuber && h > 0) {
        // |d log(h_i)/db| <= 3/(2 delta), also for the sum of positive row Hessians.
        // Integrating h(b+t) >= h(b) exp(-L |t|) bounds the distance to the root.
        auto rate = 1.5 / parameters[j];
        auto ratio = rate * (std::abs(g) / h);
        if (ratio < 1) {
          error_bound = std::min(error_bound, -std::log1p(-ratio) / rate);
        }
      }
      if (g == 0 || error_bound <= tolerance[j]) {
        done[j] = true;
        continue;
      }
      if (g > 0) {
        p.upper = p.value;
      } else {
        p.lower = p.value;
      }
      if (p.upper - p.lower <= 2 * tolerance[j]) {
        p.value = p.lower + (p.upper - p.lower) / 2;
        done[j] = true;
        continue;
      }
      auto next = p.value - g / h;
      // Do not disrupt fast local Newton convergence. Bisect when its residual has stopped
      // decreasing substantially, or when its proposed step leaves the bracket.
      if (!std::isfinite(next) || next <= p.lower || next >= p.upper ||
          std::abs(g) > 0.5 * previous_gradient[j]) {
        next = p.lower + (p.upper - p.lower) / 2;
      }
      previous_gradient[j] = std::abs(g);
      p.value = next;
    }
  }
  out->SetDevice(DeviceOrd::CPU());
  out->Reshape(n);
  auto result = out->HostView();
  for (std::size_t j = 0; j < n; ++j) {
    result(j) = points[j].value;
    if (loss == InterceptLoss::kExpectile && j > 0) {
      result(j) = std::max(result(j), result(j - 1));
    }
  }
  LOG(DEBUG) << "Intercept root solver: " << passes << " passes.";
  return passes;
}
}  // namespace xgboost::obj
