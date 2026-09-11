/**
 * Copyright 2015-2026, XGBoost Contributors
 * \file elementwise_metric.h
 * \brief Shared declarations and CPU kernels for elementwise metrics.
 */
#ifndef XGBOOST_METRIC_ELEMENTWISE_METRIC_H_
#define XGBOOST_METRIC_ELEMENTWISE_METRIC_H_

#include <cmath>    // for log1p, sqrt
#include <cstddef>  // for size_t
#include <numeric>  // for accumulate
#include <vector>   // for vector

#include "../common/kernel.h"            // for KernelRegistration
#include "../common/optional_weight.h"   // for OptionalWeights
#include "../common/threading_utils.h"   // for ParallelFor1d
#include "metric_common.h"               // for CheckRowWeights, PackedReduceResult
#include "xgboost/base.h"                // for bst_float
#include "xgboost/context.h"             // for Context, DeviceOrd
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for UnravelIndex

namespace xgboost::metric {
struct EvalRowRMSE {
  char const* Name() const { return "rmse"; }

  XGBOOST_DEVICE bst_float EvalRow(bst_float label, bst_float pred) const {
    bst_float diff = label - pred;
    return diff * diff;
  }
  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? std::sqrt(esum) : std::sqrt(esum / wsum);
  }
};

struct EvalRowRMSLE {
  char const* Name() const { return "rmsle"; }

  XGBOOST_DEVICE bst_float EvalRow(bst_float label, bst_float pred) const {
    bst_float diff = std::log1p(label) - std::log1p(pred);
    return diff * diff;
  }
  static double GetFinal(double esum, double wsum) {
    return wsum == 0 ? std::sqrt(esum) : std::sqrt(esum / wsum);
  }
};

namespace elementwise {
template <typename Policy>
struct EvalKernel {
  using Signature = PackedReduceResult(Context const*, HostDeviceVector<float> const&,
                                       MetaInfo const&, Policy);
};

namespace detail {
template <typename Policy>
PackedReduceResult EvalCpu(Context const* ctx, HostDeviceVector<float> const& preds,
                           MetaInfo const& info, Policy policy) {
  CheckRowWeights(info);
  auto labels = info.labels.HostView();
  auto predts = preds.ConstHostSpan();
  common::OptionalWeights weights{info.weights_.ConstHostSpan()};

  auto n_threads = ctx->Threads();
  std::vector<double> score_tloc(n_threads, 0.0);
  std::vector<double> weight_tloc(n_threads, 0.0);
  std::size_t constexpr kBlockSize = 2048;
  common::ParallelFor1d<kBlockSize>(labels.Size(), n_threads, [&](auto&& block) {
    double sum_score = 0.0;
    double sum_weight = 0.0;
    for (std::size_t i = block.begin(), n = block.end(); i < n; ++i) {
      auto [sample_id, target_id] = linalg::UnravelIndex(i, labels.Shape());
      float weight = weights[sample_id];
      float residue = policy.EvalRow(labels(sample_id, target_id), predts[i]) * weight;
      sum_score += residue;
      sum_weight += weight;
    }

    auto t_idx = omp_get_thread_num();
    score_tloc[t_idx] += sum_score;
    weight_tloc[t_idx] += sum_weight;
  });

  auto residue_sum = std::accumulate(score_tloc.cbegin(), score_tloc.cend(), 0.0);
  auto weights_sum = std::accumulate(weight_tloc.cbegin(), weight_tloc.cend(), 0.0);
  return PackedReduceResult{residue_sum, weights_sum};
}
}  // namespace detail

template <typename Policy>
auto RegisterEvalCpu() {
  using Kernel = EvalKernel<Policy>;
  return common::KernelRegistration<Kernel>{DeviceOrd::kCPU, &detail::EvalCpu<Policy>};
}
}  // namespace elementwise
}  // namespace xgboost::metric

#endif  // XGBOOST_METRIC_ELEMENTWISE_METRIC_H_
