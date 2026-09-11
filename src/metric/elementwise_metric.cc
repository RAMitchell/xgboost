/**
 * Copyright 2015-2026, XGBoost Contributors
 * \file elementwise_metric.cc
 * \brief Elementwise metric definitions and CPU kernel registrations.
 * \author Kailong Chen, Tianqi Chen
 */
#include "elementwise_metric.h"

#include <dmlc/registry.h>

#include <array>  // for array

#include "../collective/aggregator.h"   // for GlobalSum
#include "../common/kernel.h"           // for DispatchKernel
#include "metric_common.h"              // for MetricNoCache
#include "xgboost/collective/result.h"  // for SafeColl
#include "xgboost/metric.h"             // for Metric

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(elementwise_metric);

namespace {
auto const kRegisterRMSECpu = elementwise::RegisterEvalCpu<EvalRowRMSE>();
auto const kRegisterRMSLECpu = elementwise::RegisterEvalCpu<EvalRowRMSLE>();
}  // namespace

template <typename Policy>
class EvalEWiseKernelBase : public MetricNoCache {
 public:
  double Eval(HostDeviceVector<bst_float> const& preds, MetaInfo const& info) override {
    CHECK_EQ(preds.Size(), info.labels.Size())
        << "label and prediction size not match, "
        << "hint: use merror or mlogloss for multi-class classification";
    if (info.labels.Size() != 0) {
      CHECK_NE(info.labels.Shape(1), 0);
    }

    auto result =
        common::DispatchKernel<elementwise::EvalKernel<Policy>>(ctx_, preds, info, policy_);
    std::array<double, 2> values{result.Residue(), result.Weights()};
    auto rc = collective::GlobalSum(ctx_, linalg::MakeVec(values.data(), values.size()));
    collective::SafeColl(rc);
    return Policy::GetFinal(values[0], values[1]);
  }

  [[nodiscard]] char const* Name() const override { return policy_.Name(); }

 private:
  Policy policy_;
};

XGBOOST_REGISTER_METRIC(RMSE, "rmse")
    .describe("Rooted mean square error.")
    .set_body([](char const*) { return new EvalEWiseKernelBase<EvalRowRMSE>(); });

XGBOOST_REGISTER_METRIC(RMSLE, "rmsle")
    .describe("Rooted mean square log error.")
    .set_body([](char const*) { return new EvalEWiseKernelBase<EvalRowRMSLE>(); });
}  // namespace xgboost::metric

#if !defined(XGBOOST_USE_CUDA)
#include "elementwise_metric.cu"
#endif  // !defined(XGBOOST_USE_CUDA)
