/**
 * Copyright 2026, XGBoost Contributors
 * \file expectile_obj.cc
 * \brief CPU implementation and registration of the expectile objective.
 */
#include "expectile_obj.h"

#include <dmlc/registry.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "../common/expectile_loss_utils.h"
#include "../common/kernel.h"
#include "../common/linalg_op.h"
#include "../common/math.h"
#include "../common/optional_weight.h"
#include "../common/threading_utils.h"
#include "init_estimation.h"
#include "intercept_solver.h"
#include "xgboost/json.h"
#include "xgboost/objective.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(expectile_obj);

namespace {
void ExpectileGradientCpu(Context const* ctx, HostDeviceVector<float> const& preds,
                          MetaInfo const& info, HostDeviceVector<float> const& alpha,
                          bst_target_t n_targets, linalg::Matrix<GradientPair>* out_gpair) {
  auto labels = info.labels.HostView();
  auto weights = common::MakeOptionalWeights(DeviceOrd::CPU(), info.weights_);
  auto predt =
      linalg::MakeTensorView(DeviceOrd::CPU(), preds.ConstHostSpan(), info.num_row_, n_targets);
  auto alpha_h = alpha.ConstHostSpan();
  out_gpair->SetDevice(DeviceOrd::CPU());
  out_gpair->Reshape(info.num_row_, n_targets);
  auto gpair = out_gpair->HostView();
  linalg::cpu_impl::ElementWiseKernel(
      gpair, ctx->Threads(), [=](std::size_t i, std::size_t j) mutable {
        auto label = labels(i, 0);
        auto sample_weight = weights[i];
        float pred = predt(i, 0);
        float grad_sum{0.0f};
        float hess_sum{0.0f};
        for (std::size_t k{0}; k < alpha_h.size(); ++k) {
          if (k > 0) {
            pred += kRtEps + common::SoftPlus(predt(i, k));
          }
          if (k >= j) {
            auto diff = pred - label;
            auto weight_scale = diff >= 0.0f ? 1.0f - alpha_h[k] : alpha_h[k];
            grad_sum += weight_scale * diff * sample_weight;
            hess_sum += weight_scale * sample_weight;
          }
        }
        auto scale = j == 0 ? 1.0f : common::Sigmoid(predt(i, j));
        // Diagonal Gauss-Newton approximation for the transformed margin.
        gpair(i, j) = {scale * grad_sum, scale * scale * hess_sum};
      });
}

void ExpectilePredTransformCpu(Context const* ctx, HostDeviceVector<float>* predictions,
                               std::size_t n_alphas) {
  auto n_samples = predictions->Size() / n_alphas;
  auto predt =
      linalg::MakeTensorView(DeviceOrd::CPU(), predictions->HostSpan(), n_samples, n_alphas);
  common::ParallelFor(n_samples, ctx->Threads(), [=](std::size_t i) mutable {
    float pred = predt(i, 0);
    for (std::size_t j{1}; j < n_alphas; ++j) {
      pred += kRtEps + common::SoftPlus(predt(i, j));
      predt(i, j) = pred;
    }
  });
}

auto const kRegisterGradient =
    common::KernelRegistration<ExpectileGradientKernel>{DeviceOrd::kCPU, &ExpectileGradientCpu};
auto const kRegisterTransform = common::KernelRegistration<ExpectilePredTransformKernel>{
    DeviceOrd::kCPU, &ExpectilePredTransformCpu};
}  // namespace

class ExpectileRegression : public ObjFunction {
  common::ExpectileLossParam param_;
  HostDeviceVector<float> alpha_;

  bst_target_t Targets(MetaInfo const& info) const override {
    auto const& alpha = param_.expectile_alpha.Get();
    CHECK_EQ(alpha.size(), alpha_.Size()) << "The objective is not yet configured.";
    CHECK(info.labels.Shape(1) == 1 || (info.num_row_ == 0 && info.labels.Shape(1) == 0))
        << "Multi-target is not yet supported by the expectile loss.";
    CHECK(!alpha.empty());
    return alpha_.Size();
  }

 public:
  std::set<std::string> Configure(Args const& args) override {
    auto used = UpdateAndGetUsedParameters(&param_, args);
    param_.Validate();
    alpha_.HostVector() = param_.expectile_alpha.Get();
    return used;
  }
  ObjInfo Task() const override { return ObjInfo::kRegression; }
  void GetGradient(HostDeviceVector<float> const& preds, MetaInfo const& info, std::int32_t iter,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    if (iter == 0) {
      CheckInitInputs(info);
    }
    auto n_targets = this->Targets(info);
    CHECK_EQ(preds.Size(), info.num_row_ * n_targets);
    common::DispatchKernel<ExpectileGradientKernel>(ctx_, preds, info, alpha_, n_targets,
                                                    out_gpair);
  }
  void InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const override {
    this->Targets(info);
    auto const& alpha = alpha_.ConstHostVector();
    std::vector<double> parameters(alpha.cbegin(), alpha.cend());
    FitInterceptRoot(ctx_, info, InterceptLoss::kExpectile, parameters, base_score);
  }
  void PredTransform(HostDeviceVector<float>* predictions) const override {
    CHECK_NE(alpha_.Size(), 0);
    CHECK_EQ(predictions->Size() % alpha_.Size(), 0);
    common::DispatchKernel<ExpectilePredTransformKernel>(ctx_, predictions, alpha_.Size());
  }
  void ProbToMargin(linalg::Vector<float>* base_score) const override {
    CHECK_EQ(base_score->Size(), alpha_.Size());
    auto margin = base_score->HostView();
    for (std::size_t j = margin.Size() - 1; j > 0; --j) {
      margin(j) = common::SoftPlusInv(margin(j) - margin(j - 1) - kRtEps);
    }
  }
  char const* DefaultEvalMetric() const override { return "expectile"; }
  Json DefaultMetricConfig() const override {
    CHECK(param_.GetInitialised());
    Json config{Object{}};
    config["name"] = String{this->DefaultEvalMetric()};
    config["expectile_loss_param"] = ToJson(param_);
    return config;
  }
  void SaveConfig(Json* out) const override {
    (*out)["name"] = String("reg:expectileerror");
    (*out)["expectile_loss_param"] = ToJson(param_);
  }
  void LoadConfig(Json const& in) override {
    CHECK_EQ(get<String const>(in["name"]), "reg:expectileerror");
    auto const& obj = get<Object const>(in);
    auto it = obj.find("expectile_loss_param");
    if (it != obj.cend()) {
      FromJson(it->second, &param_);
      alpha_.HostVector() = param_.expectile_alpha.Get();
    }
  }
};

XGBOOST_REGISTER_OBJECTIVE(ExpectileRegression, "reg:expectileerror")
    .describe("Regression with expectile loss.")
    .set_body([]() { return new ExpectileRegression(); });
}  // namespace xgboost::obj
