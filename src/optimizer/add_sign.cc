/*!
 * Copyright by Contributors 2017
 */
#define _USE_MATH_DEFINES

#include <dmlc/parameter.h>
#include <math.h>
#include <xgboost/optimizer.h>

namespace xgboost {
namespace optimizer {

DMLC_REGISTRY_FILE_TAG(add_sign_optimizer);

/*! \brief Add Sign parameters */
struct AddSignOptimizerParam : public dmlc::Parameter<AddSignOptimizerParam> {
  float beta1;
  float alpha;
  float decay;
  // declare parameters
  DMLC_DECLARE_PARAMETER(AddSignOptimizerParam) {
    DMLC_DECLARE_FIELD(beta1)
        .set_range(0.0f, 1.0f)
        .set_default(0.9f)
        .describe("Decay for calculating the moving average");
    DMLC_DECLARE_FIELD(alpha)
        .set_range(0.0f, 10.0f)
        .set_default(1.0f)
        .describe("Alpha used in optimizer.");
    DMLC_DECLARE_FIELD(decay)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe(
            "Coefficent for calculating exponential decay, if 1 there is no "
            "decay.");
  }
};
DMLC_REGISTER_PARAMETER(AddSignOptimizerParam);

class AddSignOptimizer : public Optimizer {
 public:
  void Init(
      const std::vector<std::pair<std::string, std::string>>& cfg) override {
    param.InitAllowUnknown(cfg);
  }

  void OptimizeGradients(HostDeviceVector<GradientPair>* gpair) override {
    auto& host_gpair = gpair->HostVector();
    t++;
    if (!previous_gpair_.empty()) {
      // apply Add sign update
      for (size_t i = 0; i < host_gpair.size(); i++) {
        float g = host_gpair[i].GetGrad();
        m[i] = param.beta1 * m[i] + (1 - param.beta1) * g;
        float newGrad =
            (param.alpha + exponential(t) * sign(g) * sign(m[i])) * g;
        host_gpair[i] = GradientPair(newGrad, host_gpair[i].GetHess());
      }
    } else {
      m.resize(host_gpair.size());
      for (size_t i = 0; i < host_gpair.size(); i++) {
        m[i] = host_gpair[i].GetGrad();
      }
    }
    previous_gpair_ = host_gpair;
  }

  int sign(float x) { return (x > 0) - (x < 0); }

  float exponential(float t) { return pow(param.decay, t); }

 protected:
  AddSignOptimizerParam param;
  int t = 0;
  float base = M_E;
  std::vector<float> m;
  std::vector<GradientPair> previous_gpair_;
};

XGBOOST_REGISTER_OPTIMIZER(AddSignOptimizer, "add_sign_optimizer")
    .describe("Use add sign to accelerate gradient descent.")
    .set_body([]() { return new AddSignOptimizer(); });
}  // namespace optimizer
}  // namespace xgboost
