/*!
 * Copyright 2015-2019 by Contributors
 * \file regression_obj.cu
 * \brief Definition of single-value regression and classification objectives.
 * \author Tianqi Chen, Kailong Chen
 */

#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <cmath>
#include <memory>
#include <vector>

#if defined(__CUDACC__)
#include <thrust/binary_search.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include "../common/device_helpers.cuh"
#endif

#include <rabit/c_api.h>
#include "xgboost/host_device_vector.h"
#include "xgboost/json.h"
#include "xgboost/parameter.h"
#include "xgboost/span.h"

#include "../common/transform.h"
#include "../common/common.h"
#include "./regression_loss.h"
#include "../common/device_helpers.cuh"


namespace xgboost {
namespace obj {

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(regression_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

struct RegLossParam : public XGBoostParameter<RegLossParam> {
  float scale_pos_weight;
  // declare parameters
  DMLC_DECLARE_PARAMETER(RegLossParam) {
    DMLC_DECLARE_FIELD(scale_pos_weight).set_default(1.0f).set_lower_bound(0.0f)
      .describe("Scale the weight of positive examples by this factor");
  }
};

template<typename Loss>
class RegLossObj : public ObjFunction {
 protected:
  HostDeviceVector<int> label_correct_;

 public:
  RegLossObj() = default;

  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.UpdateAllowUnknown(args);
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    if (info.labels_.Size() == 0U) {
      LOG(WARNING) << "Label set is empty.";
    }
    CHECK_EQ(preds.Size(), info.labels_.Size())
        << " " << "labels are not correctly provided"
        << "preds.size=" << preds.Size() << ", label.size=" << info.labels_.Size() << ", "
        << "Loss: " << Loss::Name();
    size_t const ndata = preds.Size();
    out_gpair->Resize(ndata);
    auto device = tparam_->gpu_id;
    label_correct_.Resize(1);
    label_correct_.Fill(1);

    bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }
    auto scale_pos_weight = param_.scale_pos_weight;
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx,
                           common::Span<int> _label_correct,
                           common::Span<GradientPair> _out_gpair,
                           common::Span<const bst_float> _preds,
                           common::Span<const bst_float> _labels,
                           common::Span<const bst_float> _weights) {
          bst_float p = Loss::PredTransform(_preds[_idx]);
          bst_float w = is_null_weight ? 1.0f : _weights[_idx];
          bst_float label = _labels[_idx];
          if (label == 1.0f) {
            w *= scale_pos_weight;
          }
          if (!Loss::CheckLabel(label)) {
            // If there is an incorrect label, the host code will know.
            _label_correct[0] = 0;
          }
          _out_gpair[_idx] = GradientPair(Loss::FirstOrderGradient(p, label) * w,
                                          Loss::SecondOrderGradient(p, label) * w);
        },
        common::Range{0, static_cast<int64_t>(ndata)}, device).Eval(
            &label_correct_, out_gpair, &preds, &info.labels_, &info.weights_);

    // copy "label correct" flags back to host
    std::vector<int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag == 0) {
        LOG(FATAL) << Loss::LabelErrorMsg();
      }
    }
  }

 public:
  const char* DefaultEvalMetric() const override {
    return Loss::DefaultEvalMetric();
  }

  void PredTransform(HostDeviceVector<float> *io_preds) override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<float> _preds) {
          _preds[_idx] = Loss::PredTransform(_preds[_idx]);
        }, common::Range{0, static_cast<int64_t>(io_preds->Size())},
        tparam_->gpu_id)
        .Eval(io_preds);
  }

  float ProbToMargin(float base_score) const override {
    return Loss::ProbToMargin(base_score);
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(Loss::Name());
    out["reg_loss_param"] = toJson(param_);
  }

  void LoadConfig(Json const& in) override {
    fromJson(in["reg_loss_param"], &param_);
  }

 protected:
  RegLossParam param_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(RegLossParam);

XGBOOST_REGISTER_OBJECTIVE(SquaredLossRegression, LinearSquareLoss::Name())
.describe("Regression with squared error.")
.set_body([]() { return new RegLossObj<LinearSquareLoss>(); });

XGBOOST_REGISTER_OBJECTIVE(SquareLogError, SquaredLogError::Name())
.describe("Regression with root mean squared logarithmic error.")
.set_body([]() { return new RegLossObj<SquaredLogError>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticRegression, LogisticRegression::Name())
.describe("Logistic regression for probability regression task.")
.set_body([]() { return new RegLossObj<LogisticRegression>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticClassification, LogisticClassification::Name())
.describe("Logistic regression for binary classification task.")
.set_body([]() { return new RegLossObj<LogisticClassification>(); });

XGBOOST_REGISTER_OBJECTIVE(LogisticRaw, LogisticRaw::Name())
.describe("Logistic regression for classification, output score "
          "before logistic transformation.")
.set_body([]() { return new RegLossObj<LogisticRaw>(); });

// Deprecated functions
XGBOOST_REGISTER_OBJECTIVE(LinearRegression, "reg:linear")
.describe("Regression with squared error.")
.set_body([]() {
    LOG(WARNING) << "reg:linear is now deprecated in favor of reg:squarederror.";
    return new RegLossObj<LinearSquareLoss>(); });
// End deprecated

// declare parameter
struct PoissonRegressionParam : public XGBoostParameter<PoissonRegressionParam> {
  float max_delta_step;
  DMLC_DECLARE_PARAMETER(PoissonRegressionParam) {
    DMLC_DECLARE_FIELD(max_delta_step).set_lower_bound(0.0f).set_default(0.7f)
        .describe("Maximum delta step we allow each weight estimation to be." \
                  " This parameter is required for possion regression.");
  }
};

// poisson regression for count
class PoissonRegression : public ObjFunction {
 public:
  // declare functions
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.UpdateAllowUnknown(args);
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size()) << "labels are not correctly provided";
    size_t const ndata = preds.Size();
    out_gpair->Resize(ndata);
    auto device = tparam_->gpu_id;
    label_correct_.Resize(1);
    label_correct_.Fill(1);

    bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }
    bst_float max_delta_step = param_.max_delta_step;
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx,
                           common::Span<int> _label_correct,
                           common::Span<GradientPair> _out_gpair,
                           common::Span<const bst_float> _preds,
                           common::Span<const bst_float> _labels,
                           common::Span<const bst_float> _weights) {
          bst_float p = _preds[_idx];
          bst_float w = is_null_weight ? 1.0f : _weights[_idx];
          bst_float y = _labels[_idx];
          if (y < 0.0f) {
            _label_correct[0] = 0;
          }
          _out_gpair[_idx] = GradientPair{(expf(p) - y) * w,
                                          expf(p + max_delta_step) * w};
        },
        common::Range{0, static_cast<int64_t>(ndata)}, device).Eval(
            &label_correct_, out_gpair, &preds, &info.labels_, &info.weights_);
    // copy "label correct" flags back to host
    std::vector<int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag == 0) {
        LOG(FATAL) << "PoissonRegression: label must be nonnegative";
      }
    }
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
          _preds[_idx] = expf(_preds[_idx]);
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size())},
        tparam_->gpu_id)
        .Eval(io_preds);
  }
  void EvalTransform(HostDeviceVector<bst_float> *io_preds) override {
    PredTransform(io_preds);
  }
  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }
  const char* DefaultEvalMetric() const override {
    return "poisson-nloglik";
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("count:poisson");
    out["poisson_regression_param"] = toJson(param_);
  }

  void LoadConfig(Json const& in) override {
    fromJson(in["poisson_regression_param"], &param_);
  }

 private:
  PoissonRegressionParam param_;
  HostDeviceVector<int> label_correct_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(PoissonRegressionParam);

XGBOOST_REGISTER_OBJECTIVE(PoissonRegression, "count:poisson")
.describe("Possion regression for count data.")
.set_body([]() { return new PoissonRegression(); });


// cox regression for survival data (negative values mean they are censored)
class CoxRegression : public ObjFunction {
 public:
  void Configure(
      const std::vector<std::pair<std::string, std::string> > &args) override {}

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size()) << "labels are not correctly provided";
    const auto& preds_h = preds.HostVector();
    out_gpair->Resize(preds_h.size());
    auto& gpair = out_gpair->HostVector();
    const std::vector<size_t> &label_order = info.LabelAbsSort();

    const omp_ulong ndata = static_cast<omp_ulong>(preds_h.size()); // NOLINT(*)
    const bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }

    // pre-compute a sum
    double exp_p_sum = 0;  // we use double because we might need the precision with large datasets
    for (omp_ulong i = 0; i < ndata; ++i) {
      exp_p_sum += std::exp(preds_h[label_order[i]]);
    }

    // start calculating grad and hess
    const auto& labels = info.labels_.HostVector();
    double r_k = 0;
    double s_k = 0;
    double last_exp_p = 0.0;
    double last_abs_y = 0.0;
    double accumulated_sum = 0;
    for (omp_ulong i = 0; i < ndata; ++i) { // NOLINT(*)
      const size_t ind = label_order[i];
      const double p = preds_h[ind];
      const double exp_p = std::exp(p);
      const double w = info.GetWeight(ind);
      const double y = labels[ind];
      const double abs_y = std::abs(y);

      // only update the denominator after we move forward in time (labels are sorted)
      // this is Breslow's method for ties
      accumulated_sum += last_exp_p;
      if (last_abs_y < abs_y) {
        exp_p_sum -= accumulated_sum;
        accumulated_sum = 0;
      } else {
        CHECK(last_abs_y <= abs_y) << "CoxRegression: labels must be in sorted order, " <<
                                      "MetaInfo::LabelArgsort failed!";
      }

      if (y > 0) {
        r_k += 1.0/exp_p_sum;
        s_k += 1.0/(exp_p_sum*exp_p_sum);
      }

      const double grad = exp_p*r_k - static_cast<bst_float>(y > 0);
      const double hess = exp_p*r_k - exp_p*exp_p * s_k;
      gpair.at(ind) = GradientPair(grad * w, hess * w);

      last_abs_y = abs_y;
      last_exp_p = exp_p;
    }
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    std::vector<bst_float> &preds = io_preds->HostVector();
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
#pragma omp parallel for schedule(static)
    for (long j = 0; j < ndata; ++j) {  // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    }
  }
  void EvalTransform(HostDeviceVector<bst_float> *io_preds) override {
    PredTransform(io_preds);
  }
  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }
  const char* DefaultEvalMetric() const override {
    return "cox-nloglik";
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("survival:cox");
  }
  void LoadConfig(Json const&) override {}
};

// register the objective function
XGBOOST_REGISTER_OBJECTIVE(CoxRegression, "survival:cox")
.describe("Cox regression for censored survival data (negative labels are considered censored).")
.set_body([]() { return new CoxRegression(); });

// gamma regression
class GammaRegression : public ObjFunction {
 public:
  void Configure(
      const std::vector<std::pair<std::string, std::string> > &args) override {}

  void GetGradient(const HostDeviceVector<bst_float> &preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size()) << "labels are not correctly provided";
    const size_t ndata = preds.Size();
    auto device = tparam_->gpu_id;
    out_gpair->Resize(ndata);
    label_correct_.Resize(1);
    label_correct_.Fill(1);

    const bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx,
                           common::Span<int> _label_correct,
                           common::Span<GradientPair> _out_gpair,
                           common::Span<const bst_float> _preds,
                           common::Span<const bst_float> _labels,
                           common::Span<const bst_float> _weights) {
          bst_float p = _preds[_idx];
          bst_float w = is_null_weight ? 1.0f : _weights[_idx];
          bst_float y = _labels[_idx];
          if (y < 0.0f) {
            _label_correct[0] = 0;
          }
          _out_gpair[_idx] = GradientPair((1 - y / expf(p)) * w, y / expf(p) * w);
        },
        common::Range{0, static_cast<int64_t>(ndata)}, device).Eval(
            &label_correct_, out_gpair, &preds, &info.labels_, &info.weights_);

    // copy "label correct" flags back to host
    std::vector<int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag == 0) {
        LOG(FATAL) << "GammaRegression: label must be nonnegative";
      }
    }
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
          _preds[_idx] = expf(_preds[_idx]);
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size())},
        tparam_->gpu_id)
        .Eval(io_preds);
  }
  void EvalTransform(HostDeviceVector<bst_float> *io_preds) override {
    PredTransform(io_preds);
  }
  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }
  const char* DefaultEvalMetric() const override {
    return "gamma-nloglik";
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:gamma");
  }
  void LoadConfig(Json const&) override {}

 private:
  HostDeviceVector<int> label_correct_;
};

// register the objective functions
XGBOOST_REGISTER_OBJECTIVE(GammaRegression, "reg:gamma")
.describe("Gamma regression for severity data.")
.set_body([]() { return new GammaRegression(); });


// declare parameter
struct TweedieRegressionParam : public XGBoostParameter<TweedieRegressionParam> {
  float tweedie_variance_power;
  DMLC_DECLARE_PARAMETER(TweedieRegressionParam) {
    DMLC_DECLARE_FIELD(tweedie_variance_power).set_range(1.0f, 2.0f).set_default(1.5f)
      .describe("Tweedie variance power.  Must be between in range [1, 2).");
  }
};

// tweedie regression
class TweedieRegression : public ObjFunction {
 public:
  // declare functions
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.UpdateAllowUnknown(args);
    std::ostringstream os;
    os << "tweedie-nloglik@" << param_.tweedie_variance_power;
    metric_ = os.str();
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo &info,
                   int iter,
                   HostDeviceVector<GradientPair> *out_gpair) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size()) << "labels are not correctly provided";
    const size_t ndata = preds.Size();
    out_gpair->Resize(ndata);

    auto device = tparam_->gpu_id;
    label_correct_.Resize(1);
    label_correct_.Fill(1);

    const bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }

    const float rho = param_.tweedie_variance_power;
    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx,
                           common::Span<int> _label_correct,
                           common::Span<GradientPair> _out_gpair,
                           common::Span<const bst_float> _preds,
                           common::Span<const bst_float> _labels,
                           common::Span<const bst_float> _weights) {
          bst_float p = _preds[_idx];
          bst_float w = is_null_weight ? 1.0f : _weights[_idx];
          bst_float y = _labels[_idx];
          if (y < 0.0f) {
            _label_correct[0] = 0;
          }
          bst_float grad = -y * expf((1 - rho) * p) + expf((2 - rho) * p);
          bst_float hess =
              -y * (1 - rho) * \
              std::exp((1 - rho) * p) + (2 - rho) * expf((2 - rho) * p);
          _out_gpair[_idx] = GradientPair(grad * w, hess * w);
        },
        common::Range{0, static_cast<int64_t>(ndata), 1}, device)
        .Eval(&label_correct_, out_gpair, &preds, &info.labels_, &info.weights_);

    // copy "label correct" flags back to host
    std::vector<int>& label_correct_h = label_correct_.HostVector();
    for (auto const flag : label_correct_h) {
      if (flag == 0) {
        LOG(FATAL) << "TweedieRegression: label must be nonnegative";
      }
    }
  }
  void PredTransform(HostDeviceVector<bst_float> *io_preds) override {
    common::Transform<>::Init(
        [] XGBOOST_DEVICE(size_t _idx, common::Span<bst_float> _preds) {
          _preds[_idx] = expf(_preds[_idx]);
        },
        common::Range{0, static_cast<int64_t>(io_preds->Size())},
        tparam_->gpu_id)
        .Eval(io_preds);
  }

  bst_float ProbToMargin(bst_float base_score) const override {
    return std::log(base_score);
  }

  const char* DefaultEvalMetric() const override {
    return metric_.c_str();
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:tweedie");
    out["tweedie_regression_param"] = toJson(param_);
  }
  void LoadConfig(Json const& in) override {
    fromJson(in["tweedie_regression_param"], &param_);
  }

 private:
  std::string metric_;
  TweedieRegressionParam param_;
  HostDeviceVector<int> label_correct_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(TweedieRegressionParam);

XGBOOST_REGISTER_OBJECTIVE(TweedieRegression, "reg:tweedie")
.describe("Tweedie regression for insurance data.")
.set_body([]() { return new TweedieRegression(); });


class AUCExponentialObj : public ObjFunction {
 public:
  void Configure(
      const std::vector<std::pair<std::string, std::string> >& args) override {}

  std::pair<double, double> ReduceCPU(const HostDeviceVector<bst_float>& preds,
                                      const MetaInfo& info) {
    size_t const ndata = preds.Size();
    double sum_exp_pos = 0.0;
    double sum_exp_neg = 0.0;
    const auto& label = info.labels_.ConstHostVector();
    const auto& pred = preds.ConstHostVector();
#pragma omp parallel for reduction(+: sum_exp_pos, sum_exp_neg ) schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) {
      if (label[i] == 1.0) {
        sum_exp_pos += std::exp(-pred[i]);
      } else if (label[i] == 0.0) {
        sum_exp_neg += std::exp(pred[i]);
      }
    }
    return {sum_exp_pos, sum_exp_neg};
  }


#if defined(__CUDACC__)
  std::pair<double, double> ReduceGPU(const HostDeviceVector<bst_float>& preds,
                                      const MetaInfo& info) {
    preds.SetDevice(tparam_->gpu_id);
    info.labels_.SetDevice(tparam_->gpu_id);
    auto input = thrust::make_zip_iterator(thrust::make_tuple(
        preds.ConstDevicePointer(), info.labels_.ConstDevicePointer()));

    auto unary =
        [=] __device__(
            thrust::tuple<float, float> x) -> thrust::pair<double, double> {
      float p = x.get<0>();
      float y = x.get<1>();
      if (y == 1.0) {
        return {std::exp(-p), 0};
      } else {
        return {0, std::exp(p)};
      }
    };
    dh::XGBCachingDeviceAllocator<char> alloc;
    auto result = thrust::transform_reduce(
        thrust::cuda::par(alloc), input, input + preds.Size(), unary,
        thrust::pair<double, double>(0, 0),
        [=] __device__(thrust::pair<double, double> a,
                       thrust::pair<double, double> b) {
          b.first += a.first;
          b.second += a.second;
          return b;
        });
    return std::pair<float, float>(result.first, result.second);
  }
#else
  std::pair<double, double> ReduceGPU(const HostDeviceVector<bst_float>& preds,
                                      const MetaInfo& info) {
    LOG(FATAL) << "XGBoost not complied with GPU support.";
    return {0.0, 0.0};
  }
#endif

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info, int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    if (info.labels_.Size() == 0U) {
      LOG(WARNING) << "Label set is empty.";
    }
    CHECK_EQ(preds.Size(), info.labels_.Size());
    out_gpair->Resize(preds.Size());

    std::pair<double, double> pos_neg_sums;
    if (tparam_->gpu_id >= 0) {
      pos_neg_sums = ReduceGPU(preds, info);
    } else {
      pos_neg_sums = ReduceCPU(preds, info);
    }
    rabit::Allreduce<rabit::op::Sum, double>(
        reinterpret_cast<double*>(&pos_neg_sums), 2);
    const bool is_null_weight = info.weights_.Size() == 0;
    label_correct_.Fill(1);

    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx, common::Span<int> _label_correct,
                           common::Span<GradientPair> _out_gpair,
                           common::Span<const bst_float> _preds,
                           common::Span<const bst_float> _labels,
                           common::Span<const bst_float> _weights) {
          bst_float p = _preds[_idx];
          bst_float w = is_null_weight ? 1.0f : _weights[_idx];
          bst_float y = _labels[_idx];
          bst_float g = 0.0f;
          bst_float h = 0.0f;
          if (y == 1.0) {
            g = -std::exp(-p) * pos_neg_sums.second * w;
            h = -g;
          } else if (y == 0.0) {
            g = std::exp(p) * pos_neg_sums.first * w;
            h = g;
          } else {
            _label_correct[0] = 0;
          }
          _out_gpair[_idx] = GradientPair(g, h);
        },
        common::Range{0, static_cast<int64_t>(preds.Size())}, tparam_->gpu_id)
        .Eval(&label_correct_, out_gpair, &preds, &info.labels_,
              &info.weights_);

    if (label_correct_.ConstHostVector()[0] == 0) {
      LOG(FATAL) << "Label must be 0.0 or 1.0";
    }
  }

  void SaveConfig(Json* p_out) const override {}

  void LoadConfig(Json const& in) override {}

  const char* DefaultEvalMetric() const override { return "auc"; }

 private:
  HostDeviceVector<int> label_correct_{1};
};
XGBOOST_REGISTER_OBJECTIVE(AUCExponentialObj, "reg:auc_exp")
    .describe("Direct AUC optimisation with exponential surrogate function.")
    .set_body([]() { return new AUCExponentialObj(); });

class AUCSquaredObj : public ObjFunction {
 public:
  void Configure(
      const std::vector<std::pair<std::string, std::string> >& args) override {}

  std::pair<double, double> ReduceCPU(const HostDeviceVector<bst_float>& preds,
                                      const MetaInfo& info) {
    size_t const ndata = preds.Size();
    double sum_pos = 0.0;
    double sum_neg = 0.0;
    const auto& label = info.labels_.ConstHostVector();
    const auto& pred = preds.ConstHostVector();
#pragma omp parallel for reduction(+ : sum_pos, sum_neg) schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) {
      if (label[i] == 1.0) {
        sum_pos += pred[i];
      } else if (label[i] == 0.0) {
        sum_neg += pred[i];
      }
    }
    return {sum_pos, sum_neg};
  }

#if defined(__CUDACC__)
  std::pair<double, double> ReduceGPU(const HostDeviceVector<bst_float>& preds,
                                      const MetaInfo& info) {
    preds.SetDevice(tparam_->gpu_id);
    info.labels_.SetDevice(tparam_->gpu_id);
    auto input = thrust::make_zip_iterator(thrust::make_tuple(
        preds.ConstDevicePointer(), info.labels_.ConstDevicePointer()));

    auto unary =
        [=] __device__(
            thrust::tuple<float, float> x) -> thrust::pair<double, double> {
      float p = x.get<0>();
      float y = x.get<1>();
      if (y == 1.0) {
        return {p, 0};
      } else {
        return {0, p};
      }
    };
    dh::XGBCachingDeviceAllocator<char> alloc;
    auto result = thrust::transform_reduce(
        thrust::cuda::par(alloc), input, input + preds.Size(), unary,
        thrust::pair<double, double>(0, 0),
        [=] __device__(thrust::pair<double, double> a,
                       thrust::pair<double, double> b) {
          b.first += a.first;
          b.second += a.second;
          return b;
        });
    return std::pair<float, float>(result.first, result.second);
  }
#else
  std::pair<double, double> ReduceGPU(const HostDeviceVector<bst_float>& preds,
                                      const MetaInfo& info) {
    LOG(FATAL) << "XGBoost not complied with GPU support.";
    return {0.0, 0.0};
  }
#endif
  void LazyCountLabels(const MetaInfo& info) {
    // Lazily count positive/negative labels
    if (num_pos_neg_.first == 0) {
      for (auto y : info.labels_.ConstHostVector()) {
        if (y == 1.0) {
          num_pos_neg_.first += 1;
        } else {
          num_pos_neg_.second += 1;
        }
      }
      rabit::Allreduce<rabit::op::Sum, size_t>(
          reinterpret_cast<size_t*>(&num_pos_neg_), 2);
      CHECK(num_pos_neg_.first && num_pos_neg_.second)
          << "Can't have all positive or all negative labels";
    }
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info, int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    if (info.labels_.Size() == 0U) {
      LOG(WARNING) << "Label set is empty.";
    }
    CHECK_EQ(preds.Size(), info.labels_.Size());
    out_gpair->Resize(preds.Size());

    this->LazyCountLabels(info);
    std::pair<double, double> pos_neg_sums;
    if (tparam_->gpu_id >= 0) {
      pos_neg_sums = ReduceGPU(preds, info);
    } else {
      pos_neg_sums = ReduceCPU(preds, info);
    }
    rabit::Allreduce<rabit::op::Sum, double>(
        reinterpret_cast<double*>(&pos_neg_sums), 2);

    const bool is_null_weight = info.weights_.Size() == 0;
    label_correct_.Fill(1);
    size_t num_positive = num_pos_neg_.first;
    size_t num_negative = num_pos_neg_.second;

    common::Transform<>::Init(
        [=] XGBOOST_DEVICE(size_t _idx, common::Span<int> _label_correct,
                           common::Span<GradientPair> _out_gpair,
                           common::Span<const bst_float> _preds,
                           common::Span<const bst_float> _labels,
                           common::Span<const bst_float> _weights) {
          bst_float p = _preds[_idx];
          bst_float w = is_null_weight ? 1.0f : _weights[_idx];
          bst_float y = _labels[_idx];
          bst_float g = 0.0f;
          bst_float h = 0.0f;
          if (y == 1.0) {
            g = ((p - 1) / num_positive) -
                (pos_neg_sums.second / (num_positive * num_negative));
            h = 1.0 / num_positive;
          } else if (y == 0.0) {
            g = ((p + 1) / num_negative) -
                (pos_neg_sums.first / (num_positive * num_negative));
            h = 1.0 / num_negative;
          } else {
            _label_correct[0] = 0;
          }
          // Keep Hessian around 2 otherwise min_child_weight prevents tree
          // growth
          float normalisation = num_negative + num_positive;
          _out_gpair[_idx] =
              GradientPair(g * w * normalisation, h * w * normalisation);
        },
        common::Range{0, static_cast<int64_t>(preds.Size())}, tparam_->gpu_id)
        .Eval(&label_correct_, out_gpair, &preds, &info.labels_,
              &info.weights_);

    if (label_correct_.ConstHostVector()[0] == 0) {
      LOG(FATAL) << "Label must be 0.0 or 1.0";
    }
  }

  void SaveConfig(Json* p_out) const override {}

  void LoadConfig(Json const& in) override {}

  const char* DefaultEvalMetric() const override { return "auc"; }

 private:
  HostDeviceVector<int> label_correct_{1};
  std::pair<size_t, size_t> num_pos_neg_{0, 0};
};
XGBOOST_REGISTER_OBJECTIVE(AUCSquaredObj, "reg:auc_squared")
    .describe("Direct AUC optimisation with squared surrogate function.")
    .set_body([]() { return new AUCSquaredObj(); });

class AUCHingeObj : public ObjFunction {
 public:
  void Configure(
      const std::vector<std::pair<std::string, std::string> >& args) override {
    CHECK_EQ(rabit::GetWorldSize(), 1) << "Objective for single machine only.";
  }
#if defined(__CUDACC__)
  void GetGradientDevice(const HostDeviceVector<bst_float>& preds,
                         const MetaInfo& info, int iter,
                         HostDeviceVector<GradientPair>* out_gpair) {
    preds.SetDevice(tparam_->gpu_id);
    info.labels_.SetDevice(tparam_->gpu_id);
    info.weights_.SetDevice(tparam_->gpu_id);
    out_gpair->SetDevice(tparam_->gpu_id);
    dh::XGBCachingDeviceAllocator<char> alloc;
    dh::caching_device_vector<float> sorted_preds(
        thrust::device_pointer_cast(preds.ConstDevicePointer()),
        thrust::device_pointer_cast(preds.ConstDevicePointer() + preds.Size()));
    auto neg_iter =
        thrust::partition(thrust::cuda::par(alloc), sorted_preds.begin(),
                          sorted_preds.end(), info.labels_.ConstDevicePointer(),
                          [=] __device__(float y) { return y == 1.0; });
    size_t num_pos = neg_iter - sorted_preds.begin();
    size_t num_neg = preds.Size() - num_pos;
    // sort positives
    thrust::sort(thrust::cuda::par(alloc), sorted_preds.begin(),
                 sorted_preds.begin() + num_pos);
    // sort negatives
    thrust::sort(thrust::cuda::par(alloc), sorted_preds.begin() + num_pos,
                 sorted_preds.end());
    auto d_label = info.labels_.ConstDevicePointer();
    auto d_preds = preds.ConstDevicePointer();
    auto d_out_gpair = out_gpair->DevicePointer();
    const bool is_null_weight = info.weights_.Size() == 0;
    auto d_weight = info.weights_.ConstDevicePointer();
    common::Span<float> sorted_pos(sorted_preds.data().get(), num_pos);
    common::Span<float> sorted_neg(sorted_preds.data().get() + num_pos,
                                   num_neg);
    dh::LaunchN(tparam_->gpu_id, preds.Size(), [=] __device__(size_t idx) {
      float w = is_null_weight ? 1.0f : d_weight[idx];
      float y = d_label[idx];
      float p = d_preds[idx];
      float g, h;
      if (y == 1.0) {
        // Number of negative predictions greater than this prediction - 1
        auto itr = thrust::upper_bound(thrust::seq, sorted_neg.begin(),
                                       sorted_neg.end(), p - 1.0f);
        g = -(sorted_neg.end() - itr);
        h = 1.0;
      } else {
        // Number of positive predictions less than this prediction + 1
        auto itr = thrust::lower_bound(thrust::seq, sorted_pos.begin(),
                                       sorted_pos.end(), p + 1.0f);
        g = itr - sorted_pos.begin();
        h = 1.0;
      }
      d_out_gpair[idx] = GradientPair(g * w, h * w);
    });
  }
#else
  void GetGradientDevice(const HostDeviceVector<bst_float>& preds,
                         const MetaInfo& info, int iter,
                         HostDeviceVector<GradientPair>* out_gpair) {
    LOG(FATAL) << "XGBoost not complied with GPU support.";
  }
#endif
  void GetGradientHost(const HostDeviceVector<bst_float>& preds,
                       const MetaInfo& info, int iter,
                       HostDeviceVector<GradientPair>* out_gpair) {
    std::vector<float> sorted_pos;
    sorted_pos.reserve(preds.Size());
    std::vector<float> sorted_neg;
    sorted_neg.reserve(preds.Size());
    auto& label = info.labels_.ConstHostVector();
    auto& pred = preds.ConstHostVector();
    for (auto i = 0ull; i < preds.Size(); i++) {
      if (label[i] == 1.0) {
        sorted_pos.push_back(pred[i]);
      } else if (label[i] == 0.0) {
        sorted_neg.push_back(pred[i]);
      } else {
        LOG(FATAL) << "Label must be 0.0 or 1.0";
      }
    }
    std::sort(sorted_pos.begin(), sorted_pos.end());
    std::sort(sorted_neg.begin(), sorted_neg.end());

    size_t const ndata = preds.Size();
    auto& out = out_gpair->HostVector();
#pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < ndata; ++i) {
      float w = info.GetWeight(i);
      float y = label[i];
      float p = pred[i];
      float g, h;
      if (y == 1.0) {
        // Number of negative predictions greater than this prediction - 1
        auto itr =
            std::upper_bound(sorted_neg.begin(), sorted_neg.end(), p - 1.0f);
        g = -(sorted_neg.end() - itr);
        h = 1.0;
      } else {
        // Number of positive predictions less than this prediction + 1
        auto itr =
            std::lower_bound(sorted_pos.begin(), sorted_pos.end(), p + 1.0f);
        g = itr - sorted_pos.begin();
        h = 1.0;
      }
      out[i] = GradientPair(g * w, h * w);
    }
  }
  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info, int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    if (info.labels_.Size() == 0U) {
      LOG(WARNING) << "Label set is empty.";
    }
    CHECK_EQ(preds.Size(), info.labels_.Size());
    out_gpair->Resize(preds.Size());

    if (tparam_->gpu_id >= 0) {
      this->GetGradientDevice(preds, info, iter, out_gpair);
    } else {
      this->GetGradientHost(preds, info, iter, out_gpair);
    }
  }

  void SaveConfig(Json* p_out) const override {}

  void LoadConfig(Json const& in) override {}

  const char* DefaultEvalMetric() const override { return "auc"; }

 private:
  HostDeviceVector<int> label_correct_{1};
  std::pair<size_t, size_t> num_pos_neg_{0, 0};
};
XGBOOST_REGISTER_OBJECTIVE(AUCHingeObj, "reg:auc_hinge")
    .describe("Direct AUC optimisation with hinge surrogate function.")
    .set_body([]() { return new AUCHingeObj(); });

}  // namespace obj
}  // namespace xgboost
