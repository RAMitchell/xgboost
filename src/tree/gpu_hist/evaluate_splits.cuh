/*!
 * Copyright 2020 by XGBoost Contributors
 */
#ifndef EVALUATE_SPLITS_CUH_
#define EVALUATE_SPLITS_CUH_
#include <xgboost/span.h>
#include "../../data/ellpack_page.cuh"
#include "../constraints.cuh"

namespace xgboost {
namespace tree {

// Subset of training parameters, we cannot copy TrainParam into kernels because
// it is too large
struct GPUTrainingParam {
  // minimum amount of hessian(weight) allowed in a child
  float min_child_weight;
  // L2 regularization factor
  float reg_lambda;
  // L1 regularization factor
  float reg_alpha;
  // maximum delta update we can add in weight estimation
  // this parameter can be used to stabilize update
  // default=0 means no constraint on weight delta
  float max_delta_step;

  GPUTrainingParam() = default;

  XGBOOST_DEVICE explicit GPUTrainingParam(const TrainParam& param)
      : min_child_weight(param.min_child_weight),
        reg_lambda(param.reg_lambda),
        reg_alpha(param.reg_alpha),
        max_delta_step(param.max_delta_step) {}
};

  struct DeviceSplitCandidate {
    enum DefaultDirection {
      kLeftDir = 0,
      kRightDir
    };

    float loss_chg {-FLT_MAX};
    DefaultDirection dir {kLeftDir};
    int findex {-1};
    float fvalue {0};

    GradientPair left_sum;
    GradientPair right_sum;
    int64_t left_instances{0};

    XGBOOST_DEVICE DeviceSplitCandidate() {}  // NOLINT

    XGBOOST_DEVICE void Update(float loss_chg_in, DefaultDirection dir_in,
      float fvalue_in, int findex_in,
      GradientPair left_sum_in,
      GradientPair right_sum_in,
      const GPUTrainingParam& param) {
      if (loss_chg_in > loss_chg &&
        left_sum_in.GetHess() >= param.min_child_weight &&
        right_sum_in.GetHess() >= param.min_child_weight) {
        loss_chg = loss_chg_in;
        dir = dir_in;
        fvalue = fvalue_in;
        left_sum = left_sum_in;
        right_sum = right_sum_in;
        findex = findex_in;
      }
    }
    friend std::ostream& operator<<(std::ostream& os, DeviceSplitCandidate const& c) {
      os << "loss_chg:" << c.loss_chg << ", "
        << "dir: " << c.dir << ", "
        << "findex: " << c.findex << ", "
        << "fvalue: " << c.fvalue << ", "
        << "left sum: " << c.left_sum << ", "
        << "right sum: " << c.right_sum << std::endl;
      return os;
    }
  };

template <typename GradientSumT>
struct EvaluateSplitInputs {
  int nidx;
  GradientSumT parent_sum;
  GPUTrainingParam param;
  common::Span<const bst_feature_t> feature_set;
  common::Span<const uint32_t> feature_segments;
  common::Span<const float> feature_values;
  common::Span<const float> min_fvalue;
  common::Span<const GradientSumT> gradient_histogram;
  ValueConstraint value_constraint;
  common::Span<const int> monotonic_constraints;
};
template <typename GradientSumT>
void EvaluateSplits(common::Span<DeviceSplitCandidate> out_splits,
                    EvaluateSplitInputs<GradientSumT> left,
                    EvaluateSplitInputs<GradientSumT> right,cudaStream_t stream=nullptr);
template <typename GradientSumT>
void EvaluateSingleSplit(common::Span<DeviceSplitCandidate> out_split,
                         EvaluateSplitInputs<GradientSumT> input,cudaStream_t stream=nullptr);
}  // namespace tree
}  // namespace xgboost

#endif  // EVALUATE_SPLITS_CUH_
