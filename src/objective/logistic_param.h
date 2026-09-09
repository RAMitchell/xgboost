/*!
 * Copyright 2015-2023 by Contributors
 * \file logistic_param.h
 * \brief Parameters for logistic objectives.
 */
#ifndef XGBOOST_OBJECTIVE_LOGISTIC_PARAM_H_
#define XGBOOST_OBJECTIVE_LOGISTIC_PARAM_H_

#include "xgboost/parameter.h"

namespace xgboost {
namespace obj {

struct LogisticParam : public XGBoostParameter<LogisticParam> {
  float scale_pos_weight;
  // declare parameters
  DMLC_DECLARE_PARAMETER(LogisticParam) {
    DMLC_DECLARE_FIELD(scale_pos_weight)
        .set_default(1.0f)
        .set_lower_bound(0.0f)
        .describe("Scale the weight of positive examples by this factor");
  }
};

}  // namespace obj
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_LOGISTIC_PARAM_H_
