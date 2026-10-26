/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../src/tree/driver.h"
#include "../../../src/tree/hist/expand_entry.h"
#include "../../../src/tree/param.h"
#include "test_empty_pop.h"

namespace xgboost::tree {
namespace {
MultiExpandEntry MakeMultiExpandEntry(bst_node_t nidx, bst_node_t depth, float loss_chg) {
  MultiExpandEntry entry{nidx, depth};
  entry.split.loss_chg = loss_chg;
  return entry;
}
}  // namespace

TEST(QuantileHist, DriverEmptyPopSkipsInvalidGroup) {
  TestDriverEmptyPop<MultiExpandEntry>(MakeMultiExpandEntry);
}

TEST(QuantileHist, DriverRejectsZeroBatchSize) {
  TrainParam param;
  param.UpdateAllowUnknown(Args{});
  EXPECT_THROW((Driver<MultiExpandEntry>{param, 0}), dmlc::Error);
}
}  // namespace xgboost::tree
