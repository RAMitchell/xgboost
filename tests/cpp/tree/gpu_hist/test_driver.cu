#include <gtest/gtest.h>
#include "../../../../src/tree/gpu_hist/driver.cuh"

namespace xgboost {
namespace tree {

TEST(GpuHist, DriverDepthWise)
{
  Driver driver(TrainParam::kDepthWise);
  EXPECT_TRUE(driver.Pop().empty());
  ExpandEntry root(0,0,{},0);
  driver.Push({root});
  EXPECT_EQ(driver.Pop().front().nid, 0);
  driver.Push({ExpandEntry{1,1,{},1}});
  driver.Push({ExpandEntry{2,1,{},1}});
  driver.Push({ExpandEntry{3,2,{},1}});
  // Should return entries from level 1
  auto res = driver.Pop();
  EXPECT_EQ(res.size(), 2);
  for (auto &e : res) {
    EXPECT_EQ(e.depth, 1);
  }
  res = driver.Pop();
  EXPECT_EQ(res[0].depth, 2);
  EXPECT_TRUE(driver.Pop().empty());
}


TEST(GpuHist, DriverLossGuided)
{
  Driver driver(TrainParam::kLossGuide);
  EXPECT_TRUE(driver.Pop().empty());
  ExpandEntry root(0,0,{},0);
  driver.Push({root});
  EXPECT_EQ(driver.Pop().front().nid, 0);
  DeviceSplitCandidate high_gain;
  high_gain.loss_chg = 5.0f;
  DeviceSplitCandidate low_gain;
  low_gain.loss_chg = 1.0f;

  // Select high gain first
  driver.Push({ExpandEntry{1,1,low_gain,1}});
  driver.Push({ExpandEntry{2,2,high_gain,2}});
  auto res = driver.Pop();
  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].nid, 2);
  res = driver.Pop();
  EXPECT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].nid, 1);

  // If equal gain, use timestamp
  driver.Push({ExpandEntry{1,1,low_gain,2}});
  driver.Push({ExpandEntry{2,2,low_gain,1}});
  res = driver.Pop();
  EXPECT_EQ(res[0].nid, 2);
  res = driver.Pop();
  EXPECT_EQ(res[0].nid, 1);
}
}  // namespace tree
}  // namespace xgboost
