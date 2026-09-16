/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "../../../src/objective/intercept_solver.h"
#include "../collective/test_worker.h"
#include "../helpers.h"
#include "xgboost/json.h"
#include "xgboost/learner.h"

namespace xgboost::obj {
namespace {
void TestRoot(Context const* ctx) {
  MetaInfo info;
  info.num_row_ = 4;
  info.labels.Reshape(4, 1);
  info.labels.Data()->HostVector() = {-10, -1, 1, 10};
  linalg::Vector<float> out;
  std::vector<double> parameters{0, 0.2, 0.5, 0.8, 1};
  auto passes = FitInterceptRoot(ctx, info, InterceptLoss::kExpectile, parameters, &out);
  ASSERT_LE(passes, 20);
  std::vector<double> expected{-10, -30.0 / 7.0, 0, 30.0 / 7.0, 10};
  for (std::size_t j = 0; j < expected.size(); ++j) {
    EXPECT_NEAR(out(j), expected[j], 1e-4);
  }
  parameters = {1};
  EXPECT_EQ(FitInterceptRoot(ctx, info, InterceptLoss::kPseudoHuber, parameters, &out), 2);
  EXPECT_EQ(out(0), 0);

  // Constant labels and zero-weight outliers require only the initial statistics pass.
  info.labels.Data()->HostVector() = {7, 7, 7, 1e20f};
  info.weights_.HostVector() = {1, 2, 3, 0};
  EXPECT_EQ(FitInterceptRoot(ctx, info, InterceptLoss::kPseudoHuber, parameters, &out), 1);
  EXPECT_EQ(out(0), 7);
  info.weights_.HostVector() = {0, 0, 0, 0};
  EXPECT_EQ(FitInterceptRoot(ctx, info, InterceptLoss::kPseudoHuber, parameters, &out), 1);
  EXPECT_EQ(out(0), 0);
}
}  // namespace

TEST(InterceptSolver, CPU) {
  Context ctx;
  TestRoot(&ctx);
}
#if defined(XGBOOST_USE_CUDA)
TEST(InterceptSolver, CUDA) {
  auto ctx = MakeCUDACtx(0);
  TestRoot(&ctx);
}
#endif

TEST(InterceptSolver, EmptyWorker) {
  for (bool expectile : {false, true}) {
    collective::TestDistributedGlobal(2, [=] {
      auto empty = collective::GetRank() == 1;
      auto n_targets = expectile ? 1 : 2;
      auto data =
          RandomDataGenerator{empty ? 0ul : 4ul, 1, 0.0f}.Targets(n_targets).GenerateDMatrix(
              !empty);
      if (!empty) {
        data->Info().labels.Data()->HostVector() =
            expectile ? std::vector<float>{-10, -1, 1, 10}
                      : std::vector<float>{-10, 90, -1, 99, 1, 101, 10, 110};
      }
      std::unique_ptr<Learner> learner{Learner::Create({data})};
      Args args{{"tree_method", "hist"},
                {"max_depth", "1"},
                {"objective", expectile ? "reg:expectileerror" : "reg:pseudohubererror"}};
      if (expectile) {
        args.emplace_back("expectile_alpha", "[0.2, 0.5, 0.8]");
      }
      learner->Configure(args);
      learner->UpdateOneIter(0, data);
      Json config{Object{}};
      learner->SaveConfig(&config);
      auto base = GetBaseScore(config);
      if (expectile) {
        ASSERT_EQ(base.size(), 3);
        EXPECT_NEAR(base[0], -30.0 / 7.0, 1e-4);
        EXPECT_NEAR(base[1], 0, 1e-4);
        EXPECT_NEAR(base[2], 30.0 / 7.0, 1e-4);
      } else {
        ASSERT_EQ(base.size(), 2);
        EXPECT_NEAR(base[0], 0, 1e-4);
        EXPECT_NEAR(base[1], 100, 1e-4);
      }
      learner->UpdateOneIter(1, data);
    });
  }
}

namespace {
void TestDistributedRoot(Context const* ctx) {
  for (auto loss : {InterceptLoss::kExpectile, InterceptLoss::kPseudoHuber}) {
    for (int mode : {0, 1, 2}) {
      // Uneven partitions, with either an empty worker, a worker containing only zero-weight
      // outliers, or globally zero weight. The two populated partitions have different roots.
      std::vector<float> labels{-100, -10, -1, 0, 1, 3, 1000};
      std::vector<float> weights{0, 0.25f, 2, 1, 7, 0.5f, 0.125f};
      if (mode == 2) {
        std::fill(weights.begin(), weights.end(), 0.0f);
      }
      auto expectile = loss == InterceptLoss::kExpectile;
      std::size_t columns = expectile ? 1 : 2;
      std::vector<double> parameters =
          expectile ? std::vector<double>{0.001, 0.5, 0.999} : std::vector<double>{0.1, 1.0};
      auto set_info = [&](std::size_t begin, std::size_t end, MetaInfo* info) {
        info->num_row_ = end - begin;
        info->labels.Reshape(info->num_row_, columns);
        for (std::size_t i = begin; i < end; ++i) {
          info->labels.HostView()(i - begin, 0) = labels[i];
          if (!expectile) {
            info->labels.HostView()(i - begin, 1) = 20 + 2 * labels[i];
          }
        }
        info->weights_.HostVector().assign(weights.begin() + begin, weights.begin() + end);
      };
      MetaInfo full;
      set_info(0, labels.size(), &full);
      linalg::Vector<float> expected;
      FitInterceptRoot(ctx, full, loss, parameters, &expected);
      auto reference = expected.Data()->ConstHostVector();

      std::vector<int> passes(3);
      collective::TestDistributedGlobal(3, [&] {
        auto worker_ctx = ctx->IsCUDA() ? MakeCUDACtx(0) : Context{};
        worker_ctx.UpdateAllowUnknown(Args{{"nthread", "2"}});
        auto rank = collective::GetRank();
        MetaInfo local;
        if (rank == 0) {
          set_info(0, 2, &local);
        } else if (rank == 1) {
          set_info(2, labels.size(), &local);
        } else if (mode == 0) {
          // Exercise a genuinely label-less worker; the caller supplies the output count.
          local.num_row_ = 0;
        } else {
          local.num_row_ = 2;
          local.labels.Reshape(2, columns);
          std::fill(local.labels.Data()->HostVector().begin(),
                    local.labels.Data()->HostVector().end(), 1e30f);
          local.weights_.HostVector() = {0, 0};
        }
        linalg::Vector<float> actual;
        passes[rank] = FitInterceptRoot(&worker_ctx, local, loss, parameters, &actual);
        ASSERT_EQ(actual.Size(), reference.size());
        for (std::size_t j = 0; j < reference.size(); ++j) {
          EXPECT_NEAR(actual(j), reference[j], 1e-4);
        }
      });
      EXPECT_EQ(passes[0], passes[1]);
      EXPECT_EQ(passes[0], passes[2]);
      if (mode == 2) {
        EXPECT_EQ(passes[0], 1);
      }
    }
  }
}
}  // namespace

TEST(InterceptSolver, DistributedCPU) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"nthread", "2"}});
  TestDistributedRoot(&ctx);
}

#if defined(XGBOOST_USE_CUDA)
TEST(InterceptSolver, DistributedCUDA) {
  auto ctx = MakeCUDACtx(0);
  TestDistributedRoot(&ctx);
}
#endif
}  // namespace xgboost::obj
