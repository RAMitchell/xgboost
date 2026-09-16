/**
 * Copyright 2026, XGBoost Contributors
 */
#include <dmlc/registry.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cstddef>
#include <vector>

#include "../common/cuda_context.cuh"
#include "../common/optional_weight.h"
#include "intercept_solver.h"

namespace xgboost::obj {
DMLC_REGISTRY_FILE_TAG(intercept_solver_cuda);
namespace {
void InterceptStatsCuda(Context const* ctx, MetaInfo const& info, InterceptLoss loss,
                        common::Span<double const> parameters,
                        common::Span<InterceptPoint const> points, bool initialize,
                        std::vector<InterceptStats>* out) {
  auto labels = info.labels.View(ctx->Device());
  auto weights = common::MakeOptionalWeights(ctx->Device(), info.weights_);
  out->resize(parameters.size());
  for (std::size_t j = 0; j < parameters.size(); ++j) {
    auto column = loss == InterceptLoss::kExpectile ? 0 : j;
    auto parameter = parameters[j];
    auto point = initialize ? InterceptPoint{} : points[j];
    auto begin = thrust::make_counting_iterator(std::size_t{0});
    (*out)[j] = thrust::transform_reduce(
        ctx->CUDACtx()->CTP(), begin, begin + info.num_row_,
        [=] XGBOOST_DEVICE(std::size_t i) {
          return InterceptRow(labels(i, column), weights[i], loss, parameter, point, initialize);
        },
        InterceptStats{}, AddInterceptStats{});
  }
}
auto const kRegisterInterceptStatsCuda =
    common::KernelRegistration<InterceptStatsKernel>{DeviceOrd::kCUDA, &InterceptStatsCuda};
}  // namespace
}  // namespace xgboost::obj
