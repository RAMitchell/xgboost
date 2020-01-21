
// Copyright (c) 2019 by Contributors
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include "../../../src/data/adapter.h"
#include "../../../src/data/device_dmatrix.h"
#include "../helpers.h"
#include <thrust/device_vector.h>
#include "../../../src/data/device_adapter.cuh"
#include "../../../src/gbm/gbtree_model.h"
using namespace xgboost;  // NOLINT

TEST(DeviceDMatrix, Simple) {
  int num_rows = 10;
  int num_columns = 2;
  auto *dmat = CreateDMatrix(num_rows, num_columns, 0.0);

  data::DeviceDMatrix device_dmat(dmat->get());

  auto &batch = *device_dmat.GetBatches<EllpackPage>({0,256,0}).begin();

  auto gpu_lparam = CreateEmptyGenericParam(0);
  auto cache = std::make_shared<std::unordered_map<DMatrix*, PredictionCacheEntry>>();

  std::unique_ptr<Predictor> gpu_predictor = std::unique_ptr<Predictor>(
      Predictor::Create("gpu_predictor", &gpu_lparam, cache));

  gpu_predictor->Configure({});
  LearnerModelParam param;
  gbm::GBTreeModel model = CreateTestModel(&param);
  HostDeviceVector<float> predictions(num_rows);
  gpu_predictor->PredictBatch(&device_dmat, &predictions, model, 0);
  delete dmat;
}
