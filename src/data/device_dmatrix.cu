/*!
 * Copyright 2020 by Contributors
 * \file device_dmatrix.cu
 * \brief Device-memory version of DMatrix.
 */

#include <xgboost/base.h>
#include <xgboost/data.h>

#include <memory>

#include "adapter.h"
#include "simple_dmatrix.h"
#include "device_dmatrix.h"
#include "device_adapter.cuh"
#include "../common/hist_util.h"

namespace xgboost {
namespace data {

// Does not currently support metainfo as no on-device data source contains this
// Current implementation assumes a single batch. More batches can
// be supported in future. Does not currently support inferring row/column size
  template <typename AdapterT>
DeviceDMatrix::DeviceDMatrix(AdapterT* adapter, float missing, int nthread) {
  common::HistogramCuts cuts = common::AdapterDeviceSketch(adapter, 256, missing);
  SimpleDMatrix dmat(adapter, missing, nthread);

  ellpack_page_.reset(new EllpackPage());
  auto ellpack_impl = ellpack_page_->Impl();
  ellpack_impl.reset(new EllpackPageImpl());
  info = dmat.Info();
  info.num_col_ = adapter->NumColumns();
  info.num_row_ = adapter->NumRows();
  // Synchronise worker columns
  rabit::Allreduce<rabit::op::Max>(&info.num_col_, 1);
}
template DeviceDMatrix::DeviceDMatrix(CudfAdapter* adapter, float missing,
                                      int nthread);
template DeviceDMatrix::DeviceDMatrix(CupyAdapter* adapter, float missing,
                                      int nthread);
}  // namespace data
}  // namespace xgboost
