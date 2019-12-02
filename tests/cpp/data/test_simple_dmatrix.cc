// Copyright by Contributors
#include <xgboost/data.h>
#include <dmlc/filesystem.h>
#include "../../../src/data/simple_dmatrix.h"

#include "../helpers.h"
#include "../../../src/data/adapter.h"

using namespace xgboost;  // NOLINT

TEST(SimpleDMatrix, MetaInfo) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);

  // Test the metadata that was parsed
  EXPECT_EQ(dmat->Info().num_row_, 2);
  EXPECT_EQ(dmat->Info().num_col_, 5);
  EXPECT_EQ(dmat->Info().num_nonzero_, 6);
  EXPECT_EQ(dmat->Info().labels_.Size(), dmat->Info().num_row_);

  delete dmat;
}

TEST(SimpleDMatrix, RowAccess) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, false, false);

  // Loop over the batches and count the records
  int64_t row_count = 0;
  for (auto &batch : dmat->GetBatches<xgboost::SparsePage>()) {
    row_count += batch.Size();
  }
  EXPECT_EQ(row_count, dmat->Info().num_row_);
  // Test the data read into the first row
  auto &batch = *dmat->GetBatches<xgboost::SparsePage>().begin();
  auto first_row = batch[0];
  ASSERT_EQ(first_row.size(), 3);
  EXPECT_EQ(first_row[2].index, 2);
  EXPECT_EQ(first_row[2].fvalue, 20);

  delete dmat;
}

TEST(SimpleDMatrix, ColAccessWithoutBatches) {
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/simple.libsvm";
  CreateSimpleTestData(tmp_file);
  xgboost::DMatrix * dmat = xgboost::DMatrix::Load(tmp_file, true, false);

  // Sorted column access
  EXPECT_EQ(dmat->GetColDensity(0), 1);
  EXPECT_EQ(dmat->GetColDensity(1), 0.5);
  ASSERT_TRUE(dmat->SingleColBlock());

  // Loop over the batches and assert the data is as expected
  int64_t num_col_batch = 0;
  for (const auto &batch : dmat->GetBatches<xgboost::SortedCSCPage>()) {
    num_col_batch += 1;
    EXPECT_EQ(batch.Size(), dmat->Info().num_col_)
        << "Expected batch size = number of cells as #batches is 1.";
  }
  EXPECT_EQ(num_col_batch, 1) << "Expected number of batches to be 1";
  delete dmat;
}

TEST(SimpleDMatrix, Empty) {
  std::vector<float> data{};
  std::vector<unsigned> feature_idx = {};
  std::vector<size_t> row_ptr = {};

  data::CSRAdapter csr_adapter(row_ptr.data(), feature_idx.data(), data.data(), 0, 0, 0);
  data::SimpleDMatrix dmat(&csr_adapter,
                           std::numeric_limits<float>::quiet_NaN(), 1);
  CHECK_EQ(dmat.Info().num_nonzero_, 0);
  CHECK_EQ(dmat.Info().num_row_, 0);
  CHECK_EQ(dmat.Info().num_col_, 0);
  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    CHECK_EQ(batch.Size(), 0);
  }

  data::DenseAdapter dense_adapter(nullptr, 0, 0, 0);
  dmat = data::SimpleDMatrix(&dense_adapter,
                             std::numeric_limits<float>::quiet_NaN(), 1);
  CHECK_EQ(dmat.Info().num_nonzero_, 0);
  CHECK_EQ(dmat.Info().num_row_, 0);
  CHECK_EQ(dmat.Info().num_col_, 0);
  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    CHECK_EQ(batch.Size(), 0);
  }

  data::CSCAdapter csc_adapter(nullptr, nullptr, nullptr, 0, 0);
  dmat = data::SimpleDMatrix(&csc_adapter,
                             std::numeric_limits<float>::quiet_NaN(), 1);
  CHECK_EQ(dmat.Info().num_nonzero_, 0);
  CHECK_EQ(dmat.Info().num_row_, 0);
  CHECK_EQ(dmat.Info().num_col_, 0);
  for (auto &batch : dmat.GetBatches<SparsePage>()) {
    CHECK_EQ(batch.Size(), 0);
  }
}

TEST(SimpleDMatrix, MissingData) {
  std::vector<float> data{0.0, std::nanf(""), 1.0};
  std::vector<unsigned> feature_idx = {0, 1, 0};
  std::vector<size_t> row_ptr = {0, 2, 3};

  data::CSRAdapter adapter(row_ptr.data(), feature_idx.data(), data.data(), 2, 3, 2);
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(), 1);
  CHECK_EQ(dmat.Info().num_nonzero_, 2);
  dmat = data::SimpleDMatrix(&adapter, 1.0, 1);
  CHECK_EQ(dmat.Info().num_nonzero_, 1);
}

TEST(SimpleDMatrix, EmptyRow) {
  std::vector<float> data{0.0, 1.0};
  std::vector<unsigned> feature_idx = {0, 1};
  std::vector<size_t> row_ptr = {0, 2, 2};

  data::CSRAdapter adapter(row_ptr.data(), feature_idx.data(), data.data(), 2, 2, 2);
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(), 1);
  CHECK_EQ(dmat.Info().num_nonzero_, 2);
  CHECK_EQ(dmat.Info().num_row_, 2);
  CHECK_EQ(dmat.Info().num_col_, 2);
}
