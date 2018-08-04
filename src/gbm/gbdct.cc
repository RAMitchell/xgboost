#include <dmlc/parameter.h>
#include <xgboost/gbm.h>
#include <string>
#include <vector>
#include "../common/dct.h"
#include "../common/hist_util.h"
#include "../common/timer.h"
#include "../common/column_matrix.h"

namespace xgboost {
namespace gbm {

DMLC_REGISTRY_FILE_TAG(gbdct);

// training parameters
struct GBDCTTrainParam : public dmlc::Parameter<GBDCTTrainParam> {
  int debug_verbose;
  // Maximum number of histogram bins per feature
  int max_bin;
  // Maximum number of DCT coefficients per feature
  int max_coefficients;
  float learning_rate;
  int num_output_group;
  DMLC_DECLARE_PARAMETER(GBDCTTrainParam) {
    DMLC_DECLARE_FIELD(debug_verbose)
        .set_lower_bound(0)
        .set_default(0)
        .describe("flag to print out detailed breakdown of runtime");
    DMLC_DECLARE_FIELD(max_bin).set_lower_bound(2).set_default(256).describe(
        "if using histogram-based algorithm, maximum number of bins per "
        "feature");
    DMLC_DECLARE_FIELD(max_coefficients)
        .set_lower_bound(1)
        .set_default(8)
        .describe("Maximum number of DCT coefficients for each feature");
    DMLC_DECLARE_FIELD(learning_rate)
        .set_lower_bound(0.0f)
        .set_default(0.5f)
        .describe("Learning rate(step size) of update.");
    DMLC_DECLARE_FIELD(num_output_group)
        .set_lower_bound(1)
        .set_default(1)
        .describe("Number of output groups in the setting.");
    DMLC_DECLARE_ALIAS(learning_rate, eta);
  }
};

class DCTModel {
 public:
  void Init(int max_coefficients, size_t num_columns,
            const common::HistCutMatrix &cuts) {
    this->max_coefficients_ = max_coefficients;
    coefficients_.resize(num_columns);
    for (auto fidx = 0ull; fidx < num_columns; fidx++) {
      int column_bins = cuts.row_ptr[fidx + 1] - cuts.row_ptr[fidx];
      coefficients_[fidx].resize(column_bins);
    }
  }
  void UpdateCoefficients(const std::vector<common::DCTCoefficient> &update,
                          int fidx, float learning_rate) {
    int num_coefficients =
        std::min(size_t(max_coefficients_), coefficients_[fidx].size());
    for (auto i = 0; i < num_coefficients; i++) {
      coefficients_[fidx][i] += update[i] * learning_rate;
    }
  }
  const std::vector<common::DCTCoefficient> &GetCoefficients(int fidx) {
    return coefficients_[fidx];
  }

 private:
  std::vector<std::vector<common::DCTCoefficient>> coefficients_;
  int max_bins_{0};
  int max_coefficients_{0};
};

class GradientHistogram {
 public:
  void BuildHistogram(const common::GHistIndexMatrix &quantile_matrix,const common::Column&column,
                      const std::vector<GradientPair> &gradients, int group_idx,
                      int num_group) {
//    histogram.resize(quantile_matrix.cut.row_ptr.back());
//    const auto nsize = static_cast<omp_ulong>(column.len);
//#pragma omp parallel for schedule(static)
//    for (omp_ulong i = 0; i < nsize; ++i) {
//      auto bin_idx = column.index[i] + column.index_base;
//      if (bin_idx == std::numeric_limits<T>::max()) continue;
//      auto ridx = i;
//      if (column.type == common::kSparseColumn) {
//        ridx = column.row_ind[i];
//      }
//      histogram[bin_idx] +=
//          GradientPairPrecise(gradients[ridx * num_group + group_idx]);
//    }
//  }
      histogram.resize(quantile_matrix.cut.row_ptr.back());
      const auto nsize =
          static_cast<omp_ulong>(quantile_matrix.row_ptr.size() - 1);
  #pragma omp parallel for schedule(static)
      for (omp_ulong ridx = 0; ridx < nsize; ++ridx) {
        for (auto j = quantile_matrix.row_ptr[ridx];
             j < quantile_matrix.row_ptr[ridx + 1]; j++) {
          auto entry = quantile_matrix.index[j];
          histogram[entry] +=
              GradientPairPrecise(gradients[ridx * num_group + group_idx]);
        }
      }
  }
  std::vector<GradientPairPrecise> histogram;
};

class GBDCT : public GradientBooster {
 public:
  explicit GBDCT(const std::vector<std::shared_ptr<DMatrix>> &cache,
                 bst_float base_margin)
      : base_margin_(base_margin) {}
  void Configure(
      const std::vector<std::pair<std::string, std::string>> &cfg) override {
    param_.InitAllowUnknown(cfg);
    monitor_.Init("GBDCT", param_.debug_verbose);
  }
  void Load(dmlc::Stream *fi) override {}
  void Save(dmlc::Stream *fo) const override {}

  void LazyInit(DMatrix *p_fmat) {
    if (initialized_) return;
    quantile_matrix_.Init(p_fmat, param_.max_bin);
    column_matrix_.Init(quantile_matrix_, 0.2);
    models_.resize(param_.num_output_group);
    for (auto &m : models_) {
      m.Init(param_.max_coefficients, p_fmat->Info().num_col_,
             quantile_matrix_.cut);
    }
    initialized_ = true;
  }
  void UpdateGradients(DMatrix *p_fmat, std::vector<GradientPair> *gpair,
                       int fidx,
                       const std::vector<common::DCTCoefficient> &update,
                       int group_idx, int num_group) {
    auto delta_prediction =
        common::InverseDCT(update, update.size(), update.size());
    int bin_begin = quantile_matrix_.cut.row_ptr[fidx];
    int bin_end = quantile_matrix_.cut.row_ptr[fidx + 1];
    const auto nsize = static_cast<omp_ulong>(p_fmat->Info().num_row_);
#pragma omp parallel for schedule(static)
    for (omp_ulong ridx = 0; ridx < nsize; ++ridx) {
      auto row_begin = quantile_matrix_.row_ptr[ridx];
      auto row_end = quantile_matrix_.row_ptr[ridx + 1];
      for (auto j = row_begin; j < row_end; j++) {
        int bin_idx = quantile_matrix_.index[j];
        if (bin_idx >= bin_begin && bin_idx < bin_end) {
          int k = bin_idx - bin_begin;
          float d = delta_prediction[k] * param_.learning_rate;
          auto &g = (*gpair)[ridx * num_group + group_idx];
          g += GradientPair(d * g.GetHess(), 0.0f);
        }
      }
    }
  }
  void DoBoost(DMatrix *p_fmat, HostDeviceVector<GradientPair> *in_gpair,
               ObjFunction *obj) override {
    monitor_.Start("Init");
    this->LazyInit(p_fmat);
    monitor_.Stop("Init");
    monitor_.Start("DoBoost");
    auto &host_gpair = in_gpair->HostVector();

    for (auto gidx = 0ull; gidx < param_.num_output_group; gidx++) {
      // Find coefficients for each feature
      for (auto fidx = 0ull; fidx < p_fmat->Info().num_col_; fidx++) {
        monitor_.Start("Histogram");
        GradientHistogram histogram;
        histogram.BuildHistogram(quantile_matrix_,column_matrix_.GetColumn(fidx) ,host_gpair, gidx,
                                 param_.num_output_group);
        monitor_.Stop("Histogram");
        monitor_.Start("UpdateModel");
        int bin_begin = quantile_matrix_.cut.row_ptr[fidx];
        int bin_end = quantile_matrix_.cut.row_ptr[fidx + 1];
        int N = bin_end - bin_begin;  // NOLINT
        std::vector<float> x(N);
        for (auto i = 0; i < N; i++) {
          auto g = histogram.histogram[bin_begin + i];
          x[i] = -g.GetGrad() / g.GetHess();
        }
        auto update = common::ForwardDCT(x, x.size(), x.size());
        // Truncate the new coefficients
        update.resize(param_.max_coefficients);
        update.resize(N, 0.0f);
        models_[gidx].UpdateCoefficients(update, fidx, param_.learning_rate);
        monitor_.Stop("UpdateModel");
        monitor_.Start("UpdateGradients");
        this->UpdateGradients(p_fmat, &host_gpair, fidx, update, gidx,
                              param_.num_output_group);
        monitor_.Stop("UpdateGradients");
      }
    }
    monitor_.Stop("DoBoost");
  }

  void PredictBatch(DMatrix *p_fmat, HostDeviceVector<bst_float> *out_preds,
                    unsigned ntree_limit) override {
    monitor_.Start("PredictBatch");
    out_preds->Resize(p_fmat->Info().num_row_ * param_.num_output_group);
    auto &host_preds = out_preds->HostVector();
    auto iter = p_fmat->RowIterator();
    while (iter->Next()) {
      auto &batch = iter->Value();
      const auto nsize = static_cast<omp_ulong>(batch.Size());
#pragma omp parallel for schedule(static)
      for (omp_ulong i = 0; i < nsize; ++i) {
        const size_t ridx = batch.base_rowid + i;
        std::vector<float> pred(param_.num_output_group);
        this->PredictInstance(batch[i], &pred, 0, 0);
        for (auto gidx = 0; gidx < param_.num_output_group; gidx++) {
          host_preds[ridx * param_.num_output_group + gidx] = pred[gidx];
        }
      }
    }
    monitor_.Stop("PredictBatch");
  }

  void PredictInstance(const SparsePage::Inst &inst,
                       std::vector<bst_float> *out_preds, unsigned ntree_limit,
                       unsigned root_index) override {
    for (auto gidx = 0ull; gidx < param_.num_output_group; gidx++) {
      (*out_preds)[gidx] = base_margin_;
    }
    if (!initialized_) return;
    for (auto &elem : inst) {
      auto bin_idx = quantile_matrix_.cut.GetBinIdx(elem);
      int k = bin_idx - quantile_matrix_.cut.row_ptr[elem.index];
      for (auto gidx = 0ull; gidx < param_.num_output_group; gidx++) {
        (*out_preds)[gidx] += common::InverseDCTSingle(
            models_[gidx].GetCoefficients(elem.index),
            models_[gidx].GetCoefficients(elem.index).size(), k);
      }
    }
  }

  void PredictLeaf(DMatrix *p_fmat, std::vector<bst_float> *out_preds,
                   unsigned ntree_limit) override {
    LOG(FATAL) << "gbdct does not support prediction of leaf index";
  }

  void PredictContribution(DMatrix *p_fmat,
                           std::vector<bst_float> *out_contribs,
                           unsigned ntree_limit, bool approximate,
                           int condition = 0,
                           unsigned condition_feature = 0) override {}

  void PredictInteractionContributions(DMatrix *p_fmat,
                                       std::vector<bst_float> *out_contribs,
                                       unsigned ntree_limit,
                                       bool approximate) override {}

  std::vector<std::string> DumpModel(const FeatureMap &fmap, bool with_stats,
                                     std::string format) const override {
    return {""};
  }

 private:
  void PredictInternal(const SparsePage::Inst &inst,
                       std::vector<bst_float> *out_preds, unsigned ntree_limit,
                       unsigned root_index) {}
  bool initialized_{false};
  GBDCTTrainParam param_;
  common::GHistIndexMatrix quantile_matrix_;
  common::ColumnMatrix column_matrix_;  // Quantised matrix in column format
  std::vector<DCTModel> models_;  // One model for each class
  bst_float base_margin_;
  common::Monitor monitor_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(GBDCTTrainParam);

XGBOOST_REGISTER_GBM(GBDCT, "gbdct")
    .describe("Discrete cosine transform booster")
    .set_body([](const std::vector<std::shared_ptr<DMatrix>> &cache,
                 bst_float base_margin) {
      return new GBDCT(cache, base_margin);
    });
}  // namespace gbm
}  // namespace xgboost
