#include <dmlc/parameter.h>
#include <xgboost/gbm.h>
#include <string>
#include <vector>
#include "../common/dct.h"
#include "../common/hist_util.h"

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
        .set_default(0.3f)
        .describe("Learning rate(step size) of update.");
    DMLC_DECLARE_ALIAS(learning_rate, eta);
  }
};

class DCTModel {
 public:
  void Init(int max_bins, int max_coefficients, size_t num_columns) {
    this->max_bins_ = max_bins;
    this->max_coefficients_ = max_coefficients;
    coefficients_.resize(num_columns);
    for (auto &feature_coefficients : coefficients_) {
      feature_coefficients.resize(max_bins);
    }
  }
  void UpdateCoefficients(const std::vector<common::DCTCoefficient> &update,
                          int fidx, float learning_rate) {
    CHECK_LE(update.size(), coefficients_[fidx].size());
    for (auto i = 0; i < max_coefficients_; i++) {
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
  void BuildHistogram(const common::GHistIndexMatrix &quantile_matrix,
                      const std::vector<GradientPair> &gradients) {
    histogram.resize(quantile_matrix.cut.row_ptr.back());
    for (auto ridx = 0; ridx < quantile_matrix.row_ptr.size() - 1; ridx++) {
      for (auto j = quantile_matrix.row_ptr[ridx];
           j < quantile_matrix.row_ptr[ridx + 1]; j++) {
        auto entry = quantile_matrix.index[j];
        histogram[entry] += GradientPairPrecise(gradients[ridx]);
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
  }
  void Load(dmlc::Stream *fi) override {}
  void Save(dmlc::Stream *fo) const override {}

  void LazyInit(DMatrix *p_fmat) {
    if (initialized_) return;
    quantile_matrix_.Init(p_fmat, param_.max_bin);
    model_.Init(param_.max_bin, param_.max_coefficients,
                p_fmat->Info().num_col_);
    initialized_ = true;
  }
  void UpdateGradients(std::vector<GradientPair> *gpair, int fidx,
                       const std::vector<common::DCTCoefficient> &update) {
    // Update gradients
    int bin_begin = quantile_matrix_.cut.row_ptr[fidx];
    int bin_end = quantile_matrix_.cut.row_ptr[fidx + 1];
    for (auto i = 0ull; i < gpair->size(); i++) {
      auto row_begin = quantile_matrix_.row_ptr[i];
      auto row_end = quantile_matrix_.row_ptr[i + 1];
      for (auto j = row_begin; j < row_end; j++) {
        int bin_idx = quantile_matrix_.index[j];
        if (bin_idx >= bin_begin && bin_idx < bin_end) {
          int k = bin_idx - bin_begin;
          float d = common::InverseDCTSingle(update, update.size(), k) *
                    param_.learning_rate;
          auto &g = gpair->at(i);
          g += GradientPair(d * g.GetHess(), 0.0f);
        }
      }
    }
  }
  void DoBoost(DMatrix *p_fmat, HostDeviceVector<GradientPair> *in_gpair,
               ObjFunction *obj) override {
    this->LazyInit(p_fmat);
    auto &host_gpair = in_gpair->HostVector();

    // Find coefficients for each feature
    for (auto fidx = 0; fidx < p_fmat->Info().num_col_; fidx++) {
      GradientHistogram histogram;
      histogram.BuildHistogram(quantile_matrix_, host_gpair);
      int bin_begin = quantile_matrix_.cut.row_ptr[fidx];
      int bin_end = quantile_matrix_.cut.row_ptr[fidx + 1];
      int N = bin_end - bin_begin;  // NOLINT
      std::vector<float> x(N);
      for (auto i = 0; i < N; i++) {
        auto g = histogram.histogram[bin_begin + i];
        x[i] = -g.GetGrad() / g.GetHess();
      }
      x.resize(param_.max_bin, 0.0f);
      auto update = common::ForwardDCT(x, x.size(), param_.max_bin);
      // Truncate the new coefficients
      update.resize(param_.max_coefficients);
      update.resize(param_.max_bin, 0.0f);
      model_.UpdateCoefficients(update, fidx, param_.learning_rate);
      this->UpdateGradients(&host_gpair, fidx, update);
    }
  }

  void PredictBatch(DMatrix *p_fmat, HostDeviceVector<bst_float> *out_preds,
                    unsigned ntree_limit) override {
    out_preds->Resize(p_fmat->Info().num_row_);
    auto &host_preds = out_preds->HostVector();
    auto iter = p_fmat->RowIterator();
    while (iter->Next()) {
      auto &batch = iter->Value();
      for (auto i = 0; i < batch.Size(); i++) {
        std::vector<float> pred(1);
        this->PredictInstance(batch[i], &pred, 0, 0);
        host_preds.at(i + batch.base_rowid) = pred[0];
      }
    }
  }

  // add base margin
  void PredictInstance(const SparsePage::Inst &inst,
                       std::vector<bst_float> *out_preds, unsigned ntree_limit,
                       unsigned root_index) override {
    out_preds->at(0) = base_margin_;
    if (!initialized_) return;
    for (auto &elem : inst) {
      auto bin_idx = quantile_matrix_.cut.GetBinIdx(elem);
      int k = bin_idx - quantile_matrix_.cut.row_ptr[elem.index];
      (*out_preds)[0] += common::InverseDCTSingle(
          model_.GetCoefficients(elem.index),
          model_.GetCoefficients(elem.index).size(), k);
      // printf("prediction: %f\n", out_preds->back());
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
  bool initialized_{false};
  GBDCTTrainParam param_;
  common::GHistIndexMatrix quantile_matrix_;
  DCTModel model_;
  bst_float base_margin_;
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
