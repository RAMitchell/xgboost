/*!
 * Copyright 2018 by Contributors
 * \file gbdct.cc
 * \brief Learns by building DCT approximations for each feature
 * \author Rory Mitchell
 */
#include <dmlc/parameter.h>
#include <xgboost/gbm.h>
#include <string>
#include <vector>
#include "../common/column_matrix.h"
#include "../common/dct.h"
#include "../common/hist_util.h"
#include "../common/timer.h"

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
    this->cuts = cuts;
    coefficients_.resize(cuts.row_ptr.back());
    predicted_weights_.resize(cuts.row_ptr.back());
  }
  float GetWeight(int bin_idx) { return predicted_weights_[bin_idx]; }
  void UpdateCoefficients(const std::vector<common::DCTCoefficient> &update,
                          int fidx, float learning_rate) {
    int begin_bin_idx = cuts.row_ptr[fidx];
    int end_bin_idx = cuts.row_ptr[fidx + 1];
    int num_coefficients =
        std::min(max_coefficients_, end_bin_idx - begin_bin_idx);
    for (auto i = 0; i < num_coefficients; i++) {
      coefficients_[begin_bin_idx + i] += update[i] * learning_rate;
    }
    // Update inverse DCT
    auto inverse = common::InverseDCT(std::vector<common::DCTCoefficient>(
        coefficients_.begin() + begin_bin_idx,
        coefficients_.begin() + end_bin_idx));
    std::copy(inverse.begin(), inverse.end(),
              predicted_weights_.begin() + cuts.row_ptr[fidx]);
  }

  std::string DumpModel() {
    std::stringstream ss;
    ss << "Max DCT coefficients per feature: " << max_coefficients_ << "\n";
    for (auto fidx = 0ull; fidx < coefficients_.size(); fidx++) {
      ss << "Feature " << fidx << ":\n";
      int begin_bin_idx = cuts.row_ptr[fidx];
      int end_bin_idx = cuts.row_ptr[fidx + 1];
      for (auto i = begin_bin_idx; i < end_bin_idx; i++) {
        ss << coefficients_[i] << " ";
      }
      ss << "\n";
    }
    return ss.str();
  }

 private:
  std::vector<common::DCTCoefficient> coefficients_;
  common::HistCutMatrix cuts;
  std::vector<float>
      predicted_weights_;  // The inverse DCT of the coefficients, keep this
                           // cached for prediction. predicted_weights[i] gives
                           // the weight for bin index i
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
    training_matrix_ptr = p_fmat;
    quantile_matrix_.Init(p_fmat, param_.max_bin);
    column_matrix_.Init(quantile_matrix_, 0.2);
    models_.resize(param_.num_output_group);
    for (auto &m : models_) {
      m.Init(param_.max_coefficients, p_fmat->Info().num_col_,
             quantile_matrix_.cut);
    }
    initialized_ = true;
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
        int bin_begin = quantile_matrix_.cut.row_ptr[fidx];
        int bin_end = quantile_matrix_.cut.row_ptr[fidx + 1];
        int N = bin_end - bin_begin;  // NOLINT
        std::vector<GradientPairPrecise> histogram =
            this->BuildFeatureHistogram(N, column_matrix_.GetColumn(fidx),
                                        host_gpair, gidx,
                                        param_.num_output_group);
        monitor_.Stop("Histogram");
        monitor_.Start("UpdateModel");
        auto feature_weights = this->GetFeatureWeights(histogram);
        auto update = common::TruncatedForwardDCT(feature_weights,
                                                  param_.max_coefficients);
        models_[gidx].UpdateCoefficients(update, fidx, param_.learning_rate);
        monitor_.Stop("UpdateModel");
        monitor_.Start("UpdateGradients");
        this->UpdateGradients(column_matrix_.GetColumn(fidx), &host_gpair, fidx,
                              update, gidx, param_.num_output_group);
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
    const std::vector<bst_float> &base_margin = p_fmat->Info().base_margin_.HostVector();
    if (p_fmat == training_matrix_ptr) {
      // If predicting from the training matrix take a shortcut
      this->PredictBatchTraining(&host_preds, base_margin);
      return;
    }
    auto iter = p_fmat->RowIterator();
    while (iter->Next()) {
      auto &batch = iter->Value();
      const auto nsize = static_cast<omp_ulong>(batch.Size());
#pragma omp parallel for schedule(static)
      for (omp_ulong i = 0; i < nsize; ++i) {
        const size_t ridx = batch.base_rowid + i;
        // Initialise margin
        for (auto gidx = 0ull; gidx < param_.num_output_group; gidx++) {
          bst_float margin =
              (base_margin.size() != 0)
                  ? base_margin[ridx * param_.num_output_group + gidx]
                  : base_margin_;
          host_preds[ridx * param_.num_output_group + gidx] = margin;
        }
        this->PredictInternal(
            batch[i], host_preds.data() + ridx * param_.num_output_group);
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
    this->PredictInternal(inst, out_preds->data());
  }

  void PredictLeaf(DMatrix *p_fmat, std::vector<bst_float> *out_preds,
                   unsigned ntree_limit) override {
    LOG(FATAL) << "gbdct does not support prediction of leaf index";
  }

  void PredictContribution(DMatrix *p_fmat,
                           std::vector<bst_float> *out_contribs,
                           unsigned ntree_limit, bool approximate,
                           int condition = 0,
                           unsigned condition_feature = 0) override {
    LOG(FATAL) << "gbdct does not support prediction contributions";
  }

  void PredictInteractionContributions(DMatrix *p_fmat,
                                       std::vector<bst_float> *out_contribs,
                                       unsigned ntree_limit,
                                       bool approximate) override {
    LOG(FATAL) << "gbdct does not support prediction interaction contributions";
  }

  std::vector<std::string> DumpModel(const FeatureMap &fmap, bool with_stats,
                                     std::string format) const override {
    return {""};
  }

 private:
  // Calculate the optimal weight update for each quantile
  std::vector<float> GetFeatureWeights(
      const std::vector<GradientPairPrecise> &histogram) {
    std::vector<float> x(histogram.size());
    for (auto i = 0; i < x.size(); i++) {
      auto g = histogram[i];
      x[i] = -g.GetGrad() / g.GetHess();
    }
    return x;
  }
  // If we are predicting from the training matrix take advantage of the fact
  // that we have an already quantised version
  void PredictBatchTraining(std::vector<bst_float> *out_preds,
                            const std::vector<float> &base_margin) {
    const auto nsize =
        static_cast<omp_ulong>(quantile_matrix_.row_ptr.size() - 1);
#pragma omp parallel for schedule(static)
    for (omp_ulong ridx = 0; ridx < nsize; ++ridx) {
      // Initialise margin
      for (auto gidx = 0; gidx < param_.num_output_group; gidx++) {
        bst_float margin =
            (base_margin.size() != 0)
                ? base_margin[ridx * param_.num_output_group + gidx]
                : base_margin_;
        (*out_preds)[ridx * param_.num_output_group + gidx] = margin;
      }

      for (auto gidx = 0; gidx < param_.num_output_group; gidx++) {
        float pred = 0;
        size_t row_begin = quantile_matrix_.row_ptr[ridx];
        size_t row_end = quantile_matrix_.row_ptr[ridx + 1];
        for (auto elem_idx = row_begin; elem_idx < row_end; elem_idx++) {
          int k = quantile_matrix_.index[elem_idx];
          pred += models_[gidx].GetWeight(k);
        }
        (*out_preds)[ridx * param_.num_output_group + gidx] += pred;
      }
    }
  }
  void PredictInternal(const SparsePage::Inst &inst, bst_float *out_preds) {
    if (!initialized_) return;
    for (const auto &elem : inst) {
      auto bin_idx = quantile_matrix_.cut.GetBinIdx(elem);
      for (auto gidx = 0ull; gidx < param_.num_output_group; gidx++) {
        out_preds[gidx] += models_[gidx].GetWeight(bin_idx);
      }
    }
  }
  std::vector<GradientPairPrecise> BuildFeatureHistogram(
      int feature_bins, const common::Column&column,
      const std::vector<GradientPair> &gradients, int group_idx,
      int num_group) {
    // Prepare thread local histograms
    std::vector<std::vector<GradientPairPrecise>> thread_histograms(
        omp_get_max_threads());
    for (auto &histogram : thread_histograms) {
      histogram.resize(feature_bins);
    }
    // Build histogram for each thread
    const auto nsize = static_cast<omp_ulong>(column.Size());
#pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < nsize; ++i) {
      if (column.IsMissing(i)) continue;
      auto feature_bin_idx = column.GetFeatureBinIdx(i);
      if (column.IsMissing(i)) continue;
      auto ridx = column.GetRowIdx(i);
      auto thread_idx = omp_get_thread_num();
      thread_histograms[thread_idx][feature_bin_idx] +=
          GradientPairPrecise(gradients[ridx * num_group + group_idx]);
    }
    std::vector<GradientPairPrecise> result(feature_bins);
    // Sum histograms back together
    for (const auto &local_histogram : thread_histograms) {
      for (auto i = 0; i < feature_bins; i++) {
        result[i] += local_histogram[i];
      }
    }
    return result;
  }
  void UpdateGradients(const common::Column&column,
                       std::vector<GradientPair> *gpair, int fidx,
                       const std::vector<common::DCTCoefficient> &update,
                       int group_idx, int num_group) {
    auto delta_prediction =
      common::InverseDCT(update);
    const auto nsize = static_cast<omp_ulong>(column.Size());
#pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < nsize; ++i) {
      if (column.IsMissing(i)) continue;
      auto feature_bin_idx = column.GetFeatureBinIdx(i);
      auto ridx = column.GetRowIdx(i);
      float d = delta_prediction[feature_bin_idx] * param_.learning_rate;
      auto &g = (*gpair)[ridx * num_group + group_idx];
      g += GradientPair(d * g.GetHess(), 0.0f);
    }
  }
  bool initialized_{false};
  GBDCTTrainParam param_;
  common::GHistIndexMatrix quantile_matrix_;
  common::ColumnMatrix column_matrix_;  // Quantised matrix in column format
  std::vector<DCTModel> models_;        // One model for each output class
  bst_float base_margin_;
  common::Monitor monitor_;
  DMatrix *training_matrix_ptr{
      nullptr};  // Remember the training matrix so we can reuse some
                 // information at prediction time
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
