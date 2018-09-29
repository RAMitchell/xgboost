/*!
 * Copyright 2018 by Contributors
 * \file gbdct.cc
 * \brief Learns by building DCT approximations for each feature
 * \author Rory Mitchell
 */
#include <dmlc/parameter.h>
#include <xgboost/gbm.h>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include "../common/column_matrix.h"
#include "../common/hist_util.h"
#include "../common/random.h"
#include "../common/timer.h"

namespace xgboost {
namespace gbm {
#ifndef GTEST_INCLUDE_GTEST_GTEST_H_
DMLC_REGISTRY_FILE_TAG(gbdct);
#endif

constexpr float kPi = 3.14159265358979f;
using DCTCoefficient = double;

enum RegularisingTransformType { kDCT, kHaar, kIdentity,kRandomProjection };

// training parameters
struct GBDCTTrainParam : public dmlc::Parameter<GBDCTTrainParam> {
  int debug_verbose;
  // Maximum number of histogram bins per feature
  int max_bin;
  // Maximum number of DCT coefficients per feature
  int max_coefficients;
  float learning_rate;
  int num_output_group;
  int regularising_transform;
  int random_projection_seed;
  DMLC_DECLARE_PARAMETER(GBDCTTrainParam) {
    DMLC_DECLARE_FIELD(debug_verbose)
        .set_lower_bound(0)
        .set_default(0)
        .describe("flag to print out detailed breakdown of runtime");
    DMLC_DECLARE_FIELD(max_bin).set_lower_bound(2).set_default(64).describe(
        "if using histogram-based algorithm, maximum number of bins per "
        "feature");
    DMLC_DECLARE_FIELD(max_coefficients)
        .set_lower_bound(1)
        .set_default(64)
        .describe("Maximum number of DCT coefficients for each feature");
    DMLC_DECLARE_FIELD(learning_rate)
        .set_lower_bound(0.0f)
        .set_default(0.5f)
        .describe("Learning rate(step size) of update.");
    DMLC_DECLARE_FIELD(num_output_group)
        .set_lower_bound(1)
        .set_default(1)
        .describe("Number of output groups in the setting.");
    DMLC_DECLARE_FIELD(random_projection_seed)
        .set_lower_bound(0)
        .set_default(0)
        .describe("Seed for random projection transform.");
    DMLC_DECLARE_FIELD(regularising_transform)
        .set_default(kDCT)
        .add_enum("dct", kDCT)
        .add_enum("haar", kHaar)
        .add_enum("identity", kIdentity)
        .add_enum("rp", kRandomProjection)
        .describe("Transform type to use.");
    DMLC_DECLARE_ALIAS(learning_rate, eta);
  }
};

template <typename T>
class Matrix {
  std::vector<T> data_;
  const size_t m_;
  const size_t n_;

 public:
  Matrix(size_t m, size_t n) : m_(m), n_(n) { data_.resize(m * n); }
  size_t Rows() const { return m_; }
  size_t Columns() const { return n_; }
  T &operator()(size_t i, size_t j) { return data_[i * n_ + j]; }
  void Print() const {
    for (auto i = 0ull; i < m_; i++) {
      for (auto j = 0ull; j < n_; j++) {
        printf("%f ", data_[i * n_ + j]);
      }
      printf("\n");
    }
  }

  static Matrix<T> Diagonal(const std::vector<float> &x) {
    Matrix<T> m(x.size(), x.size());
    for (auto i = 0ull; i < x.size(); i++) {
      m(i, i) = x[i];
    }
    return m;
  }

  static Matrix<T> Identity(size_t m, size_t n) {
    Matrix<T> I(m, n);
    for (auto i = 0ull; i < I.Rows(); i++) {
      if (i < I.Columns()) {
        I(i, i) = 1.0f;
      }
    }
    return I;
  }

  static Matrix<T> RandomProjection(size_t m, size_t n, int seed) {
    common::RandomEngine re(seed);
    std::uniform_real_distribution<T> distribution(-1.0, 1.0);
    Matrix<T> RP(m, n);
    for (auto j = 0ull; j < n; j++) {
      // Generate random vector
      for (auto i = 0ull; i < m; i++) {
        RP(i, j) = distribution(re);
      }
      // Normalise j
      double length = 0.0;
      for (auto i = 0ull; i < m; i++) {
        length += RP(i, j) * RP(i, j);
      }
      length = std::sqrt(length);
      for (auto i = 0ull; i < m; i++) {
        RP(i, j) /= length;
      }

      // Orthogonalise
      for (auto jj = 0ull; jj < j; jj++) {
        // Project j onto jj
        double projection = 0.0;
        for (auto i = 0ull; i < m; i++) {
          projection += RP(i, j) * RP(i, jj);
        }
        // Remove projection on j
        for (auto i = 0ull; i < m; i++) {
          RP(i, j) -= projection * RP(i, jj);
        }
        // Normalise j
        double length = 0.0;
        for (auto i = 0ull; i < m; i++) {
          length += RP(i, j) * RP(i, j);
        }
        length = std::sqrt(length);
        for (auto i = 0ull; i < m; i++) {
          RP(i, j) /= length;
        }
      }
    }
    return RP;
  }

  // http://fourier.eng.hmc.edu/e161/lectures/dct/node1.html
  static Matrix<T> InverseDCT(size_t m, size_t n) {
    Matrix<T> inv_DCT(m, n);
    for (auto j = 0ull; j < n; j++) {
      for (auto i = 0ull; i < m; i++) {
        T a = j == 0 ? sqrt(1.0 / m) : sqrt(2.0 / m);
        inv_DCT(i, j) = a * cos(((2.0 * i + 1.0) * j * kPi) / (2.0 * m));
      }
    }
    return inv_DCT;
  }

  // http://fourier.eng.hmc.edu/e161/lectures/Haar/index.html
  static Matrix<T> InverseHaar(size_t m, size_t n) {
    Matrix<T> haar(m, n);
    const double kScaleFactor = 1.0 / std::sqrt(m);
    for (auto j = 0ull; j < n; j++) {
      double p = std::floor(std::log2(static_cast<double>(j)));
      double exp_p = std::pow(2.0, p);
      double exp_half_p = std::pow(2.0, p / 2);
      double q = (j - std::pow(2.0, p)) + 1.0;
      for (auto i = 0ull; i < m; i++) {
        double t = static_cast<double>(i) / m;
        if (j == 0) {
          haar(i, j) = kScaleFactor;
        } else if ((q - 1.0) / exp_p <= t && t < (q - 0.5) / exp_p) {
          haar(i, j) = kScaleFactor * exp_half_p;
        } else if ((q - 0.5) / exp_p <= t && t < q / exp_p) {
          haar(i, j) = kScaleFactor * -exp_half_p;
        } else {
          haar(i, j) = 0.0;
        }
      }
    }
    return haar;
  }

  Matrix<T> Transpose() const {
    Matrix<T> AT(n_, m_);
    for (auto i = 0ull; i < AT.m_; i++) {
      for (auto j = 0ull; j < AT.n_; j++) {
        AT.data_[i * AT.n_ + j] = data_[j * n_ + i];
      }
    }
    return AT;
  }

  Matrix<T> operator*(const Matrix<T> &B) const {
    CHECK_EQ(n_, B.Rows());
    Matrix<T> C(this->Rows(), B.Columns());
    for (auto i = 0ull; i < m_; i++) {
      for (auto j = 0ull; j < n_; j++) {
        for (auto k = 0ull; k < B.Columns(); k++) {
          C.data_[i * B.Columns() + k] +=
              data_[i * n_ + j] * B.data_[j * B.Columns() + k];
        }
      }
    }
    return C;
  }

  std::vector<T> operator*(const std::vector<T> &x) const {
    CHECK_EQ(Columns(), x.size());
    std::vector<T> result(Rows());
    for (auto i = 0ull; i < Rows(); i++) {
      for (auto j = 0ull; j < Columns(); j++) {
        result[i] += data_[i * Columns() + j] * x[j];
      }
    }
    return result;
  }

  static std::vector<T> CholeskySolve(const Matrix<T> &A,
                                      const std::vector<T> &b) {
    CHECK_EQ(A.Columns(), A.Rows()) << "Matrix must be square.";
    CHECK_EQ(A.Rows(), b.size());
    size_t n = A.Rows();

    // Calculate cholesky decomposition LLt = A
    Matrix<T> L = A;
    for (auto i = 0ull; i < n; i++) {
      for (auto j = i; j < n; j++) {
        double sum = L(i, j);
        for (int64_t k = i - 1; k >= 0; k--) {
          sum = sum - L(i, k) * L(j, k);
        }

        if (i == j) {
          if (sum <= 0.0) sum = 1e-5;
          L(i, i) = sqrt(sum);
        } else {
          L(j, i) = sum / L(i, i);
        }
      }
    }

    for (auto i = 0ull; i < n; i++) {
      for (auto j = 0ull; j < i; j++) {
        L(j, i) = 0.0;
      }
    }

    // Solve linear equations
    std::vector<T> x(n);
    for (auto i = 0ull; i < n; i++) {
      double sum = b[i];
      for (int64_t k = i - 1; k >= 0; k--) {
        sum = sum - L(i, k) * x[k];
      }
      x[i] = sum / L(i, i);
    }

    for (int64_t i = n - 1; i >= 0; i--) {
      double sum = x[i];
      for (auto k = i + 1; k < n; k++) {
        sum = sum - L(k, i) * x[k];
      }
      x[i] = sum / L(i, i);
    }

    return x;
  }
};

class RTLModel {
 public:
  void Init(const GBDCTTrainParam &param, const common::HistCutMatrix &cuts) {
    this->max_coefficients_ = param.max_coefficients;
    this->cuts = cuts;
    this->regularising_transform = static_cast<RegularisingTransformType>(param.regularising_transform);
    this->seed_ = param.random_projection_seed;
    coefficients_.resize(cuts.row_ptr.back());
    predicted_weights_.resize(cuts.row_ptr.back());
  }
  float GetWeight(int bin_idx) { return predicted_weights_[bin_idx]; }

  Matrix<double> GetTransform(size_t m, size_t n) {
    if (regularising_transform == kDCT) {
      return Matrix<double>::InverseDCT(m, n);
    } else if (regularising_transform == kHaar) {
      return Matrix<double>::InverseHaar(m, n);
    } else if (regularising_transform == kIdentity) {
      return Matrix<double>::Identity(m, n);
    } else if (regularising_transform == kRandomProjection) {
      return Matrix<double>::RandomProjection(m, n, seed_);
    }
    LOG(FATAL) << "Unknown regularising transform: " << regularising_transform;
    return Matrix<double>::Identity(m, n);
  }

  std::vector<double> UpdateCoefficients(
      const std::vector<GradientPairPrecise> &train_histogram, int fidx,
      float learning_rate) {
    int begin_bin_idx = cuts.row_ptr[fidx];
    int end_bin_idx = cuts.row_ptr[fidx + 1];
    int num_coefficients =
        std::min(end_bin_idx - begin_bin_idx, this->max_coefficients_);
    auto T = this->GetTransform(train_histogram.size(), num_coefficients);
    auto Tt = T.Transpose();
    Matrix<double> H(train_histogram.size(), train_histogram.size());
    std::vector<double> grad(train_histogram.size());
    for (auto i = 0ull; i < train_histogram.size(); i++) {
      // Don't allow the hessian values to become too small, otherwise numerical
      // precision issues
      H(i, i) = std::max(train_histogram[i].GetHess(), 1e-5f);
      grad[i] = train_histogram[i].GetGrad();
    }
    grad = Tt * grad;
    auto TtHT = Tt * H * T;
    auto update = Matrix<double>::CholeskySolve(TtHT, grad);
    for (auto &coefficient : update) {
      coefficient *= -learning_rate;
    }

    for (auto i = 0ull; i < update.size(); i++) {
      coefficients_[begin_bin_idx + i] += update[i];
    }
    // Update inverse DCT
    auto inverse =
        T * std::vector<DCTCoefficient>(
                coefficients_.begin() + begin_bin_idx,
                coefficients_.begin() + begin_bin_idx + num_coefficients);
    CHECK_EQ(inverse.size(), end_bin_idx - begin_bin_idx);
    std::copy(inverse.begin(), inverse.end(),
              predicted_weights_.begin() + cuts.row_ptr[fidx]);
    return T * update;
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
  std::vector<DCTCoefficient> coefficients_;
  common::HistCutMatrix cuts;
  std::vector<float>
      predicted_weights_;  // The inverse DCT of the coefficients, keep this
                           // cached for prediction. predicted_weights[i] gives
                           // the weight for bin index i
  int max_coefficients_{0};
  RegularisingTransformType regularising_transform{kDCT};
  int seed_{0}; // Used if random projection
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
      m.Init(param_, quantile_matrix_.cut);
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
        std::vector<GradientPairPrecise> train_histogram;
        this->BuildFeatureHistograms(N, column_matrix_.GetColumn(fidx),
                                     host_gpair, gidx, param_.num_output_group,
                                     &train_histogram);
        monitor_.Stop("Histogram");
        monitor_.Start("UpdateModel");
        auto update = models_[gidx].UpdateCoefficients(train_histogram, fidx,
                                                       param_.learning_rate);
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
    const std::vector<bst_float> &base_margin =
        p_fmat->Info().base_margin_.HostVector();
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
  void BuildFeatureHistograms(
      int feature_bins, const common::Column &column,
      const std::vector<GradientPair> &gradients, int group_idx, int num_group,
      std::vector<GradientPairPrecise> *train_histogram) {
    // Prepare thread local histograms
    std::vector<std::vector<GradientPairPrecise>> thread_train_histograms(
        omp_get_max_threads());

    for (auto &histogram : thread_train_histograms) {
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

      thread_train_histograms[thread_idx][feature_bin_idx] +=
          GradientPairPrecise(gradients[ridx * num_group + group_idx]);
    }
    train_histogram->resize(feature_bins);

    // Sum histograms back together
    for (const auto &local_histogram : thread_train_histograms) {
      for (auto i = 0; i < feature_bins; i++) {
        (*train_histogram)[i] += local_histogram[i];
      }
    }
  }
  void UpdateGradients(const common::Column &column,
                       std::vector<GradientPair> *gpair, int fidx,
                       const std::vector<double> &weight_updates, int group_idx,
                       int num_group) {
    const auto nsize = static_cast<omp_ulong>(column.Size());
#pragma omp parallel for schedule(static)
    for (omp_ulong i = 0; i < nsize; ++i) {
      if (column.IsMissing(i)) continue;
      auto feature_bin_idx = column.GetFeatureBinIdx(i);
      auto ridx = column.GetRowIdx(i);
      float d = weight_updates[feature_bin_idx];
      auto &g = (*gpair)[ridx * num_group + group_idx];
      g += GradientPair(d * g.GetHess(), 0.0f);
    }
  }
  bool initialized_{false};
  GBDCTTrainParam param_;
  common::GHistIndexMatrix quantile_matrix_;
  common::ColumnMatrix column_matrix_;  // Quantised matrix in column format
  std::vector<RTLModel> models_;        // One model for each output class
  bst_float base_margin_;
  common::Monitor monitor_;
  DMatrix *training_matrix_ptr{
      nullptr};  // Remember the training matrix so we can reuse some
                 // information at prediction time
};

// register the objective functions
#ifndef GTEST_INCLUDE_GTEST_GTEST_H_
DMLC_REGISTER_PARAMETER(GBDCTTrainParam);

XGBOOST_REGISTER_GBM(GBDCT, "gbdct")
    .describe("Discrete cosine transform booster")
    .set_body([](const std::vector<std::shared_ptr<DMatrix>> &cache,
                 bst_float base_margin) {
      return new GBDCT(cache, base_margin);
    });
#endif
}  // namespace gbm
}  // namespace xgboost
