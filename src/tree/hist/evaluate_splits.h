/**
 * Copyright 2021-2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
#define XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_

#include <algorithm>  // for copy
#include <array>      // for array
#include <cstddef>    // for size_t
#include <limits>     // for numeric_limits
#include <memory>     // for shared_ptr
#include <numeric>    // for accumulate
#include <utility>    // for move
#include <vector>     // for vector

#include "../../common/categorical.h"  // for CatBitField
#include "../../common/hist_util.h"    // for GHistRow, HistogramCuts
#include "../../common/linalg_op.h"    // for cbegin, cend, begin
#include "../../common/random.h"       // for ColumnSampler
#include "../constraints.h"            // for FeatureInteractionConstraintHost
#include "../param.h"                  // for TrainParam
#include "../split_evaluator.h"        // for TreeEvaluator
#include "../tree_view.h"              // for MultiTargetTreeView
#include "expand_entry.h"              // for MultiExpandEntry
#include "hist_cache.h"                // for BoundedHistCollection
#include "xgboost/base.h"              // for bst_node_t, bst_target_t, bst_feature_t
#include "xgboost/context.h"           // for COntext
#include "xgboost/linalg.h"            // for Constants, Vector

namespace xgboost::tree {
class HistMultiEvaluator {
  std::vector<double> gain_;
  linalg::Matrix<GradientPairPrecise> stats_;
  TrainParam const *param_;
  TreeEvaluator tree_evaluator_;
  FeatureInteractionConstraintHost interaction_constraints_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  Context const *ctx_;

 private:
  template <bst_bin_t d_step, bool kSingleTarget>
  bool EnumerateSplitImpl(common::HistogramCuts const &cut, bst_feature_t fidx,
                          common::Span<common::ConstGHistRow> hist,
                          linalg::VectorView<GradientPairPrecise const> parent_sum,
                          double parent_gain, bst_node_t nidx,
                          TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                          SplitEntryContainer<std::vector<GradientPairPrecise>> *p_best) const {
    auto const &cut_ptr = cut.Ptrs();
    auto const &cut_val = cut.Values();

    auto scan = [&](auto &sums) {
      // Construct views here to keep the one-target extent visible to the compiler.
      auto [left_sum, right_sum] = [&] {
        if constexpr (kSingleTarget) {
          return std::make_pair(linalg::MakeVec(sums.data(), 1),
                                linalg::MakeVec(sums.data() + 1, 1));
        } else {
          return std::make_pair(sums.Slice(0, linalg::All()), sums.Slice(1, linalg::All()));
        }
      }();
      auto n_targets = kSingleTarget ? 1 : hist.size();
      bst_bin_t ibegin, iend;
      if (d_step > 0) {
        ibegin = static_cast<bst_bin_t>(cut_ptr[fidx]);
        iend = static_cast<bst_bin_t>(cut_ptr[fidx + 1]);
      } else {
        ibegin = static_cast<bst_bin_t>(cut_ptr[fidx + 1]) - 1;
        iend = static_cast<bst_bin_t>(cut_ptr[fidx]) - 1;
      }
      for (bst_bin_t i = ibegin; i != iend; i += d_step) {
        for (bst_target_t t = 0; t < n_targets; ++t) {
          auto t_hist = hist[t];
          auto t_p = parent_sum(t);
          left_sum(t) += t_hist[i];
          right_sum(t) = t_p - left_sum(t);
        }

        if (d_step > 0) {
          auto split_pt = cut_val[i];
          auto loss_chg =
              evaluator.CalcSplitGain(*param_, nidx, fidx, left_sum, right_sum) - parent_gain;
          p_best->Update(loss_chg, fidx, split_pt, d_step == -1, false, left_sum, right_sum);
        } else {
          auto split_pt = common::HistogramCuts::NumericBinLowerBound(cut_ptr, cut_val, fidx, i);
          auto loss_chg =
              evaluator.CalcSplitGain(*param_, nidx, fidx, right_sum, left_sum) - parent_gain;
          p_best->Update(loss_chg, fidx, split_pt, d_step == -1, false, right_sum, left_sum);
        }
      }
      // return true if there's missing. Doesn't handle floating-point error well.
      if (d_step == +1) {
        return !std::equal(linalg::cbegin(left_sum), linalg::cend(left_sum),
                           linalg::cbegin(parent_sum));
      }
      return false;
    };

    if constexpr (kSingleTarget) {
      std::array<GradientPairPrecise, 2> sums{};
      return scan(sums);
    } else {
      auto sums = linalg::Empty<GradientPairPrecise>(ctx_, 2, hist.size());
      return scan(sums);
    }
  }

  template <bst_bin_t d_step>
  bool EnumerateSplit(common::HistogramCuts const &cut, bst_feature_t fidx,
                      common::Span<common::ConstGHistRow> hist,
                      linalg::VectorView<GradientPairPrecise const> parent_sum, double parent_gain,
                      bst_node_t nidx, TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                      SplitEntryContainer<std::vector<GradientPairPrecise>> *p_best) const {
    if (hist.size() == 1) {
      return this->EnumerateSplitImpl<d_step, true>(cut, fidx, hist, parent_sum, parent_gain, nidx,
                                                    evaluator, p_best);
    }
    return this->EnumerateSplitImpl<d_step, false>(cut, fidx, hist, parent_sum, parent_gain, nidx,
                                                   evaluator, p_best);
  }

  void EnumerateOneHot(common::HistogramCuts const &cut, bst_feature_t fidx,
                       common::Span<common::ConstGHistRow> hist,
                       linalg::VectorView<GradientPairPrecise const> parent_sum, double parent_gain,
                       bst_node_t nidx, TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                       SplitEntryContainer<std::vector<GradientPairPrecise>> *p_best) const {
    auto const &cut_ptr = cut.Ptrs();
    auto const &cut_val = cut.Values();

    bst_bin_t ibegin = static_cast<bst_bin_t>(cut_ptr[fidx]);
    bst_bin_t iend = static_cast<bst_bin_t>(cut_ptr[fidx + 1]);
    bst_bin_t n_bins = iend - ibegin;
    auto n_targets = hist.size();

    auto sum = linalg::Empty<GradientPairPrecise>(ctx_, 2, n_targets);
    auto left_sum = sum.Slice(0, linalg::All());
    auto right_sum = sum.Slice(1, linalg::All());

    // Per-target missing gradient: parent_sum - sum_of_all_bins.
    auto missing_storage = linalg::Empty<GradientPairPrecise>(ctx_, n_targets);
    auto missing = missing_storage.HostView();
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto f_hist = hist[t].subspan(cut_ptr[fidx], n_bins);
      GradientPairPrecise feature_sum{};
      for (bst_bin_t b = 0; b < n_bins; ++b) {
        feature_sum += f_hist[b];
      }
      missing(t) = parent_sum(t) - feature_sum;
    }

    SplitEntryContainer<std::vector<GradientPairPrecise>> best;
    best.is_cat = false;

    for (bst_bin_t i = ibegin; i != iend; ++i) {
      auto split_pt = cut_val[i];

      // Missing on left (missing grouped with other categories).
      for (bst_target_t t = 0; t < n_targets; ++t) {
        right_sum(t) = GradientPairPrecise{hist[t][i]};
        left_sum(t) = parent_sum(t) - right_sum(t);
      }
      auto missing_left_gain =
          evaluator.CalcSplitGain(*param_, nidx, fidx, left_sum, right_sum) - parent_gain;
      best.Update(missing_left_gain, fidx, split_pt, true, true, left_sum, right_sum);

      // Missing on right (missing grouped with chosen category).
      for (bst_target_t t = 0; t < n_targets; ++t) {
        right_sum(t) = GradientPairPrecise{hist[t][i]} + missing(t);  // NOLINT
        left_sum(t) = parent_sum(t) - right_sum(t);
      }
      auto missing_right_gain =
          evaluator.CalcSplitGain(*param_, nidx, fidx, left_sum, right_sum) - parent_gain;
      best.Update(missing_right_gain, fidx, split_pt, false, true, left_sum, right_sum);
    }

    if (best.is_cat) {
      auto n = common::CatBitField::ComputeStorageSize(n_bins + 1);
      best.cat_bits.resize(n, 0);
      common::CatBitField cat_bits{best.cat_bits};
      cat_bits.Set(best.split_value);
    }

    p_best->Update(best);
  }

  template <bst_bin_t d_step>
  void EnumeratePart(common::HistogramCuts const &cut, common::Span<size_t const> sorted_idx,
                     common::Span<common::ConstGHistRow> hist, bst_feature_t fidx, bst_node_t nidx,
                     TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                     SplitEntryContainer<std::vector<GradientPairPrecise>> *p_best) {
    static_assert(d_step == +1 || d_step == -1, "Invalid step.");
    auto n_targets = hist.size();

    auto const &cut_ptr = cut.Ptrs();
    auto const &cut_val = cut.Values();
    auto parent_sum = stats_.Slice(nidx, linalg::All());
    auto parent_gain = gain_[nidx];

    bst_bin_t f_begin = cut_ptr[fidx];
    bst_bin_t f_end = cut_ptr[fidx + 1];
    bst_bin_t n_bins_feature{f_end - f_begin};
    auto n_bins = std::min(param_->max_cat_threshold, n_bins_feature);

    bst_bin_t it_begin, it_end;
    if (d_step > 0) {
      it_begin = f_begin;
      it_end = it_begin + n_bins - 1;
    } else {
      it_begin = f_end - 1;
      it_end = it_begin - n_bins + 1;
    }

    auto sum = linalg::Constant(ctx_, GradientPairPrecise{}, 2, n_targets);
    auto left_sum = sum.Slice(0, linalg::All());
    auto right_sum = sum.Slice(1, linalg::All());

    SplitEntryContainer<std::vector<GradientPairPrecise>> best;
    bst_bin_t best_thresh{-1};
    for (bst_bin_t i = it_begin; i != it_end; i += d_step) {
      auto j = i - f_begin;  // index local to current feature
      for (bst_target_t t = 0; t < n_targets; ++t) {
        auto f_hist = hist[t].subspan(f_begin, n_bins_feature);
        if (d_step == 1) {
          right_sum(t) += f_hist[sorted_idx[j]];
          left_sum(t) = parent_sum(t) - right_sum(t);  // missing on left
        } else {
          left_sum(t) += f_hist[sorted_idx[j]];
          right_sum(t) = parent_sum(t) - left_sum(t);  // missing on right
        }
      }

      auto loss_chg =
          evaluator.CalcSplitGain(*param_, nidx, fidx, left_sum, right_sum) - parent_gain;
      // We don't have a numeric split point, nan here is a dummy split.
      if (best.Update(loss_chg, fidx, std::numeric_limits<float>::quiet_NaN(), d_step == 1, true,
                      left_sum, right_sum)) {
        best_thresh = i;
      }
    }

    if (best_thresh != -1) {
      auto n = common::CatBitField::ComputeStorageSize(n_bins_feature);
      best.cat_bits = decltype(best.cat_bits)(n, 0);
      common::CatBitField cat_bits{best.cat_bits};
      bst_bin_t partition = d_step == 1 ? (best_thresh - it_begin + 1) : (best_thresh - f_begin);
      CHECK_GT(partition, 0);
      std::for_each(sorted_idx.begin(), sorted_idx.begin() + partition, [&](std::size_t c) {
        auto cat = cut_val[c + f_begin];
        cat_bits.Set(cat);
      });
    }

    p_best->Update(best);
  }

 public:
  void EvaluateSplits(common::Span<const BoundedHistCollection *> hist,
                      common::HistogramCuts const &cut,
                      common::Span<FeatureType const> feature_types,
                      std::vector<MultiExpandEntry> *p_entries) {
    auto &entries = *p_entries;
    CHECK_EQ(stats_.Shape(1), hist.size());
    CHECK(!hist.empty());
    std::vector<std::shared_ptr<HostDeviceVector<bst_feature_t>>> features(entries.size());

    for (std::size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      features[nidx_in_set] = column_sampler_->GetFeatureSet(ctx_, entries[nidx_in_set].depth);
    }
    CHECK(!features.empty());

    std::int32_t n_threads = ctx_->Threads();
    std::size_t const grain_size = std::max<std::size_t>(1, features.front()->Size() / n_threads);
    common::BlockedSpace2d space{
        entries.size(), [&](std::size_t nidx_in_set) { return features[nidx_in_set]->Size(); },
        grain_size};

    std::vector<MultiExpandEntry> tloc_candidates(n_threads * entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
      for (std::int32_t j = 0; j < n_threads; ++j) {
        tloc_candidates[i * n_threads + j] = entries[i];
      }
    }

    auto evaluator = tree_evaluator_.GetEvaluator();
    common::ParallelFor2d(
        space, n_threads,
        common::WithWorker([&](std::size_t nidx_in_set, common::Range1d r, common::Worker worker) {
          auto tidx = worker.Id();
          auto entry = &tloc_candidates[n_threads * nidx_in_set + tidx];
          auto best = &entry->split;

          auto parent_sum = stats_.Slice(entry->nid, linalg::All());
          auto base_weight = linalg::Zeros<float>(ctx_, parent_sum.Shape());
          auto h_bw = base_weight.HostView();
          evaluator.CalcWeightCat(*param_, parent_sum, h_bw);

          std::vector<common::ConstGHistRow> node_hist;
          for (auto t_hist : hist) {
            node_hist.emplace_back((*t_hist)[entry->nid]);
          }
          auto features_set = features[nidx_in_set]->ConstHostSpan();
          auto n_targets = hist.size();

          for (auto fidx_in_set = r.begin(); fidx_in_set < r.end(); fidx_in_set++) {
            auto fidx = features_set[fidx_in_set];
            if (!interaction_constraints_.Query(entry->nid, fidx)) {
              continue;
            }
            auto parent_gain = gain_[entry->nid];
            bool is_cat = common::IsCat(feature_types, fidx);
            // Numeric split
            if (!is_cat) {
              bool missing = this->EnumerateSplit<+1>(cut, fidx, node_hist, parent_sum, parent_gain,
                                                      entry->nid, evaluator, best);
              if (missing) {
                this->EnumerateSplit<-1>(cut, fidx, node_hist, parent_sum, parent_gain, entry->nid,
                                         evaluator, best);
              }
              continue;
            }

            auto const &cut_ptr = cut.Ptrs();
            auto n_bins = cut_ptr.at(fidx + 1) - cut_ptr[fidx];
            // One hot split
            if (common::UseOneHot(n_bins, param_->max_cat_to_onehot)) {
              this->EnumerateOneHot(cut, fidx, node_hist, parent_sum, parent_gain, entry->nid,
                                    evaluator, best);
              continue;
            }

            // Partition split
            std::vector<size_t> sorted_idx(n_bins);
            std::iota(sorted_idx.begin(), sorted_idx.end(), 0ul);
            linalg::Vector<GradientPairPrecise> grads({n_targets}, this->ctx_->Device());
            auto h_grads = grads.HostView();
            std::vector<float> child_w(n_targets, .0f);

            // Sort by s_c = w_p^T w_c
            // Project onto the parent's update direction to compare scores l_s < r_s
            // Since we care only about the ordering, no need to divide the norm.
            std::vector<double> scores(n_bins);
            for (std::size_t bin_idx = 0; bin_idx < n_bins; ++bin_idx) {
              for (decltype(n_targets) t_idx = 0; t_idx < n_targets; ++t_idx) {
                auto f_hist = node_hist[t_idx].subspan(cut_ptr[fidx], n_bins);
                h_grads(t_idx) = f_hist[bin_idx];
              }
              evaluator.CalcWeightCat(*param_, h_grads,
                                      linalg::MakeVec(child_w.data(), child_w.size()));
              double sc = .0;
              for (decltype(n_targets) t_idx = 0; t_idx < n_targets; ++t_idx) {
                sc += h_bw(t_idx) * child_w[t_idx];
              }
              // A scalar parent can have zero weight even when its children differ.
              scores[bin_idx] = n_targets == 1 ? child_w[0] : sc;
            }

            std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                             [&](std::size_t l, std::size_t r) { return scores[l] < scores[r]; });

            this->EnumeratePart<+1>(cut, sorted_idx, node_hist, fidx, entry->nid, evaluator, best);
            this->EnumeratePart<-1>(cut, sorted_idx, node_hist, fidx, entry->nid, evaluator, best);
          }
        }));

    for (std::size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      for (auto tidx = 0; tidx < n_threads; ++tidx) {
        entries[nidx_in_set].split.Update(tloc_candidates[n_threads * nidx_in_set + tidx].split);
      }
    }
  }

  linalg::Vector<float> InitRoot(linalg::VectorView<GradientPairPrecise const> root_sum) {
    auto n_targets = root_sum.Size();
    stats_ = linalg::Constant(ctx_, GradientPairPrecise{}, 1, n_targets);
    gain_.resize(1);

    linalg::Vector<float> weight({n_targets}, ctx_->Device());
    auto evaluator = tree_evaluator_.GetEvaluator();
    evaluator.CalcWeight(RegTree::kRoot, *param_, root_sum, weight.HostView());
    auto root_gain = evaluator.CalcGainGivenWeight(*param_, root_sum, weight.HostView());
    gain_.front() = root_gain;

    auto h_stats = stats_.HostView();
    std::copy(linalg::cbegin(root_sum), linalg::cend(root_sum), linalg::begin(h_stats));

    return weight;
  }

  void ApplyTreeSplit(MultiExpandEntry const &candidate, RegTree *p_tree) {
    // Use the split gradient's number of targets for intermediate weights
    // This may differ from p_tree->NumTargets() when using reduced gradient
    auto n_split_targets = candidate.split.left_sum.size();
    auto parent_sum = stats_.Slice(candidate.nid, linalg::All());

    auto evaluator = tree_evaluator_.GetEvaluator();
    auto weight = linalg::Empty<float>(ctx_, 3, n_split_targets);
    auto base_weight = weight.Slice(0, linalg::All());
    evaluator.CalcWeight(candidate.nid, *param_, parent_sum, base_weight);

    auto left_weight = weight.Slice(1, linalg::All());
    auto left_sum =
        linalg::MakeVec(candidate.split.left_sum.data(), candidate.split.left_sum.size());

    auto right_weight = weight.Slice(2, linalg::All());
    auto right_sum =
        linalg::MakeVec(candidate.split.right_sum.data(), candidate.split.right_sum.size());
    evaluator.CalcSplitWeights(*param_, candidate.nid, candidate.split.SplitIndex(), left_sum,
                               right_sum, left_weight, right_weight);

    // Compute the loss_chg and sum hessians for parent and children
    float loss_chg = candidate.split.loss_chg;
    // Sum hessians across all targets for each child
    double left_sum_hess = 0.0, right_sum_hess = 0.0;
    for (std::size_t t = 0; t < candidate.split.left_sum.size(); ++t) {
      left_sum_hess += candidate.split.left_sum[t].GetHess();
      right_sum_hess += candidate.split.right_sum[t].GetHess();
    }

    common::Span<CatWordT const> cat_bits;
    if (candidate.split.is_cat) {
      cat_bits = candidate.split.cat_bits;
    }
    auto as_span = [](linalg::VectorView<float const> weight) {
      // sliced weight might have larger span as the underlying data
      return weight.Values().subspan(0, weight.Size());
    };
    p_tree->Expand(ctx_, ExpandBatch{{SplitInfo{candidate.nid, candidate.split.SplitIndex(),
                                                candidate.split.split_value,
                                                candidate.split.DefaultLeft(), cat_bits},
                                      {as_span(base_weight), left_sum_hess + right_sum_hess},
                                      {as_span(left_weight), left_sum_hess},
                                      {as_span(right_weight), right_sum_hess},
                                      loss_chg}});

    auto left_child = p_tree->LeftChild(candidate.nid);
    CHECK_GT(left_child, candidate.nid);
    auto right_child = p_tree->RightChild(candidate.nid);
    CHECK_GT(right_child, candidate.nid);

    tree_evaluator_.AddSplit(candidate.nid, left_child, right_child, candidate.split.SplitIndex(),
                             left_weight, right_weight);
    evaluator = tree_evaluator_.GetEvaluator();
    interaction_constraints_.Split(candidate.nid, candidate.split.SplitIndex(), left_child,
                                   right_child);

    std::size_t n_nodes = p_tree->Size();
    gain_.resize(n_nodes);
    gain_[left_child] = evaluator.CalcGainGivenWeight(*param_, left_sum, left_weight);
    gain_[right_child] = evaluator.CalcGainGivenWeight(*param_, right_sum, right_weight);

    if (n_nodes >= stats_.Shape(0)) {
      stats_.Reshape(n_nodes * 2, stats_.Shape(1));
    }
    CHECK_EQ(stats_.Shape(1), n_split_targets);
    auto left_sum_stat = stats_.Slice(left_child, linalg::All());
    std::copy(candidate.split.left_sum.cbegin(), candidate.split.left_sum.cend(),
              linalg::begin(left_sum_stat));
    auto right_sum_stat = stats_.Slice(right_child, linalg::All());
    std::copy(candidate.split.right_sum.cbegin(), candidate.split.right_sum.cend(),
              linalg::begin(right_sum_stat));
  }

  explicit HistMultiEvaluator(Context const *ctx, MetaInfo const &info, TrainParam const *param,
                              bst_target_t n_targets,
                              std::shared_ptr<common::ColumnSampler> sampler)
      : param_{param},
        tree_evaluator_{*param, static_cast<bst_feature_t>(info.num_col_), DeviceOrd::CPU(),
                        n_targets},
        column_sampler_{std::move(sampler)},
        ctx_{ctx} {
    interaction_constraints_.Configure(*param, info.num_col_);
    column_sampler_->Init(ctx, info.num_col_, info.feature_weights, param_->colsample_bynode,
                          param_->colsample_bylevel, param_->colsample_bytree);
  }

  [[nodiscard]] auto Evaluator() const { return tree_evaluator_.GetEvaluator(); }
};

}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
