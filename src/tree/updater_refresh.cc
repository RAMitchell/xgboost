/*!
 * Copyright 2014 by Contributors
 * \file updater_refresh.cc
 * \brief refresh the statistics and leaf value on the tree on the dataset
 * \author Tianqi Chen
 */
#include <rabit/rabit.h>
#include <xgboost/tree_updater.h>

#include <vector>
#include <limits>

#include "./param.h"
#include "../common/io.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_refresh);

enum RefreshType {
  kDontUpdateLeaf = 0,  // Only update tree summary statistics
  kRefreshLeaf = 1,     // Replace leaf values using new statistics
  kUpdateLeaf = 2       // Perform Newton update on leaf using new statistics
};

/*! \brief training parameters for regression tree */
struct RefreshParam : public dmlc::Parameter<RefreshParam> {
  // whether refresh updater needs to update the leaf values
  int refresh_leaf;
  DMLC_DECLARE_PARAMETER(RefreshParam) {
    DMLC_DECLARE_FIELD(refresh_leaf)
        .set_default(kRefreshLeaf)
        .set_range(0, 3)
        .describe(
            "0 to update summary statistics only, 1 to replace leaf weights, 2 "
            "to perform Newton update on leaf weights");
  };
};

DMLC_REGISTER_PARAMETER(RefreshParam);

/*! \brief pruner that prunes a tree after growing finishs */
template<typename TStats>
class TreeRefresher: public TreeUpdater {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    tparam_.InitAllowUnknown(args);
    refresh_param_.InitAllowUnknown(args);
  }
  // update the tree, do pruning
  void Update(HostDeviceVector<GradientPair> *gpair,
              DMatrix *p_fmat,
              const std::vector<RegTree*> &trees) override {
    if (trees.size() == 0) return;
    const std::vector<GradientPair> &gpair_h = gpair->ConstHostVector();
    // number of threads
    // thread temporal space
    std::vector<std::vector<TStats> > stemp;
    std::vector<RegTree::FVec> fvec_temp;
    // setup temp space for each thread
    const int nthread = omp_get_max_threads();
    fvec_temp.resize(nthread, RegTree::FVec());
    stemp.resize(nthread, std::vector<TStats>());
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int num_nodes = 0;
      for (auto tree : trees) {
        num_nodes += tree->param.num_nodes;
      }
      stemp[tid].resize(num_nodes, TStats(tparam_));
      std::fill(stemp[tid].begin(), stemp[tid].end(), TStats(tparam_));
      fvec_temp[tid].Init(trees[0]->param.num_feature);
    }
    // if it is C++11, use lazy evaluation for Allreduce,
    // to gain speedup in recovery
#if __cplusplus >= 201103L
    auto lazy_get_stats = [&]()
#endif
    {
      const MetaInfo &info = p_fmat->Info();
      // start accumulating statistics
      for (const auto &batch : p_fmat->GetRowBatches()) {
        CHECK_LT(batch.Size(), std::numeric_limits<unsigned>::max());
        const auto nbatch = static_cast<bst_omp_uint>(batch.Size());
        #pragma omp parallel for schedule(static)
        for (bst_omp_uint i = 0; i < nbatch; ++i) {
          SparsePage::Inst inst = batch[i];
          const int tid = omp_get_thread_num();
          const auto ridx = static_cast<bst_uint>(batch.base_rowid + i);
          RegTree::FVec &feats = fvec_temp[tid];
          feats.Fill(inst);
          int offset = 0;
          for (auto tree : trees) {
            AddStats(*tree, feats, gpair_h, info, ridx,
                     dmlc::BeginPtr(stemp[tid]) + offset);
            offset += tree->param.num_nodes;
          }
          feats.Drop(inst);
        }
      }
      // aggregate the statistics
      auto num_nodes = static_cast<int>(stemp[0].size());
      #pragma omp parallel for schedule(static)
      for (int nid = 0; nid < num_nodes; ++nid) {
        for (int tid = 1; tid < nthread; ++tid) {
          stemp[0][nid].Add(stemp[tid][nid]);
        }
      }
    };
#if __cplusplus >= 201103L
    reducer_.Allreduce(dmlc::BeginPtr(stemp[0]), stemp[0].size(), lazy_get_stats);
#else
    reducer_.Allreduce(dmlc::BeginPtr(stemp[0]), stemp[0].size());
#endif
    // rescale learning rate according to size of trees
    float lr = tparam_.learning_rate;
    tparam_.learning_rate = lr / trees.size();
    int offset = 0;
    for (auto tree : trees) {
      for (int rid = 0; rid < tree->param.num_roots; ++rid) {
        this->Refresh(dmlc::BeginPtr(stemp[0]) + offset, rid, tree);
      }
      offset += tree->param.num_nodes;
    }
    // set learning rate back
    tparam_.learning_rate = lr;
  }

 private:
  inline static void AddStats(const RegTree &tree,
                              const RegTree::FVec &feat,
                              const std::vector<GradientPair> &gpair,
                              const MetaInfo &info,
                              const bst_uint ridx,
                              TStats *gstats) {
    // start from groups that belongs to current data
    auto pid = static_cast<int>(info.GetRoot(ridx));
    gstats[pid].Add(gpair, info, ridx);
    // tranverse tree
    while (!tree[pid].IsLeaf()) {
      unsigned split_index = tree[pid].SplitIndex();
      pid = tree.GetNext(pid, feat.Fvalue(split_index), feat.IsMissing(split_index));
      gstats[pid].Add(gpair, info, ridx);
    }
  }
  inline void Refresh(const TStats *gstats,
                      int nid, RegTree *p_tree) {
    RegTree &tree = *p_tree;
    tree.Stat(nid).base_weight = static_cast<bst_float>(gstats[nid].CalcWeight(tparam_));
    tree.Stat(nid).sum_hess = static_cast<bst_float>(gstats[nid].sum_hess);
    if (tree[nid].IsLeaf()) {
      if (refresh_param_.refresh_leaf == 1) {
        tree[nid].SetLeaf(tree.Stat(nid).base_weight * tparam_.learning_rate);
      }
      else if (refresh_param_.refresh_leaf == 2) {
        tree[nid].SetLeaf(tree[nid].LeafValue() + tree.Stat(nid).base_weight * tparam_.learning_rate);
      }
    } else {
      tree.Stat(nid).loss_chg = static_cast<bst_float>(
          gstats[tree[nid].LeftChild()].CalcGain(tparam_) +
          gstats[tree[nid].RightChild()].CalcGain(tparam_) -
          gstats[nid].CalcGain(tparam_));
      this->Refresh(gstats, tree[nid].LeftChild(), p_tree);
      this->Refresh(gstats, tree[nid].RightChild(), p_tree);
    }
  }
  // training parameter
  TrainParam tparam_;
  RefreshParam refresh_param_;
  // reducer
  rabit::Reducer<TStats, TStats::Reduce> reducer_;
};

XGBOOST_REGISTER_TREE_UPDATER(TreeRefresher, "refresh")
.describe("Refresher that refreshes the weight and statistics according to data.")
.set_body([]() {
    return new TreeRefresher<GradStats>();
  });
}  // namespace tree
}  // namespace xgboost
