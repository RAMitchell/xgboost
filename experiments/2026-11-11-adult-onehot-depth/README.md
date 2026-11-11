# Adult one-hot depth-50 pilot

Purpose: user requested expanding Adult categorical features and checking distinct-
feature depth at requested50 before selecting an accuracy-study representation.

Data: pinned OpenML179, the version loaded by the current benchmark. It has48842
rows,12 categorical columns (including four binned numeric fields) and2numeric.
For each seed432/433/434, use the existing study's stratified80/20split. Fit
OneHotEncoder only on training rows, retaining every learned category (drop=None),
handle_unknown=ignore, dense float32 output with explicit zeros. Numerical columns
are retained. All three splits yield124columns:122indicators+2numeric.
Each case saves encoder.joblib, encoding.json, encoded data and split indices.

Training:30rounds, binary:logistic, CPU hist, lossguide, max_depth50, max_leaves512,
eta.05, base_score.5,16threads/cores20–35, colsample_bylevel1 (all features).
Uses the corrected standard-rule CPU library; only training/structure metrics are
collected here, without SHAP accuracy or runtime measurements. Native benchmark
models and paper tables are unchanged.

Results (seed: raw depth / distinct encoded-feature depth / cover mass beyond16):
432:33/25/10.716%;433:35/27/9.285%;434:37/26/11.893%.
Uniform leaf fractions beyond16 are16.608%,13.574%,16.771%, respectively. Native
categorical Adult models in the same study have maximum distinct depth13 for all
three seeds. Expansion therefore provides substantial additional d>16 coverage.
No per-level feature-sampling adjustment was used.

Validation: independent JSON traversal checks raw depth, maximum distinct depth
and the full leaf-depth distribution against the existing reference flattener.
Deepest-path witnesses, source/library/model hashes and dataframe fingerprint are
retained. Verified all models contain only numerical splits. complete.json marks
all three cases finished; summary.csv contains compact metrics.

Exact invocation (from the algorithms worktree):

```bash
env PYTHONPATH=/home/nfs/rorym/xgboost-wt/shapley-value-algorithms/experiments/2026-11-10-high-dimensional-accuracy/python-runtime-standard   CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1   taskset -c20-35 /home/nfs/rorym/anaconda3/envs/xgboost/bin/python -u   experiments/2026-11-11-adult-onehot-depth/probe_adult_onehot.py   --out-dir experiments/2026-11-11-adult-onehot-depth/depth50-seeds3
```

Next: use these encoded-column models for a standard-rule normalized accuracy
sweep, labeled Adult (one-hot). An encoded indicator is a feature player in that
study; do not describe124columns as124 independent original variables. No accuracy
claim follows from the structural depth probe alone.
