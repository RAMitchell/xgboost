# Matched-count and matched-precision Appendix comparison

Scope: colleague's item 3, revisiting the first-order Python comparison in Appendix
A.1 / Table 5. Reuse the twelve cached paper models and ten explanation rows selected
by the original loader with seed432. Nine models have a supported TreeGrad baseline;
the three Adult models retain an explicit unsupported-categorical status.

## Controls

Both algorithms use float64 inputs, model parameters, state, accumulation, outputs,
and standard Gauss–Legendre nodes on [0,1]. This is the Python appendix comparison,
not the float32 endpoint-transformed C++ path studied in item2.

Point count must be matched per tree, not just by the ensemble maximum:

- Native TreeGrad: n=ceil(min(raw tree depth D, feature count F)/2), per tree.
- Tight exact count: n=ceil(distinct-feature depth d/2), per tree.
- Fixed-count control: eight points per tree, the same generalized arithmetic path.

The tighter bound handles repeated split features. TreeGrad's `max_depth` adapter
metadata is set to d for the tight variant. Inspection of the pinned upstream
`treegrad_shap.py` confirms that field is used solely to choose the quadrature count;
node arrays, traversal, thresholds, covers and values remain unchanged. Count audits
record actual upstream `leggauss` calls outside timing and check every nonconstant
tree against its expected native/tight count. No TreeGrad source files are edited.

Compare six variants:

| Label | Implementation | Count | Node generation |
|---|---|---|---|
| q_fixed8 | Quadrature | fixed8 | cached outside timing |
| q_native_count | Quadrature | TreeGrad native per-tree count | cached outside timing |
| q_distinct_count | Quadrature | tight exact per-tree count | cached outside timing |
| tg_native | TreeGrad | native | upstream recomputes per tree and row |
| tg_distinct | TreeGrad | tight exact | upstream recomputes per tree and row |
| tg_native_cached | TreeGrad | native | canonical rules cached outside timing |

For the cache control, only TreeGrad's node provider is replaced. It returns copies
of cached [-1,1] rules because upstream transforms points and weights in place.
Transformation, traversal and all other upstream work remain timed. This separates
node generation cost from the fixed-point-budget effect; it is not an optimized
rewrite of TreeGrad.

The former appendix harness used Numba arithmetic helpers only at 8 points. Simply
changing its point-count argument would switch implementation paths. The new
`quadrature.py` uses identical dynamic-length Numba helpers at every count, retaining
Python recursive traversal. Therefore the new fixed8 control is measured again;
do not substitute historical fixed8 times or present differences as solely due to
point counts when comparing against the old table.

Unlike the old adapter's single flattened sum across class trees, this comparison
preserves all output groups. This prevents attribution errors cancelling across
classes and computes the conventional class-specific first-order outputs. Constant
root-only trees contribute zero feature attribution and are skipped by both methods.
Bias/intercept values are excluded by both methods. Numeric inputs/model parameters
are first restored to their float32 XGBoost representations, then promoted to
float64 for both algorithms. Thresholds are nudged in the TreeGrad adapter to match
XGBoost's strict-less-than decisions. Upstream TreeGrad does not implement XGBoost
missing-value routing; a numeric case containing missing evaluation values fails
explicitly rather than silently timing different outputs.

## Protocol and provenance

One physical CPU core 20, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
NUMBA_NUM_THREADS=1, isolated CPU-only XGBoost Release library from item2. Cached
models are not retrained. Each method warms one full-tree row, then explains the
same ten rows in three timed repetitions. Method order is deterministically shuffled
within each repetition. Store every row time, batch total, mean, and sample SD; SD
uses the three complete ten-row batches, not 30 distinct input observations.

Timer scope is the row explanation call over all output groups. Model loading,
shared conversion, construction, quadrature cache setup, count audits, Numba
compilation, warm-up and result serialization are excluded. Native TreeGrad's own
within-call node computation is included. Node-cache setup is outside timing for
both cached variants. Warm-up and construction costs are recorded separately.

Native TreeGrad can take many minutes per ten-row batch on the largest models.
The six variants × three repeats can take several hours on those models. The
per-model wall timeout is 21600 seconds, including setup/audit/warm-up and all
trials. Checkpoint every completed trial and retain a live status.json for the
current method/repetition; do not infer completion from a quiet top-level log.

TreeGrad is pinned at commit 8f7fdc6 in the local ignored `dependencies/TreeGrad`
checkout, copied from the previously used checkout. Manifest records the full
commit, upstream source hashes, library/model hashes, script hashes, versions,
affinity, environment and timing protocol. Source snapshots are retained in each
run. Uncommitted source changes from earlier items are preserved.

## Commands

From the worktree, using the isolated runtime prepared in the item2 experiment:

```bash
env PYTHONPATH=/home/nfs/rorym/xgboost-wt/shapley-value-algorithms/experiments/2026-11-10-high-dimensional-accuracy/python-runtime \
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /home/nfs/rorym/anaconda3/envs/xgboost/bin/python \
  experiments/2026-11-10-treegrad-matched/validate_algorithms.py
```

```bash
env PYTHONUNBUFFERED=1 \
  PYTHONPATH=/home/nfs/rorym/xgboost-wt/shapley-value-algorithms/experiments/2026-11-10-high-dimensional-accuracy/python-runtime \
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  taskset -c 20 /home/nfs/rorym/anaconda3/envs/xgboost/bin/python -u \
  experiments/2026-11-10-treegrad-matched/benchmark_matched.py \
  --out-dir experiments/2026-11-10-treegrad-matched/matched-full \
  --rows 10 --niter 3
```

Only one scheduler should run per output directory. It skips
completed models on resume, but reruns incomplete model cases from the start.
Source/configuration identity must match the manifest. Partial timing arrays and
outputs remain inspectable even for failed/timed-out cases. `finished.json` reports
attempt counts and failures; unsupported categories are explicit completed outcomes,
not successful matched comparisons. A failure produces a nonzero scheduler exit.

Validate finished models and regenerate tables at any point:

```bash
/home/nfs/rorym/anaconda3/envs/xgboost/bin/python \
  experiments/2026-11-10-treegrad-matched/validate_results.py \
  experiments/2026-11-10-treegrad-matched/matched-full
/home/nfs/rorym/anaconda3/envs/xgboost/bin/python \
  experiments/2026-11-10-treegrad-matched/summarize.py \
  experiments/2026-11-10-treegrad-matched/matched-full
```

`validate_algorithms.py` compares both point policies, generic helper counts and
cached/native TreeGrad to an independent float64 reference on small regression
and multiclass trees with repeated features, including threshold equality. It
also checks that cached node arrays cannot be mutated by a caller. All passed.
`validate_results.py` independently recomputes timing moments, speedups and output
agreement, checks dtype/counts, and confirms repeated outputs are identical.

A first pilot exposed a JSON serialization issue in metadata before timing began;
those failed logs remain under `pilot/`. The initial corrected pilot is under
`pilot-v2/`; its cached-node proxy was then found to add avoidable NumPy attribute
lookup overhead. That control was corrected by exposing NumPy attributes directly,
and all methods were rerun under `pilot-v3/`. The early `full/` launch was stopped
before continuing long measurements; keep it only as superseded diagnostic history.
The active final run is `matched-full/`, with `matched-full.log` / `matched-full.pid`.
Completed `pilot-v3` cases are reused only after source/library/environment/protocol
identity checks, recorded in `matched-full/reused-cases.json`. Generated models, arrays, dependency checkout, logs and tables remain
local under the existing experiment exclusion.


## Final pilot and active long run

`pilot-v3` completed four supported models (CalHousing-small, Covtype-small,
Covtype-sparse and Fashion-MNIST-small), plus explicit Adult-small unsupported.
All24 method/model cases and72 full-batch timing samples passed independent
validation; repeated output arrays match exactly. Matched native/tight speedups
range2.56–2.80×, and native-matched comparisons with cached TreeGrad nodes range
2.33–2.57×. These are pilot results; remaining large cases are pending.

Validated pilot cases are reused in `matched-full`, with source-manifest hash and
full measured-code/library/environment/protocol identity checks. Live process
state: `matched-full.log`, `matched-full.pid`, and per-case `status.json`. Complete
models have `complete.json`; each trial updates results.json and output arrays.
`table5-matched-native.md` and `table5-matched-distinct.md` contain timings with
sample SD, while summary.md/CSV retain the implementation controls.
