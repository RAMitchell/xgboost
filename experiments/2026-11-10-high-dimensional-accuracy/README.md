# Current rule correction (2026-11-11)

The user requires standard Gauss–Legendre throughout the paper. The current C++
CPU/GPU rule is now the standard linear mapping to [0,1]. New collector outputs
are labeled cpu_standard4/6/8/16_float32 and use standard8_float64 as the independent
implementation comparison. The old sweep-rounds30-seeds3/ files and sources remain
an archive of the transformed rule; do not relabel or use its production curves
as standard-rule results. Use a fresh output directory for a corrected sweep;
manifest/source hashes prevent resuming the old directory with changed code.
Prepare the corrected CPU runtime with `python prepare_runtime.py`; it defaults
to `build-cpu-test/lib/libxgboost.so` and writes `python-runtime-standard/`.
Use that directory in PYTHONPATH and a new collector output directory.
The historical protocol below describes the archived experiment.

# Eight-point accuracy beyond the exactness regime

Scope: colleague's item 2. Collect data first for a possible combined normalized-error
plot resembling the earlier Fashion-MNIST depth sweep. No plot has been chosen or
constructed yet. Item 3 (matched-count/precision TreeGrad runtime comparison) remains
untouched.

## Questions and controls

Sweep Adult, CalHousing, Covtype and Fashion-MNIST through requested depths
4, 8, 12, 16, 24, 32, 40 and 50. Record realized raw depth and distinct-feature depth,
including the leaf fraction and cover-probability mass beyond 16 distinct features.
Adult (14 features) and CalHousing (8) cannot leave the standard-eight-point exactness
regime; retain them as controls, not evidence about d > 16. Covtype and Fashion-MNIST
must actually attain d > 16 before their deeper requested-depth cases count as such
evidence. A requested depth of 50 is not a guarantee of realized depth 50.

Models use lossguide growth, a 512-leaf cap, learning rate .05, 30 boosting rounds,
CPU hist, explicit base_score=.5, and seeds 432/433/434. Each seed produces a fixed
80/20 train/held-out split (stratified for classification), with 256 distinct held-out
rows selected once and reused across depths. These are newly trained accuracy-study
models, not replacements for the paper's cached runtime models. Covtype retains the
paper loader's labels 1..7 / eight output groups. The pilot uses three rounds,
32 evaluation rows, eight reference rows, one seed, and depths 16/50.

## An important quadrature distinction

The current C++ CPU/GPU code calls `MakeEndpointQuadrature`, which transforms a
Gauss–Legendre rule by t=s² and weights by 2s. This differs from the standard
Gauss–Legendre rule described in the paper. For a degree-(d-1) first-order polynomial,
a standard eight-point rule is exact when d <= 16, whereas the transformed rule has
that polynomial guarantee only when d <= 8. This does not establish the observed
size of either approximation error: measure both.

Compare these explicitly labeled methods:

- standard eight-point Gauss–Legendre, independent float64 leaf-path evaluation;
- transformed endpoint eight-point rule, the same float64 evaluator;
- existing C++ CPU endpoint rules with 4, 6, 8 and 16 float32 points;
- existing C++ CPU TreeSHAP as a comparator, not an assumed reliable reference.

The reference is standard Gauss–Legendre with max(16, ceil(realized d / 2)) points,
which integrates the attribution polynomial exactly in real arithmetic. Repeat at
twice as many points to check float64 convergence. It multiplies probability factors
rather than constructing polynomial coefficients or using the production recursive
return-update implementation. It aggregates repeated-feature branch probabilities
and tests all feature constraints, including categorical splits and missing values.
It computes every feature attribution, rather than checking only attribution sums.

`validate_reference.py` checks against exhaustive coalition enumeration on small
regression, binary, and multiclass models with repeated features, categorical splits
and missing values. It also checks known-answer symmetric conjunctions with 16/32/50
distinct features. No claim of universal eight-point accuracy is assumed.

## Build and invocation

The existing CUDA binary configures its GPU predictor even for CPU predictions when
a GPU is visible, rejecting point counts other than eight. Hiding GPUs also exposed
a CUDA allocation dependency in multiclass metric construction. Initial failed pilot
logs are retained; those runs contain no completed accuracy data. Use an isolated
CPU-only Release build, preserving the existing CUDA library:

```bash
cmake -S . -B build-accuracy-cpu -DUSE_CUDA=OFF -DGOOGLE_TEST=OFF \
  -DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/home/nfs/rorym/anaconda3/envs/xgboost/bin/x86_64-conda-linux-gnu-c++
cmake --build build-accuracy-cpu --target xgboost -j35
/home/nfs/rorym/anaconda3/envs/xgboost/bin/python \
  experiments/2026-11-10-high-dimensional-accuracy/prepare_runtime.py
```

A local `python-runtime/xgboost` package links to the worktree Python sources and
loads the isolated build through its own `lib/libxgboost.so`. The manifest records
the actual loaded library path and SHA256; the collector also snapshots its sources.

```bash
env PYTHONUNBUFFERED=1 \
  PYTHONPATH=/home/nfs/rorym/xgboost-wt/shapley-value-algorithms/experiments/2026-11-10-high-dimensional-accuracy/python-runtime \
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=16 \
  taskset -c 20-35 /home/nfs/rorym/anaconda3/envs/xgboost/bin/python -u \
  experiments/2026-11-10-high-dimensional-accuracy/collect_accuracy.py \
  --out-dir experiments/2026-11-10-high-dimensional-accuracy/sweep-rounds30-seeds3 \
  --rounds 30 --seeds 432 433 434
```

Only one collector should run against an output directory. It checks configuration,
source and library identity on resume, skips completed model cases, and reloads saved
models for incomplete cases. `complete.json` marks all requested cases finished.
Builds, runtime links, cached models and outputs are local experiment artifacts.

## Inspectable measurements

Each case has a saved model, metrics JSON, and compressed raw arrays containing
train/evaluation row indices, evaluation inputs (categorical values are category
codes), output margins, reference attributions at both point counts, and all method
outputs. Reference rules explain the first 16 of the 256 held-out rows; full C++
outputs and efficiency errors cover all 256. JSON records these sample counts.

Absolute errors: mean/p99/max attribution error, bias error, efficiency residual
(sum of all contributions minus output margin), and nonfinite counts.

Candidate normalizations retained for later plotting:

- attribution NRMSE = RMS attribution error / RMS reference attribution;
- normalized MAE = mean absolute attribution error / mean absolute reference attribution;
- maximum attribution error / global maximum absolute reference attribution;
- row/output relative L2 error, excluding only norms <= 1e-12 of the largest norm
  (also report the fraction retained);
- maximum efficiency error / RMS(output margin minus reference bias), using the
  same prediction scale across all methods for a case.

Attribution comparisons exclude the bias column, whose error is recorded separately.
All reductions use float64. Zero denominators yield null, not an invented epsilon
normalization. Nonfinite errors are counted explicitly; finite-only distribution
summaries do not certify accuracy when any nonfinite output exists. Errors refer to
output margins (logits for classifiers), not probabilities. Across-seed variation
should remain visible rather than collapsing it into execution-repeat variability.

Run `validate_collection.py <output-directory>` with the xgboost conda Python to
recompute the saved norms, check the raw samples, model hashes, disjoint evaluation
rows, reference convergence and exactness-regime controls.


## Validated pilot and current collection

`pilot-isolated-cpu/` completed eight cases (all four datasets, depths 16/50,
three rounds, one seed, 32 evaluation rows, eight reference rows). All raw-array
metrics and exactness controls passed independent validation. At requested depth
50, realized maximum raw/distinct depths were Adult 23/12, CalHousing 17/8,
Covtype 26/17 and Fashion-MNIST 40/37. Fractions of leaves beyond distinct depth
16 were about 0.04% for Covtype and 8.20% for Fashion-MNIST. Maxima alone should
not be presented as representative of all paths.

Production eight-point attribution NRMSE against the independent reference was
1.33e-7 for Covtype and 1.76e-7 for Fashion-MNIST at requested depth 50. These
are pilot estimates on small ensembles and few rows, not final accuracy claims.
The full `sweep-rounds30-seeds3/` collection targets 96 model cases. Its live log
is `collection.log`; `collection.pid` records the process. Per-case checkpoints
are durable, and complete.json marks the full requested grid.

The full collector additionally records production-eight-point versus float64
endpoint-eight-point errors directly, separating implementation roundoff from the
endpoint-rule approximation. Raw arrays also preserve both for the pilot.


## Final collection status

All96 main-sweep model cases completed and independently validated. See
`final-validation.log`. The collection process exited and the completion marker
is present. `sweep-rounds30-seeds3/accuracy-summary.csv` provides672 method/model
rows for plotting, alongside the complete raw arrays and metrics.

Worst production-eight-point attribution NRMSE across the collection is2.014e-7.
Maximum realized distinct-feature depths are13 (Adult),8 (CalHousing),20 (Covtype)
and42 (Fashion-MNIST). These are empirical results on30-round ensembles and16
reference rows per model; the rare beyond16 paths on Covtype remain a coverage
limitation. No combined plot has been produced yet.
