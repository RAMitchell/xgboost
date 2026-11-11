# Third-order runtime repetitions

Purpose: obtain mean and sample standard deviation for the Table 4 Python
third-order SII comparison, which previously had only one timing per model.

Current run: `full-10-idfix/`, coordinator `full-idfix.log` / `full-idfix.pid`. The five published
numeric models plus previously excluded CalHousing-large are attempted. Adult
categorical models and Fashion-MNIST are outside the original Table 4 scope.
Pilot: `pilot/`, CalHousing-small, three repetitions after one warm-up; both methods
passed independent timing/output validation (max disagreement 4.94e-17).

## Protocol

- Fixed cached models, same single explained row selected with seed432.
- Both algorithms use float64 and ordinary Gauss–Legendre on [0,1]; quadrature
  uses eight points. Python algorithm source is the original benchmark_order3.py.
- Ten timed repetitions after three warm-up predictions per algorithm/model.
  Construction, JIT, I/O and warm-ups are excluded. All measured samples retained.
- One CPU core20, one OpenMP/BLAS/Numba thread, no GPU execution. Algorithm order
  is shuffled deterministically per model; each method uses a separate process.
- Every warm-up/timed explanation has its own600s process watchdog. Any overrun
  invalidates the entire method/model entry, including earlier timing samples.
  No aggregate timing budget. Setup has a separate600s guard, and its failures
  are explicitly distinguished from explanation timeouts.
- Timing scope preserves the old comparison: quadrature returns a dictionary;
  TreeSHAP-IQ explanation plus extraction of order3 dictionary values is timed.
- Both methods skip constant trees. For multiclass models, the comparison sums
  all tree outputs, matching the original Table4 Python target; it does not select
  one class or claim to explain every class separately.

## Canonical model adapter

The old dataframe parser promotes rounded float32 model parameters directly to
float64, and the baseline conversion tests <= while XGBoost and quadrature test <.
For this rerun, restore exact float32 thresholds/covers/leaf values before promoting
to float64; do the same for the single input row. Baseline thresholds use the
previous float64 representable value so its <= test matches XGBoost's < test.
Compute child cover ratios identically. These are model/input-conversion repairs;
the two explanation algorithms are unchanged. Historical timings are preserved,
but this run is not an exact historical execution replay.

`validate_adapters.py` directly enumerates all coalition values and third-order
SII on small trees, testing repeated features and threshold-equality inputs. Both
implementations passed (worst error2.54e-16). `validate_timeout.py` verifies a total
longer than the limit succeeds and a later over-limit call voids the whole entry.
`validate_results.py` recomputes timing moments, checks every saved warm-up/repeat
interaction array and compares the two methods. SD describes within-batch execution
variability, not a confidence interval or variability across trained models.

## Exact command

Run only one instance against a given output directory:

```bash
env PYTHONPATH=/home/nfs/rorym/xgboost-wt/shapley-value-algorithms/experiments/2026-11-10-high-dimensional-accuracy/python-runtime-standard \
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  NUMBA_CACHE_DIR=/home/nfs/rorym/xgboost-wt/shapley-value-algorithms/experiments/2026-11-11-order3-variability/numba-cache \
  taskset -c 20 /home/nfs/rorym/anaconda3/envs/xgboost/bin/python -u \
  experiments/2026-11-11-order3-variability/run_repeated.py \
  --out-dir experiments/2026-11-11-order3-variability/full-10-idfix
```

Code base revision:72d918fec, plus the measured source snapshots under full-10/sources/.
The manifest records source/model/library hashes, dependency versions, affinity,
environment and all protocol settings. Source/configuration changes refuse resume.
Each model has row.npy and one folder per method with raw arrays and results.json.
status.json records the active call; summary.md/results.json update after each
method completes; complete.json exists only when all requested cases finish.


## Feature-ID labeling correction and reuse

The first partial full-10/ run was stopped after independent checks found that
TreeSHAP-IQ enumerates noncontiguous feature IDs through a set while its interaction
lookup sorts them. This can permute labels without changing the underlying values.
The adapter now supplies a set with deterministic ascending iteration, keeping
input remapping, internal numbering and output labels aligned. No library source
or mathematical algorithm is changed. A 54-feature embedding of the small exact
coalition test (active IDs0/12/15/45) now passes; real CovType-small pilot discrepancy
is below9.55e-17. The old CovType baseline outputs are superseded, not paper data.

To avoid repeating unaffected measurements, reuse_unchanged.py certifies unchanged
quadrature code paths by AST comparison, and checks that all CalHousing tree feature
orders were already ascending. Six completed method/model cases are reused into
full-10-idfix/ with explicit original source/manifest references and raw-result
hashes. The independent validator checks their original statistics, all copied
arrays, and matching model/library/environment/protocol identities. Affected
CovType baseline cases are remeasured. The pilot uses separate output directories;
its short timings are never mixed into the ten-repeat summaries.


## Baseline computation scope

The installed shapiq TreeExplainer accepts min_order=3 but constructs per-tree
TreeSHAPIQ objects with their default minimum order1. The timed baseline therefore
computes orders1–3 and extracts order3, matching the published benchmark. Quadrature
computes order3 alone. This run measures variability of that existing comparison;
it is not a matched-output-order ablation. This distinction must be stated when
using the new speedups. The order2–6 scaling figure has not been rerun here.


## Final outcome

The full-10-idfix suite is complete and independently validated:11 successful
method/model entries with110 timing samples, one explicit setup timeout. CalHousing-
large Q is 117.506691 ± 0.175272 s, while TreeSHAP-IQ exceeded
600s during setup and produced no explanation timings. All five paired comparisons
passed, with speedups25.60–43.39× (median29.04×). The active paper includes all six
models and the specific setup failure reason in follow-up commit9d67800. No pending
measurement wording remains. Full-project local paper compilation passed.
