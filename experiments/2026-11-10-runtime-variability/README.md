# Current implementation and locked timeout policy (2026-11-11)

`run_suite.py` now uses standard Gauss–Legendre and a 600-second limit per
prediction, without an aggregate case cutoff. Always use a new output directory
for this protocol, e.g. `standard-gl-suite-10/`; old manifests intentionally reject
changed code/library/configuration. This update does not relabel archived results.
The shared benchmark now implements the policy directly; the older per-prediction
recheck helpers below are retained as provenance for already completed experiments.

`validate_benchmark_prediction_timeout.py` checks both first-order and pairwise
accuracy/warm-up/timed phases, permits long repeated totals, invalidates all samples
on a later timeout, and distinguishes optional aggregate deadlines.

# Runtime variability for paper review

Purpose: expose execution variability for both paper runtime tables and check the
marginal CalHousing-small first-order CPU result with more repetitions.

Source revision: `7ee42815b` on `shapley-value-algorithms`, with the benchmark
changes in this working diff. The new run uses the existing compiled XGBoost
library (3.3.0-dev); its build revision is not independently established.
Environment: Xeon E5-2698 v4, Tesla V100-SXM2-32GB, GPU 1 selected as logical
CUDA device 0, CPU affinity 20–35, 16 OpenMP threads, one OpenBLAS thread.
The conda environment is `xgboost`; Python imports are pinned to this worktree.
Other host activity can still cause timing jitter despite CPU affinity and an idle GPU.

## Existing paper measurements

The original tables already used three timed repetitions and retained population
standard deviations in JSON. The paper-facing tables omitted them. Regenerate
the complete 12-model first-order and pairwise tables with:

```bash
python experiments/2026-04-10-rapids-style-shap-benchmark/summarize_runtime_variability.py \
  --inputs \
    experiments/2026-04-10-rapids-style-shap-benchmark/results-all-rows1000-niter3-additive-v3.json \
    experiments/2026-04-10-rapids-style-shap-benchmark/results-all-rows100-niter3-interactions-v4.json \
  --out-dir experiments/2026-11-10-runtime-variability/historical
```

The historical directory contains Markdown and LaTeX tables with mean ± sample SD,
plus CSV/JSON with per-case sample counts, means, sample SD, coefficient of variation,
failure descriptions, and source paths. Conversion is s = population_SD * sqrt(n/(n-1)).
No means, speedups, or failure states are changed. These are historical measurements,
not freshly rerun results. Missing raw samples prevent reconstruction of historical
medians, timing distributions, or paired comparisons.

## New targeted run

```bash
env PYTHONPATH=/home/nfs/rorym/xgboost-wt/shapley-value-algorithms/python-package \
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 \
  taskset -c 20-35 /home/nfs/rorym/anaconda3/envs/xgboost/bin/python \
  experiments/2026-04-10-rapids-style-shap-benchmark/quadratureshap_rapids_benchmark.py \
  --model cal_housing-small \
  --model-dir experiments/2026-04-10-rapids-style-shap-benchmark/model-cache \
  --nrows 1000 --niter 30 --warmup 5 --nthread 16 --case-timeout-seconds 180 \
  --output experiments/2026-11-10-runtime-variability/cal-housing-small-30.json \
  --out-results experiments/2026-11-10-runtime-variability/cal-housing-small-30.csv \
  --out-details experiments/2026-11-10-runtime-variability/cal-housing-small-30-details.csv \
  --out-markdown experiments/2026-11-10-runtime-variability/cal-housing-small-30.md
```

All four cases completed with 30 samples each. The independent validator recomputes
means, sample SDs, and medians from raw samples and checks the historical conversion
and speedup preservation for all 96 historical cases:

```bash
python experiments/2026-11-10-runtime-variability/validate_variability.py \
  experiments/2026-11-10-runtime-variability/cal-housing-small-30.json
```

Interpretation: historical CalHousing-small CPU Q-TreeSHAP has mean 2.955 ms and
sample SD 1.272 ms, versus TreeSHAP 3.118 ± 0.0315 ms (n=3). The 1.055x point
estimate alone is insufficient evidence of a repeatable small advantage.
The new run is also noisy: TreeSHAP 14.671 ± 6.388 ms and Q-TreeSHAP
12.246 ± 8.639 ms (n=30). Its ratio of means is 1.198x. Do not infer statistical
significance, or replace the historical 1.055x with this ratio: the new thread/affinity
configuration is explicit, while historical configuration metadata is incomplete.
Retain all samples, including outliers; median and quartiles are in the run JSON.

Recommended paper change: report mean ± sample SD and n=3 for the original tables,
describe the repeats as execution repeats on fixed models/rows, and soften any claim
that every CPU workload is demonstrably faster. A fresh complete suite with more
repetitions remains a follow-up under a quiet, fixed hardware configuration; this
experiment does not claim that all 12 models were freshly rerun.


## Pairwise check and validation

Also completed the same CalHousing-small run with `--nrows 100 --interactions`,
retaining `--niter 30 --warmup 5 --nthread 16` and the same CPU/GPU placement.
Outputs use the prefix `cal-housing-small-interactions-30`; all four cases completed.
Run the independent validator with both `cal-housing-small-30.json` and
`cal-housing-small-interactions-30.json` to check all 240 new timing observations.

`validate_timing_worker.py` exercises a 2,000-sample result larger than a pipe
buffer, the one-repetition path (SD unavailable), and timeout handling. All passed.
The timeout worker now drains results before joining the process to avoid blocking
when many raw samples are retained. The three changed/added benchmark/validation
scripts pass syntax checks; git diff passes whitespace checks.


## Full-suite rerun (started 2026-11-10)

Scope agreed in chat: colleague's item 1, repeated runtime experiments and variability.
Items 2 (additional high-dimensional accuracy evidence) and 3 (matched quadrature
count and precision for Appendix A.1/Table 5) are pending and have not been started
by this run.

The CUDA Release target was checked with
`cmake --build build-gpu-bench-release --target xgboost -j35` before launch.
The build completed successfully. The run pins Python imports to this worktree;
`full-suite-10/manifest.json` records the source revision, loaded-library path and
SHA256, benchmark SHA256, hashes of all 12 cached models, affinity and environment.
The source diff at launch is retained alongside it. Models are reused without
retraining. The new run is separate from all historical measurements above.

Reproduce or resume from this worktree (run only one instance at a time):

```bash
env PYTHONUNBUFFERED=1 \
  PYTHONPATH=/home/nfs/rorym/xgboost-wt/shapley-value-algorithms/python-package \
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 \
  taskset -c 20-35 /home/nfs/rorym/anaconda3/envs/xgboost/bin/python -u \
  experiments/2026-11-10-runtime-variability/run_suite.py \
  --out-dir experiments/2026-11-10-runtime-variability/full-suite-10
```

Protocol: both algorithms on CPU and GPU; 10 timed repeats after 3 explicit warm-ups
and untimed accuracy calls; 16 threads; seed 432; 1,000 rows first-order and 100 rows
pairwise. The 1,800-second per-case timeout covers setup, accuracy, warm-ups and all
repetitions. This differs from the historical 600-second timeout, so DNF states are
not directly comparable. Raw samples include all observed outliers. Calls are
sequential batches, so sample SD measures within-batch execution variability and
does not establish independent-run uncertainty or a significant speedup.

`runner.log` and `runner.pid` identify the live run. Each model has its own command,
log and JSON checkpoint under `first-order/` or `pairwise/`. The combined JSONs and
`tables/` are updated after each completed model; they are explicitly marked
`model_filter: partial` until all 12 models for that table are present.
`complete.json` is written only after both tables finish. Resume verifies the
manifest identity and reuses finished checkpoints, including explicit failures.
Use a new output directory to change the protocol or retry failed cases.

Validate partial or final measurements with:

```bash
/home/nfs/rorym/anaconda3/envs/xgboost/bin/python \
  experiments/2026-11-10-runtime-variability/validate_variability.py \
  --allow-failures \
  experiments/2026-11-10-runtime-variability/full-suite-10/first-order.json \
  experiments/2026-11-10-runtime-variability/full-suite-10/pairwise.json
```

Only pass JSONs that already exist. Failed cases must retain missing means/SDs and
no timing samples; successful cases are independently checked against every sample.
Generated logs, timing tables and checkpoints remain local under the existing
experiment-output exclusion.


## Threading diagnostic (2026-11-10)

`check_threading.py` compared pinned16, wide16, wide40, wide80, and automatic
threading on CalHousing-small, Adult-small and CalHousing-sparse. All 300 timing
samples passed independent validation. Raw results, exact commands, provenance,
and the full mean ± sample SD comparison are in `threading-check/summary.md`.

The original harness did not set nthread; automatic OpenMP currently exposes 80
threads with no CPU quota. Historical effective settings are unknown. Threading
changes timings materially, but automatic defaults do not recover the paper's
ratios: CalHousing-sparse is 1.30× versus 3.33× historically, and Adult-small is
1.41× versus 2.26×. The diagnostic uses in-process case workers; the new full suite
uses spawned case workers, so process mode remains a separate comparison to check.
Do not label the difference a code regression or pure execution variability.
The full 16-thread suite resumed after the isolated diagnostic; its original
manifest and configuration remain unchanged.


## Corrected policy: 600 seconds per individual prediction (2026-11-11)

User clarified that the 600s limit applies independently to each explanation call,
not to the sum of repetitions. One over-limit call invalidates the entire
algorithm/device entry; a total experiment longer than600s is allowed. The new
watchdog guards accuracy, warm-up and timed calls separately using synchronous
progress messages and an independent parent process. IPC is outside prediction
timers. Previous timed samples are never published if a later call exceeds600s.

Active data: `first-order-per-prediction600/first-order.json`; log/pid:
`per-prediction600.log` / `per-prediction600.pid`. The source archive under
`full-suite-10/` is unchanged. The attempted aggregate-cutoff export under
`first-order-cutoff600/` is superseded and must not be used. Its helpers have been
moved to ignored superseded-sources for audit history.

All46 completed original first-order entries are reusable: their maximum timed
call is below600s, and conservative bounds on every untimed accuracy/warm-up call
are below600s. Those bounds use original per-model start timestamps, result-file
completion timestamps, recorded timed totals, and guaranteed time spent in failed
cases. They do not treat a sum of many short calls as one slow call. Audit JSON
records the bounds and source hashes; CalHousing-large CPU TreeSHAP is retained.
The GPU depth failure is unchanged. Original provenance_manifest points to the
source archive; the new policy and rerun manifest are saved alongside this export.

Fashion-MNIST-large CPU TreeSHAP was stopped by the old1800s aggregate budget;
that does not establish an individual-call failure. It is rerunning for the same
1000 rows, 10 timed repetitions, three warm-ups, 16 threads/cores20–35 and original
library with600s per call, without a shared experiment deadline. Its paper entry
is explicitly pending until it completes or one call exceeds600s. This correction
is currently scoped to the first-order table; do not infer per-call failure from
old aggregate pairwise timeouts without a separate audit.

Run `validate_per_prediction_timeout.py` to check that a total longer than the
limit succeeds and one slow call discards the entire entry. Run
`validate_prediction_limit600.py` for the retained-data certificates and rerun
outcome, then `validate_variability.py --allow-failures` on the active JSON for raw
sample moments. Watchdog tests and all46 retained entries passed validation.
Per-case status.json records the active call and each completed call duration.
The exact launch command/environment and script/library hashes are retained.


### Final per-call result (2026-11-11)

The rerun is complete: 47 supported first-order algorithm/device entries are valid,
with one unsupported GPU depth entry. No calls exceeded 600s. Fashion-MNIST-large
CPU TreeSHAP is 149.538050 ± 0.259070 s over ten timed calls; its slowest of all
14 calls was a 228.144208 s warm-up. Its speedup is 11.083346×. All 46 original
successful entries are retained unchanged. Both final validators passed; exports
are in first-order-per-prediction600/tables/. The paper table and prose have been
updated and compiled locally: CPU 11/12 wins, median 3.45×, max 12.85×.


## Second-order missing-entry check (2026-11-11)

User requested a common per-call 600-second failure rule for both runtime tables.
The two old CPU Fashion-MNIST second-order failures were aggregate-budget failures,
so check_pairwise_prediction600.py remeasures large and sparse models sequentially.
Uses the unchanged original CUDA library, model-cache models and seed432 rows;
100 explained rows, CPU cores20–35, nthread/OMP16, BLAS1, three warm-ups and ten
measured repetitions. The accuracy interaction call and auxiliary first-order
accuracy call each receive their own deadline. No aggregate budget applies.
Any one slow call voids the complete entry. Unsupported GPU depth stays unchanged.

The pairwise watchdog is a separate module so archived first-order provenance is
preserved. validate_pairwise_prediction_timeout.py passed: both accuracy phases,
warm-up and timed indexing verified, combined time may exceed the limit, a later
slow call discards earlier samples. Source snapshots, hashes, exact launch command,
per-call progress and final JSONs are retained under pairwise-per-prediction600/;
coordinator log/pid use that prefix in this experiment directory.


Final second-order outcome: both CPU Fashion-MNIST interaction calls exceeded
600 seconds individually and were stopped on their first accuracy call. Both
entries remain missing, with no samples or timing statistics. The independent
real-outcome validator passed for both; complete.json exists. No table values
changed. Paper wording now states the common individual-run 600-second rule for
both XGBoost runtime tables; the full paper compiled locally.
