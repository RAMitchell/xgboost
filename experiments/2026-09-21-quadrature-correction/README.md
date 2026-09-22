# Corrected QuadratureTreeSHAP paper verification

This experiment follows up the confirmed Figure 1 discrepancy. The production
change is only the quadrature node/weight mapping in `quadrature.h`: use standard
Gauss-Legendre on [0,1], rather than applying t=s^2.

Source baseline: origin/shapley-value-algorithms at de5718b22.
Original paper hardware: Xeon E5-2698 v4 / Tesla V100-SXM2-32GB.
Rerun hardware: Threadripper PRO 7975WX / RTX PRO 6000 Blackwell.
New timings must be labeled with this hardware; they do not replace V100
measurements under their old hardware caption.

## Scope

- Figure 1: CPU depth sweep 4,8,12,16,24,32,48,55,64; 512 explained rows,
  100 rounds, Fashion-MNIST, max_leaves=1024, seed 20260421, 35 training threads; native accuracy uses 32 threads.
  Depth 55 matches the displayed figure endpoint; 64 matches the tracked script.
  Depth 8 reuses the model already verified in the first investigation.
- Tables 2/3: regenerate all 12 models from the tracked GPU-training script;
  1,000 first-order / 100 second-order explained rows, seed 432, 32 CPU threads,
  one untimed warm-up plus three timed predictions, 600s deadline per case.
  Device/algorithm parameter selection remains inside each timed call, as in
  the original script.
  Run CPU/GPU TreeSHAP and QuadratureTreeSHAP on identical models and inputs.
  Training seed explicitly 0 (the original default).
- First-order accuracy: 100 rows per sweep model, 16 per timing model.
  Compare individual contributions to a float64 reference using
  ceil(unique_feature_depth/2) points, cross-checked with eight extra points.
  Measure separate float64 8/16-point approximation errors.
- Second-order accuracy: one full feature-pair matrix per timing model against
  a float64 leaf-product reference with enough points for exactness, including
  diagonals. This tests every pair for that row, not all timing rows.
- Both references are verified against exhaustive coalition values on small
  multiclass, categorical, missing-value, repeated-feature trees.
- No original trained models, source results, or float64 appendix-generating
  scripts were found locally or in tracked experiment history. Appendix A.2/A.3
  provenance remains unverified. The new float64 measurements are independent
  corroboration, not a claim to reproduce those exact tables.
- Python higher-order/TreeGrad scripts already use standard Gauss-Legendre and
  are unaffected by the C++ node-mapping bug. They are not rerun here.

## Execution

Use conda environment `xgboost`. `sweep.py` uses the separate CPU build and local
`python-package`. The GPU build is under `build-gpu` with
`KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=ON`, CUDA 12.9, compute capability 120.
A copy of the Python package in `artifacts/python-gpu` points specifically to
that GPU library; it is used by model preparation, accuracy, and timings.

`prepare_models.py` retains models, sampled rows, parameters, metadata and model
hashes. `run_accuracy.py` processes models as they become available. Run
`run_timings.py` only after training, accuracy work and compilation finish, so
this task's workloads do not compete with timings. Every case retains its own
log, result, failure or timeout. Generated large artifacts are ignored by Git.

`check_reference.py` is the exhaustive first/second-order regression check.
`reference.py` contains the independent float64 traversal and bounded leaf-product
interaction calculation. Neither calls native SHAP for its reference values.

The GPU-enabled branch rejects CPU 4/6/16-point configuration as well as GPU
non-eight-point configuration. CPU sweep checks therefore use the separate
CPU-only library. Initial configuration failures are retained in the audit
records alongside their successful CPU-only replacements.

## Reproduce or resume

With the `xgboost` conda environment active (including Numba), run:

```bash
bash experiments/2026-09-21-quadrature-correction/run.sh
```

This uses compute capability 120 for the Blackwell GPU on this host. Adapt that
build flag and record the new hardware if reproducing elsewhere. Existing
models/results are reused; use a separate artifacts directory for a fresh run
or changed configuration. The shell script executes stages serially; the
original rerun overlapped model preparation and accuracy work, then waited for
all of them to finish before running the timing stage serially.

## Figure 1 metric

The current figure uses maximum absolute per-feature error:

`max_{image, class, feature} |native_SHAP - float64_reference_SHAP|`.

This covers all 512 saved images, all 10 classes and all 784 features, excluding
bias contributions. Predictions use the CPU float32 implementation for TreeSHAP
and 4/6/8/16-point QuadratureTreeSHAP. The independent reference uses standard
Gauss–Legendre in float64 with `max(2, ceil(d/2))` points, where `d` is the largest
number of distinct features on any root-to-leaf path. The first-order integrand
has degree at most `d-1`; this point count therefore integrates it exactly in
real arithmetic. Float64 roundoff remains and is checked by recomputing every
reference with eight extra points (maximum permitted discrepancy 1e-9).
This metric includes both approximation and native float32 arithmetic error.

Run `figure_accuracy.py` with `PYTHONPATH` pointing to the CPU-only
`python-package`, then `report.py`. The reference arrays, model/input hashes,
worst-error indices and measurements are retained in `artifacts/figure1-accuracy/`
and `artifacts/figure1-accuracy.{json,csv}`. The rendered figure is
`artifacts/figure1-corrected.{pdf,png}`. The former mean efficiency-error figure
is retained separately as `artifacts/figure1-efficiency.{pdf,png}`.
The earlier 100-row independent approximation checks remain separate diagnostics.
