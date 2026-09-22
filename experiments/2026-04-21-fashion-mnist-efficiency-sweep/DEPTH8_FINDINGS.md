# Figure 1 depth-8 discrepancy

Verified on 2026-09-21. Experiment branch: `origin/shapley-value-algorithms`,
commit `de5718b22` (experiment introduced in `108402fd3`). Investigation and fix:
`codex/quadrature-figure1`, `/home/rorym/xgboost-worktrees/quadrature-figure1`.

## Cause

The paper's four-point exactness claim is correct for standard Gauss-Legendre
quadrature on [0, 1]. First-order SHAP at unique-feature depth d has integrand
degree at most d-1, so n points suffice when 2n-1 >= d-1.

But `src/predictor/interpretability/quadrature.h`, used by both CPU and GPU,
constructed `(s*s, 2*s*ws)` from standard nodes/weights `(s, ws)`. This performs
`t=s^2`, integrating `2*s*f(s^2)`. A degree-m polynomial becomes degree 2m+1.
Consequently the old rule guarantees exactness only when n >= d, rather than
n >= ceil(d/2). At d=8, four points can fail and eight suffice. This also explains
why the four-point curve is accurate at depth 4 and jumps at depth 8.

The fix is to store `(s, ws)` directly. CPU and GPU share this rule, but only CPU
was built and run in this investigation.

## Reproduction

Used the existing experiment's training and sampling functions, with 70,000
Fashion-MNIST rows, 784 features, 100 rounds / 1,000 trees, seed 20260421,
lossguide, max_leaves=1024, max_depth=8, eta=0.01, and 512 explained rows.
Every tree's realized maximum depth was exactly 8. Mean leaf count was 144.148.
The saved model and explained rows were reused unchanged after rebuilding.
The efficiency metric is the original float32 `abs(sum(phi)-margin)`, averaged
across rows and classes. Bias is included in the sum.

| Method | Original mean error | Corrected mean error |
| --- | ---: | ---: |
| TreeSHAP | 7.08140e-08 | 7.08140e-08 |
| Quadrature, 4 points | 3.28801e-04 | 7.61001e-08 |
| Quadrature, 6 points | 6.52459e-07 | 7.34715e-08 |
| Quadrature, 8 points | 7.39963e-08 | 8.07709e-08 |
| Quadrature, 16 points | 7.47137e-08 | 7.55535e-08 |

Four-point maximum efficiency error fell from 4.93622e-03 to 9.53674e-07.
Its maximum elementwise attribution difference from TreeSHAP fell from
9.39518e-04 to 3.57628e-07. All corrected point counts passed elementwise
comparison with TreeSHAP at absolute tolerance 2e-6 and zero relative tolerance.

Independent checks:

- A depth-8 AND tree on the complete eight-feature binary grid reproduces the
  failure. Four-point mean efficiency error falls from 1.75101e-04 to 2.04273e-09;
  maximum error falls from 1.69957e-03 to 1.19209e-07.
- `verify_quadrature.cc` checks the actual C++ rule against exact monomial
  integrals for degrees 0 through 2n-1, for n=4,6,8,16. The old rule fails
  (four-point maximum error 5.19441e-03). The corrected rule passes all cases
  with maximum error at most 2.23e-16.

These results verify the depth-8 discrepancy. The old figure's source CSV and
plotting assets are not tracked in the branch; the attached PDF was inspected
and its depth-8 error is consistent with this rerun. The complete depth sweep
and GPU predictions were not rerun. Figure 1 should be regenerated with the
corrected rule before publication; these measurements do not replace the
remaining depths or establish revised high-depth accuracy claims.

## Files and commands

`depth8-verification-results.json` contains the before/after measurements.
Generated models, rows, and predictions remain in the ignored `results-depth8/`.
`verify_depth8.py` saves/reuses these models to avoid retraining between builds.

From the isolated worktree, activate the xgboost conda environment:

```bash
source /home/rorym/miniforge3/etc/profile.d/conda.sh
conda activate xgboost
export PYTHONPATH=$PWD/python-package
cmake -S . -B build -GNinja -DUSE_CUDA=OFF -DGOOGLE_TEST=OFF
cmake --build build -j4
```

Run each dataset with `--label original` before the rule change, then repeat
with `--label corrected` after rebuilding, using the same output directory:

```bash
python experiments/2026-04-21-fashion-mnist-efficiency-sweep/verify_depth8.py \
  --dataset fashion --label corrected \
  --out-dir experiments/2026-04-21-fashion-mnist-efficiency-sweep/results-depth8
python experiments/2026-04-21-fashion-mnist-efficiency-sweep/verify_depth8.py \
  --dataset synthetic --label corrected \
  --out-dir experiments/2026-04-21-fashion-mnist-efficiency-sweep/results-depth8
```

Compile and run the standalone rule regression check:

```bash
$CXX -std=c++17 -fopenmp -I. -Iinclude -Idmlc-core/include \
  experiments/2026-04-21-fashion-mnist-efficiency-sweep/verify_quadrature.cc \
  -Llib -lxgboost -Wl,-rpath,$PWD/lib -o /tmp/verify-quadrature
/tmp/verify-quadrature
```
