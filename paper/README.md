# Reproducing the paper

This directory contains the experiment code for the current manuscript. The rest
of the repository provides the modified XGBoost C++/CUDA implementation. Run the
commands below from the repository root. Outputs and trained models are local
artifacts and are not included in the source bundle.

## Build and environment

The recorded environment uses Python 3.14, the package versions in
`requirements.txt`, an Intel Xeon E5-2698 v4, and an NVIDIA V100. Native timings use
16 CPU threads; Python comparisons use one. GPU model training is part of the
benchmark definition. Use this repository's XGBoost build, not a PyPI XGBoost
installation lacking the quadrature predictor.

```bash
git submodule update --init --recursive
cmake -S . -B build -DUSE_CUDA=ON -DUSE_NCCL=OFF -DGPU_COMPUTE_VER=70 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python -m pip install -r paper/requirements.txt
python -m pip install -e ./python-package --no-build-isolation
export PYTHONPATH="$PWD/python-package:$PWD"
export OPENBLAS_NUM_THREADS=1
```

The source zip includes submodule sources, so its users can omit the submodule
command. Adjust the CUDA architecture for other hardware. Keep other workloads
off the benchmark CPU/GPU; CPU affinity can be selected with `taskset` or `--cpu`.

## Experiments in the manuscript

| Paper result | Command/module | Settings |
|---|---|---|
| Benchmark-model table | `paper.native --models-only` | Four datasets, small/large/sparse; Adult has 124 one-hot features |
| First-order CPU/GPU runtime table | `paper.native` | 1000 rows, 10 repetitions, 3 warm-ups, 16 threads |
| Second-order CPU/GPU runtime table | `paper.native --interactions` | 100 rows, same repetition/thread settings |
| Python implementation ablation | `paper.ablation` | A–E, 128 rows, batch size 128 for C/D/E, 3 repetitions, one thread |
| Third-order interaction table | `paper.higher_order` | One row, SII order 3, 10 repetitions, one thread |
| CovType-sparse order-scaling figure | `paper.higher_order --order ...` | One row at each orders 2–6, one measured prediction |
| Four-dataset depth figure | `paper.accuracy`, `paper.plot_depth` | Depths 4–50, 3 seeds, normalized attribution RMSE |
| Float64 accuracy and conservative-error appendix | `paper.accuracy`, `paper.render_accuracy` | Fashion-MNIST, depths20/30/40/50, 100 rows, 8/16/reference points |

All quadrature uses standard Gauss–Legendre mapped linearly to [0,1]. Timing
repetitions explain the same model and input; they are not retraining seeds.
The runtime tables use a **600-second limit for each complete explanation call**.
One over-limit call invalidates the entire entry; there is no combined deadline
across repetitions. For scalar A/B, one explanation call covers all 128 rows.
Python setup has a separate 600-second guard, recorded separately from prediction
timeouts. Setup and warm-up are excluded from timing statistics. Standard
deviation is the sample SD of individual measured predictions.

### 1. Models and native runtime tables

```bash
export OMP_NUM_THREADS=16
python -m paper.native --models-only --model-dir paper/models --output paper/results/models.json
python -m paper.native --model-dir paper/models --nrows 1000 --niter 10 --warmup 3 --nthread 16 --output paper/results/first_order.json
python -m paper.native --model-dir paper/models --nrows 100 --interactions --niter 10 --warmup 3 --nthread 16 --output paper/results/second_order.json
python -m paper.render_native --inputs paper/results/first_order.json paper/results/second_order.json --out-dir paper/results/native_tables
```

Model generation uses the canonical dataset loader and GPU histogram training.
`--models-only` emits metadata including raw and distinct-feature depths. Cached
models are reused; `--force-retrain` explicitly rebuilds them. The benchmark seed
432 selects explained rows. The native GPU implementation uses eight points.

### 2. Python ablation

```bash
python -m paper.fetch_treegrad
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_SLP_VECTORIZE=1
python -m paper.ablation --model-dir paper/models --out-dir paper/results/ablation
python -m paper.render_python --input paper/results/ablation --out-dir paper/results/ablation_tables
```

A is the unmodified authors' TreeGrad-Shap reference at commit
`8f7fdc61decc24a2525f89c481932aa3098aa4b1`. B is scalar NumPy quadrature, C batches
the NumPy traversal, D compiles variable-point traversal/arithmetic using Numba,
and E specializes eight-point arithmetic. D/E share traversal, float64 precision
and compiler flags. Variable rules use `max(1, ceil(min(raw tree depth,F)/2))`
points per tree; E uses eight. There is no separate eight-point NumPy variant.

### 3. Higher-order interactions

```bash
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_SLP_VECTORIZE=0
python -m paper.higher_order --model-dir paper/models --out-dir paper/results/order 3
python -m paper.render_python --input paper/results/order 3 --out-dir paper/results/order 3_tables
for order in 2 3 4 5 6; do
  python -m paper.higher_order --model-dir paper/models --models covtype-sparse --order "$order" --niter 1 --warmup 0 --limit 43200 --setup-limit 43200 --out-dir "paper/results/scaling$order"
done
python -m paper.plot_scaling --inputs paper/results/scaling{2,3,4,5,6} --out-dir paper/results/scaling_figure
```

The third-order table covers Adult, CalHousing and CovType. Both methods return
SII interactions of the requested order, summing tree outputs across multiclass
groups in the higher-order comparison. Numeric XGBoost splits retain strict-left
threshold semantics; the TreeSHAP-IQ adapter preserves original feature IDs.
The scaling figure has a separate 12-hour budget: order 6 TreeSHAP-IQ construction
and prediction can exceed 600 seconds. Construction remains excluded from the
reported runtime. This is distinct from the runtime-table deadline.

### 4. Depth and float64 accuracy

```bash
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16
python -m paper.accuracy --out-dir paper/results/depth
python -m paper.plot_depth --input paper/results/depth --out-dir paper/results/depth_figure
python -m paper.accuracy --datasets fashion_mnist --depths 20 30 40 50 --seeds 432 --eval-rows 100 --reference-rows 100 --reference-min-points 1 --out-dir paper/results/float64
python -m paper.render_accuracy --input paper/results/float64 --out-dir paper/results/accuracy_tables
```

The depth study trains 30 boosting rounds with a 512-leaf cap, at depths
4/8/12/16/24/32/40/50 and seeds 432/433/434. The figure uses the first 16 held-out
rows per model and compares eight-point float32 quadrature against float32
TreeSHAP. NRMSE excludes the bias column and pools rows, output groups and features;
curves show the median and range over seeds. Actual distinct-feature depth is
recorded because it may saturate below the requested limit.

`reference.py` independently integrates leaf-path contributions in float64 using
an exactness-sufficient point count. Float64 efficiency errors use an independent
float64 evaluation of the model, avoiding a float32 prediction floor. The appendix
renderer reports maximum per-feature error, efficiency error, and maximum/mean
sum of absolute feature errors.

## Verification and submission bundle

```bash
python -m pytest -q tests/python/test_quadrature_shap_numba.py tests/python/test_paper_reproduction.py
python -m paper.bundle --output paper/results/submission.zip
```

The bundle is built from the committed Git tree, including the pinned C++
submodule contents. It excludes Git history and all local models, result files,
notes, dependency checkouts and historical experiment directories. Fetch the
pinned TreeGrad dependency after extraction. Raw outputs include per-call timing
records, source/model hashes and execution settings; retain those when rerunning.
