# TreeSHAP-IQ vs Quadrature TreeSHAP Experiments

This directory contains the scripts used for the paper comparison between a Python quadrature implementation of Shapley interaction indices and TreeSHAP-IQ from `shapiq`.

Generated result data, plots, logs, and cached models are intentionally not committed.

## Environment

Run from the repository root using the XGBoost conda environment:

```bash
/home/nfs/rorym/anaconda3/bin/conda run -n xgboost python <script> ...
```

The benchmark scripts expect cached XGBoost models under:

```text
experiments/2026-04-10-rapids-style-shap-benchmark/model-cache/
```

Regenerate them with `demo/guide-python/quadratureshap_rapids_benchmark.py --model-dir ... --models-only`.

## Main Order-3 SII Benchmark

One row is explained per model. The benchmark compares order-3 `SII` values and records runtime plus value agreement.

```bash
/home/nfs/rorym/anaconda3/bin/conda run -n xgboost \
  python experiments/2026-04-24-order3-treeshapiq/benchmark_order3.py \
  --models cal_housing-small cal_housing-sparse covtype-small covtype-sparse \
  --rows 1 \
  --order 3 \
  --index SII \
  --points 8 \
  --timeout 600 \
  --out experiments/2026-04-24-order3-treeshapiq/results-four-models-order3-sii-row1.json

/home/nfs/rorym/anaconda3/bin/conda run -n xgboost \
  python experiments/2026-04-24-order3-treeshapiq/summarize_order3.py \
  --input experiments/2026-04-24-order3-treeshapiq/results-four-models-order3-sii-row1.json \
  --out-md experiments/2026-04-24-order3-treeshapiq/sii-speed-table.md \
  --out-csv experiments/2026-04-24-order3-treeshapiq/sii-speed-table.csv
```

## CovType Sparse Order Scaling

This is the high-order runtime experiment used for the order-scaling plot. Each run explains one row with `SII`.

```bash
for order in 2 3 4 5 6; do
  /home/nfs/rorym/anaconda3/bin/conda run -n xgboost \
    python experiments/2026-04-24-order3-treeshapiq/benchmark_order3.py \
    --models covtype-sparse \
    --rows 1 \
    --order ${order} \
    --index SII \
    --points 8 \
    --timeout 43200 \
    --out experiments/2026-04-24-order3-treeshapiq/results-covtype-sparse-order${order}-sii-row1.json
done

/home/nfs/rorym/anaconda3/bin/conda run -n xgboost \
  python experiments/2026-04-24-order3-treeshapiq/plot_sparse_order_scaling.py \
  --model covtype-sparse \
  --inputs experiments/2026-04-24-order3-treeshapiq/results-covtype-sparse-order*-sii-row1*.json \
  --out-md experiments/2026-04-24-order3-treeshapiq/covtype-sparse-order-scaling-2to6.md \
  --out-csv experiments/2026-04-24-order3-treeshapiq/covtype-sparse-order-scaling-2to6.csv \
  --out-chart experiments/2026-04-24-order3-treeshapiq/covtype-sparse-order-scaling-speedup-2to6.png \
  --out-time-chart experiments/2026-04-24-order3-treeshapiq/covtype-sparse-order-scaling-time-2to6.png \
  --out-log-time-chart experiments/2026-04-24-order3-treeshapiq/covtype-sparse-order-scaling-time-log-2to6.png
```

The order-6 run can take several hours because TreeSHAP-IQ construction dominates.

## TreeGrad Appendix Benchmark

The appendix benchmark compares first-order quadrature TreeSHAP against TreeGrad's stable
TreeGrad-Shap implementation. TreeGrad expects sklearn-style trees, so the harness converts the
cached XGBoost models before calling TreeGrad. XGBoost categorical splits are supported by the
quadrature implementation but are recorded as unsupported for the TreeGrad adapter.

TreeGrad is not vendored in this repository. Clone it next to the benchmark run:

```bash
git clone --depth 1 https://github.com/watml/TreeGrad /tmp/TreeGrad
```

Run the appendix table with ten explanation rows per model:

```bash
/home/nfs/rorym/anaconda3/bin/conda run -n xgboost \
  python experiments/2026-04-24-order3-treeshapiq/benchmark_treegrad_first_order.py \
  --models \
    adult-small adult-large adult-sparse \
    cal_housing-small cal_housing-large cal_housing-sparse \
    covtype-small covtype-large covtype-sparse \
    fashion_mnist-small fashion_mnist-large fashion_mnist-sparse \
  --rows 10 \
  --quadrature-points 8 \
  --timeout 7200 \
  --treegrad-root /tmp/TreeGrad \
  --out experiments/2026-04-24-order3-treeshapiq/results-treegrad-first-order-all-row10.json \
  --out-md experiments/2026-04-24-order3-treeshapiq/treegrad-first-order-all-row10.md
```

## What To Commit

Commit these scripts and this README:

- `benchmark_order3.py`
- `benchmark_treegrad_first_order.py`
- `summarize_order3.py`
- `plot_sparse_order_scaling.py`

Do not commit:

- `results-*.json`
- `*.csv`
- `*.png`
- `logs/`
- cached model files
