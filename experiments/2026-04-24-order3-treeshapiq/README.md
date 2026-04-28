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

## What To Commit

Commit these scripts and this README:

- `benchmark_order3.py`
- `summarize_order3.py`
- `plot_sparse_order_scaling.py`
- optionally `benchmark_ksii_efficiency_depth.py` and `probe_fashion_sparse_trees.py` as exploratory scripts

Do not commit:

- `results-*.json`
- `*.csv`
- `*.png`
- `logs/`
- cached model files
