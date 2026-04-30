# TreeGrad Appendix Benchmark

This appendix experiment compares first-order quadrature TreeSHAP against TreeGrad's stable
TreeGrad-Shap implementation.

Generated result data, logs, plots, and cached models are intentionally not committed.

## Environment

Run from the repository root using the XGBoost conda environment:

```bash
/home/nfs/rorym/anaconda3/bin/conda run -n xgboost python <script> ...
```

The benchmark expects cached XGBoost models under:

```text
experiments/2026-04-10-rapids-style-shap-benchmark/model-cache/
```

TreeGrad is not vendored in this repository. Clone it next to the benchmark run:

```bash
git clone --depth 1 https://github.com/watml/TreeGrad /tmp/TreeGrad
```

## Appendix Table

The harness converts cached XGBoost models into sklearn-style tree objects before calling
TreeGrad. XGBoost categorical splits are supported by the quadrature implementation but are
recorded as unsupported for the TreeGrad adapter.

Run the appendix table with ten explanation rows per model:

```bash
/home/nfs/rorym/anaconda3/bin/conda run -n xgboost \
  python experiments/2026-04-30-treegrad-appendix/benchmark_treegrad_first_order.py \
  --models \
    adult-small adult-large adult-sparse \
    cal_housing-small cal_housing-large cal_housing-sparse \
    covtype-small covtype-large covtype-sparse \
    fashion_mnist-small fashion_mnist-large fashion_mnist-sparse \
  --rows 10 \
  --quadrature-points 8 \
  --timeout 7200 \
  --treegrad-root /tmp/TreeGrad \
  --out experiments/2026-04-30-treegrad-appendix/results-treegrad-first-order-all-row10.json \
  --out-md experiments/2026-04-30-treegrad-appendix/treegrad-first-order-all-row10.md
```

Do not commit:

- `results-*.json`
- `*.csv`
- `*.md` result tables
- `*.log`
