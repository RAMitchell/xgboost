"""Exercise worker result delivery with a timing payload larger than a pipe buffer."""

import importlib.util
import statistics
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
path = (
    ROOT
    / "experiments/2026-04-10-rapids-style-shap-benchmark/quadratureshap_rapids_benchmark.py"
)
spec = importlib.util.spec_from_file_location("qbench_runtime", path)
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


def main():
    """Check large payload delivery, one-sample statistics and failed cases."""
    x = np.random.default_rng(0).normal(size=(32, 3)).astype(np.float32)
    data = xgb.DMatrix(x, label=x[:, 0], nthread=1)
    model = xgb.train({"nthread": 1, "max_depth": 2, "tree_method": "hist"}, data, 2)
    margin = model.predict(data, output_margin=True)
    result = bench.run_case_with_timeout(
        model, x, "cpu", "quadratureshap", False, 2000, margin, 60, warmup=1, nthread=1
    )
    assert result["error"] is None, result
    assert len(result["samples_s"]) == 2000
    assert np.isclose(result["std_time_s"], statistics.stdev(result["samples_s"]))
    print("PASS: 2,000-sample worker payload delivered without pipe/join deadlock")
    one = bench.run_case_with_timeout(
        model,
        x,
        "cpu",
        "treeshap",
        False,
        1,
        margin,
        None,
        warmup=0,
        nthread=1,
        prediction_timeout_seconds=None,
    )
    assert one["error"] is None and one["std_time_s"] is None and one["n_samples"] == 1
    print(
        "PASS: single repetition reports unavailable SD; inline worker path completed"
    )
    timeout = bench.run_case_with_timeout(
        model, x, "cpu", "treeshap", False, 1, margin, 0.001, warmup=0, nthread=1
    )
    assert timeout["mean_time_s"] is None and "exceeded" in timeout["error"]
    print("PASS: timed-out case stays missing rather than reporting zero runtime")


if __name__ == "__main__":
    main()
