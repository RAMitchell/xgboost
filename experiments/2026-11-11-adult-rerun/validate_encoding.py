"""Validate canonical Adult encoding and rejection of historical model caches."""

# pylint: disable=missing-function-docstring,invalid-name
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "experiments/2026-04-10-rapids-style-shap-benchmark/quadratureshap_rapids_benchmark.py"
)
spec = importlib.util.spec_from_file_location("adult_benchmark", SOURCE)
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)
x, y = bench.fetch_adult()
assert x.shape == (48842, 124)
assert all(dtype == np.float32 for dtype in x.dtypes)
assert np.isfinite(x.to_numpy()).all()
assert set(np.unique(x.to_numpy()[:, 2:])) == {0.0, 1.0}
assert x.attrs["benchmark_encoding"]["representation"] == "one_hot"
for seed in (432, 433, 434):
    path = (
        ROOT
        / f"experiments/2026-11-11-adult-onehot-depth/depth50-seeds3/seed{seed}/data.npz"
    )
    with np.load(path) as previous:
        np.testing.assert_array_equal(previous["x"], x.to_numpy())
        np.testing.assert_array_equal(previous["y"], y)
# Run against the archived cache, so this check remains valid after regeneration.
cache = Path(sys.argv[1])
dataset = bench.TestDataset("adult", "binary:logistic", x, y)
for suffix in ("small", "large", "sparse"):
    try:
        bench.train_or_load_model(
            dataset, bench.MODEL_SPECS[suffix], f"adult-{suffix}", cache, False
        )
    except ValueError as error:
        assert "different dataset representation" in str(error)
    else:
        raise AssertionError("Historical Adult cache was silently accepted")
print(
    json.dumps(
        {
            "shape": x.shape,
            "signature": x.attrs["benchmark_encoding"]["signature"],
            "status": "validated",
        }
    )
)
