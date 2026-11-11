"""Check the largest comparator discrepancy with a higher-point quadrature control."""
# pylint: disable=duplicate-code

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from adapters import CACHE, canonical_trees, load_old, pack
from validate_results import values

root = Path(__file__).resolve().parent / "full-10-idfix"
model = "covtype-sparse"
folder = root / model
b = xgb.Booster()
b.load_model(CACHE / f"{model}.ubj")
old = load_old()
trees = canonical_trees(old, b)
row = np.load(folder / "row.npy")
control = old.OrderQuadrature(trees, len(row), 3, 16).explain(row)
keys, a = pack(control)
np.savez_compressed(folder / "quadrature16-control.npz", keys=keys, values=a)
results = {}
for method in ["quadrature", "treeshapiq"]:
    observed = values(folder / method / "timed-03.npz")
    union = sorted(set(control) | set(observed))
    error = np.array([observed.get(k, 0) - control.get(k, 0) for k in union])
    ref = np.array([control.get(k, 0) for k in union])
    results[method] = {
        "max_abs_error_vs_16": float(np.max(np.abs(error), initial=0)),
        "normalized_l2_vs_16": float(np.linalg.norm(error) / np.linalg.norm(ref)),
    }
assert results["quadrature"]["max_abs_error_vs_16"] < 1e-10
(folder / "higher-point-control.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
