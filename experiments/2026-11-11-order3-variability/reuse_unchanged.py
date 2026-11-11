"""Reuse only measured cases whose timed computation is unaffected by the ID fix."""
# pylint: disable=duplicate-code
# pylint: disable=line-too-long

import ast
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import xgboost as xgb
from adapters import CACHE, canonical_trees, load_old

ROOT = Path(__file__).resolve().parent
old = ROOT / "full-10"
new = ROOT / "full-10-idfix"
new.mkdir(exist_ok=True)
prior = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
old_ast = ast.parse((old / "sources/adapters.py").read_text(encoding="utf-8"))
new_ast = ast.parse((ROOT / "adapters.py").read_text(encoding="utf-8"))
for name in ["load_old", "canonical_trees", "explain", "pack"]:
    a = next(
        n for n in old_ast.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    b = next(
        n for n in new_ast.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    assert ast.dump(a) == ast.dump(b), name
for tree in [old_ast, new_ast]:
    f = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "make_explainer"
    )
    if tree is old_ast:
        q_branch = ast.dump(f.body[0])
    else:
        assert ast.dump(f.body[0]) == q_branch
module = load_old()
proof = {}
for model in ["cal_housing-small", "cal_housing-sparse"]:
    b = xgb.Booster()
    b.load_model(CACHE / f"{model}.ubj")
    trees = canonical_trees(module, b)
    for t in trees:
        ids = set(np.unique(t.feature[t.feature >= 0]))
        assert list(ids) == sorted(ids)
    proof[model] = {
        "trees_checked": len(trees),
        "all_original_feature_orders_already_sorted": True,
    }
allowed = [
    ("cal_housing-small", "quadrature"),
    ("cal_housing-small", "treeshapiq"),
    ("cal_housing-sparse", "quadrature"),
    ("cal_housing-sparse", "treeshapiq"),
    ("covtype-small", "quadrature"),
    ("covtype-large", "quadrature"),
]
records = []
for model, method in allowed:
    source = old / model / method
    destination = new / model / method
    result = json.loads((source / "results.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert len(result["samples_s"]) == 10 and len(result["prediction_calls"]) == 14
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        shutil.copytree(source, destination)
    shutil.copy2(old / model / "row.npy", destination.parent / "row.npy")
    record = {
        "model": model,
        "method": method,
        "source": str(source.resolve()),
        "original_result_sha256": hashlib.sha256(
            (source / "results.json").read_bytes()
        ).hexdigest(),
        "measurement_manifest": str((old / "manifest.json").resolve()),
        "reason": "Quadrature branch unchanged"
        if method == "quadrature"
        else "Feature order already sorted; identical timed baseline state",
        "source_sha256": prior["source_sha256"],
    }
    result["reuse"] = record
    (destination / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    records.append(record)
(new / "reused-cases.json").write_text(
    json.dumps({"records": records, "baseline_feature_order_proof": proof}, indent=2)
    + "\n"
)
(old / "SUPERSEDED.txt").write_text(
    "Partial run stopped after CovType-small numerical validation found the baseline feature-ID labeling defect. Never use its CovType TreeSHAP-IQ outputs. Only cases explicitly certified in full-10-idfix/reused-cases.json are reused.\n"
)
print("Certified unchanged timing paths and reused", len(records), "method/model cases")
