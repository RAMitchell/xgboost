"""Measure distinct-feature depth of numerically encoded Adult models."""
# pylint: disable=duplicate-code
# pylint: disable=missing-function-docstring,protected-access,too-many-locals,too-many-statements,wrong-import-position

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "experiments/2026-11-10-high-dimensional-accuracy/reference.py"
PROBE = ROOT / "experiments/2026-11-11-make-regression-depth/probe_depth.py"
sys.path.insert(0, str(REF.parent))
sys.path.insert(0, str(PROBE.parent))
from probe_depth import independent_stats
from reference import flatten


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[432, 433, 434])
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--colsample-bylevel", type=float, default=1.0)
    parser.add_argument("--nthread", type=int, default=16)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    x, target = fetch_openml(data_id=179, return_X_y=True)
    y = np.asarray(target != "<=50K", dtype=np.int32)
    categorical = list(x.select_dtypes(include=["category", "object", "str"]).columns)
    numerical = [c for c in x.columns if c not in categorical]
    identity = {
        "command": sys.argv,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": sha(__file__),
        "reference_sha256": sha(REF),
        "depth_checker_sha256": sha(PROBE),
        "library": str(Path(xgb.core._LIB._name).resolve()),
        "library_sha256": sha(xgb.core._LIB._name),
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "numpy": np.__version__,
        "openml_data_id": 179,
        "rows": len(x),
        "original_features": len(x.columns),
        "categorical_columns": categorical,
        "numerical_columns": numerical,
        "dataframe_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(x, index=True).values.tobytes()
        ).hexdigest(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            k: os.environ.get(k)
            for k in [
                "PYTHONPATH",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            ]
        },
    }
    if (out / "manifest.json").exists():
        assert (
            json.loads((out / "manifest.json").read_text(encoding="utf-8")) == identity
        )
    else:
        save(out / "manifest.json", identity)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    results = []
    for seed in args.seeds:
        case = out / f"seed{seed}"
        case.mkdir(exist_ok=True)
        if (case / "metrics.json").exists():
            results.append(
                json.loads((case / "metrics.json").read_text(encoding="utf-8"))
            )
            continue
        train, test = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y
        )
        encoder = OneHotEncoder(
            handle_unknown="ignore", drop=None, sparse_output=False, dtype=np.float32
        )
        encoder.fit(x.iloc[train][categorical])
        encoded = np.concatenate(
            [
                x[numerical].to_numpy(dtype=np.float32),
                encoder.transform(x[categorical]),
            ],
            axis=1,
        )
        assert encoded.dtype == np.float32 and np.isfinite(encoded).all()
        names = numerical + list(encoder.get_feature_names_out(categorical))
        schema = {
            "categorical_columns": categorical,
            "numerical_columns": numerical,
            "encoded_feature_names": names,
            "categories": {
                c: [None if pd.isna(v) else str(v) for v in values]
                for c, values in zip(categorical, encoder.categories_)
            },
            "encoder_fit": "training rows only",
            "drop": None,
            "handle_unknown": "ignore",
            "storage": "dense float32; zeros are explicit",
        }
        save(case / "encoding.json", schema)
        joblib.dump(encoder, case / "encoder.joblib")
        np.savez_compressed(
            case / "data.npz", x=encoded, y=y, train_indices=train, test_indices=test
        )
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": "cpu",
            "grow_policy": "lossguide",
            "max_depth": args.depth,
            "max_leaves": 512,
            "eta": 0.05,
            "seed": seed,
            "nthread": args.nthread,
            "base_score": 0.5,
            "colsample_bylevel": args.colsample_bylevel,
        }
        started = time.monotonic()
        model = xgb.train(
            params,
            xgb.QuantileDMatrix(encoded[train], y[train], nthread=args.nthread),
            30,
        )
        model.save_model(case / "model.ubj")
        arrays, _, stats = flatten(model)
        raw, distinct, witness, unique, used = independent_stats(model)
        assert (
            raw == stats["max_depth"] <= args.depth
            and distinct == stats["max_unique_depth"]
        )
        np.testing.assert_array_equal(np.sort(np.diff(arrays[0])), np.sort(unique))
        payload = json.loads(model.save_raw(raw_format="json"))["learner"][
            "gradient_booster"
        ]["model"]
        assert all(not t["categories"] for t in payload["trees"]), (
            "One-hot model must have no categorical split nodes"
        )
        result = {
            "dataset": "adult_onehot",
            "seed": seed,
            "requested_depth": args.depth,
            "encoded_columns": encoded.shape[1],
            "numeric_columns": len(numerical),
            "indicator_columns": encoded.shape[1] - len(numerical),
            "train_rows": len(train),
            "heldout_rows": len(test),
            "training_params": params,
            "rounds": 30,
            **stats,
            "features_used_across_forest": used,
            "unique_depth_percentiles": {
                str(q): float(np.percentile(unique, q))
                for q in [0, 25, 50, 75, 90, 95, 99, 100]
            },
            "deepest_distinct_path": witness,
            "elapsed_s": time.monotonic() - started,
            "model_sha256": sha(case / "model.ubj"),
        }
        save(case / "metrics.json", result)
        results.append(result)
        print(
            seed,
            "columns",
            encoded.shape[1],
            "raw",
            raw,
            "distinct",
            distinct,
            "leaf>16",
            stats["leaf_fraction_unique_gt16"],
            "mass>16",
            stats["cover_mass_fraction_unique_gt16"],
            flush=True,
        )
    save(out / "results.json", {"results": results})
    save(out / "complete.json", {"completed_seeds": len(results)})


if __name__ == "__main__":
    main()
