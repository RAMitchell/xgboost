"""Measure realized distinct-feature depth before choosing a synthetic accuracy study."""
# pylint: disable=duplicate-code
# pylint: disable=missing-function-docstring,protected-access,too-many-locals,too-many-statements,wrong-import-position

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/2026-11-10-high-dimensional-accuracy"))
from reference import flatten


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, data):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(p)


def independent_stats(model):
    trees = json.loads(model.save_raw(raw_format="json"))["learner"][
        "gradient_booster"
    ]["model"]["trees"]
    maxdepth = 0
    maxdistinct = 0
    witness = None
    depths = []
    used = set()
    for ti, t in enumerate(trees):
        stack = [(0, [])]
        while stack:
            node, path = stack.pop()
            if t["left_children"][node] < 0:
                unique = len(set(path))
                depths.append(unique)
                maxdepth = max(maxdepth, len(path))
                if unique > maxdistinct:
                    maxdistinct = unique
                    witness = {"tree": ti, "leaf": node, "feature_sequence": path}
                continue
            f = int(t["split_indices"][node])
            used.add(f)
            for child in [t["left_children"][node], t["right_children"][node]]:
                stack.append((child, path + [f]))
    return maxdepth, maxdistinct, witness, depths, len(used)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-samples", type=int)
    p.add_argument("--n-informative", type=int)
    p.add_argument("--colsample-bylevel", type=float, default=1.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[432])
    p.add_argument("--depth", type=int, default=50)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--max-leaves", type=int, default=512)
    p.add_argument("--nthread", type=int, default=16)
    args = p.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = {
        k: v.default for k, v in inspect.signature(make_regression).parameters.items()
    }
    source = Path(__file__)
    identity = {
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "command": sys.argv,
        "source_sha256": sha(source),
        "reference_sha256": sha(
            ROOT / "experiments/2026-11-10-high-dimensional-accuracy/reference.py"
        ),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "library": str(Path(xgb.core._LIB._name).resolve()),
        "library_sha256": sha(xgb.core._LIB._name),
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "numpy": np.__version__,
        "generator_defaults": base,
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
    (out / source.name).write_bytes(source.read_bytes())
    results = []
    for seed in args.seeds:
        folder = out / f"seed{seed}"
        folder.mkdir(exist_ok=True)
        if (folder / "metrics.json").exists():
            results.append(
                json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            )
            continue
        generator = {"random_state": seed}
        if args.n_samples is not None:
            generator["n_samples"] = args.n_samples
        if args.n_informative is not None:
            generator["n_informative"] = args.n_informative
        x, y = make_regression(**generator)
        train, test = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=seed
        )
        np.savez_compressed(
            folder / "data.npz", x=x, y=y, train_indices=train, test_indices=test
        )
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cpu",
            "grow_policy": "lossguide",
            "max_depth": args.depth,
            "max_leaves": args.max_leaves,
            "eta": 0.05,
            "seed": seed,
            "nthread": args.nthread,
            "base_score": 0.5,
            "colsample_bylevel": args.colsample_bylevel,
        }
        started = time.monotonic()
        model = xgb.train(
            params,
            xgb.QuantileDMatrix(x[train], y[train], nthread=args.nthread),
            args.rounds,
        )
        model.save_model(folder / "model.ubj")
        arrays, _, stats = flatten(model)
        raw, distinct, witness, unique, used = independent_stats(model)
        assert (
            raw == stats["max_depth"] <= args.depth
            and distinct == stats["max_unique_depth"]
        )
        np.testing.assert_array_equal(np.sort(np.diff(arrays[0])), np.sort(unique))
        assert (
            0 <= stats["leaf_fraction_unique_gt16"] <= 1
            and 0 <= stats["cover_mass_fraction_unique_gt16"] <= 1
        )
        result = {
            "seed": seed,
            "generator": {**base, **generator},
            "training_params": params,
            "rounds": args.rounds,
            "train_rows": len(train),
            "heldout_rows": len(test),
            **stats,
            "unique_depth_percentiles": {
                str(q): float(np.percentile(unique, q))
                for q in [0, 25, 50, 75, 90, 95, 99, 100]
            },
            "features_used_across_forest": used,
            "deepest_distinct_path": witness,
            "elapsed_s": time.monotonic() - started,
            "model_sha256": sha(folder / "model.ubj"),
        }
        save(folder / "metrics.json", result)
        results.append(result)
        print(
            seed,
            "requested",
            args.depth,
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
