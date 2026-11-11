"""Recompute repeated timings, check every saved output, and compare the methods."""
# pylint: disable=duplicate-code
# pylint: disable=missing-function-docstring,too-many-branches,too-many-locals,too-many-statements

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np


def values(path):
    a = np.load(path)
    assert a["values"].dtype == np.float64 and np.isfinite(a["values"]).all()
    assert a["keys"].shape == (len(a["values"]), 3)
    return {tuple(k): float(v) for k, v in zip(a["keys"], a["values"])}


def main():
    for name in sys.argv[1:]:
        root = Path(name)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        checked = 0
        assert (
            manifest["order"] == 3
            and manifest["points"] == 8
            and manifest["precision"] == "float64"
        )
        assert manifest["quadrature_rule"] == "standard_gauss_legendre"
        for path in (root / "sources").glob("*.py"):
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == manifest["source_sha256"][path.name]
            )
        agreement = []
        for model in manifest["models"]:
            methods = {}
            for method in ["quadrature", "treeshapiq"]:
                folder = root / model / method
                p = folder / "results.json"
                if not p.exists():
                    continue
                r = json.loads(p.read_text(encoding="utf-8"))
                limit = manifest["prediction_timeout_seconds"]
                assert r["prediction_timeout_seconds"] == limit
                if "reuse" in r:
                    record = r["reuse"]
                    source = Path(record["source"])
                    original = json.loads(
                        (source / "results.json").read_text(encoding="utf-8")
                    )
                    assert (
                        hashlib.sha256(
                            (source / "results.json").read_bytes()
                        ).hexdigest()
                        == record["original_result_sha256"]
                    )
                    assert {k: v for k, v in r.items() if k != "reuse"} == original
                    prior = json.loads(
                        Path(record["measurement_manifest"]).read_text(encoding="utf-8")
                    )
                    for key in [
                        "niter",
                        "warmup",
                        "rows",
                        "order",
                        "index",
                        "points",
                        "precision",
                        "quadrature_rule",
                        "prediction_timeout_seconds",
                        "setup_timeout_seconds",
                        "affinity",
                        "environment",
                        "library_sha256",
                        "model_sha256",
                    ]:
                        assert prior[key] == manifest[key], key
                    assert record["source_sha256"] == prior["source_sha256"]
                    for raw in folder.glob("*.npz"):
                        assert (
                            hashlib.sha256(raw.read_bytes()).hexdigest()
                            == hashlib.sha256(
                                (source / raw.name).read_bytes()
                            ).hexdigest()
                        )

                if r["status"] == "complete":
                    samples = r["samples_s"]
                    assert len(samples) == manifest["niter"] == r["n_samples"]
                    assert max(samples) <= limit and all(
                        t > 0 and math.isfinite(t) for t in samples
                    )
                    assert min(samples) == r["min_s"] and max(samples) == r["max_s"]
                    for call in r["prediction_calls"]:
                        bound = (
                            manifest["setup_timeout_seconds"]
                            if call["phase"] == "setup"
                            else limit
                        )
                        assert call["elapsed_s"] <= bound
                    assert math.isclose(
                        statistics.mean(samples), r["mean_s"], rel_tol=1e-13
                    )
                    assert math.isclose(
                        statistics.stdev(samples), r["sample_sd_s"], rel_tol=1e-13
                    )
                    calls = [c for c in r["prediction_calls"] if c["phase"] == "timed"]
                    assert samples == [c["elapsed_s"] for c in calls]
                    first = None
                    for i in range(manifest["warmup"] + manifest["niter"]):
                        phase = "warmup" if i < manifest["warmup"] else "timed"
                        observed = values(folder / f"{phase}-{i:02d}.npz")
                        if first is None:
                            first = observed
                        else:
                            assert first == observed, "Repeated output changed"
                    methods[method] = first
                else:
                    assert (
                        r["samples_s"] == []
                        and r["mean_s"] is None
                        and r["sample_sd_s"] is None
                    )
                    if r["status"] == "timeout":
                        t = r["timed_out_call"]
                        bound = (
                            manifest["setup_timeout_seconds"]
                            if t["phase"] == "setup"
                            else limit
                        )
                        assert (
                            t.get("observed_elapsed_s", t.get("elapsed_s", 0)) > bound
                        )
                    else:
                        raise AssertionError(r)
                checked += 1
                print(model, method, r["status"], "timings and outputs validated")
            if len(methods) == 2:
                a = methods["quadrature"]
                b = methods["treeshapiq"]
                keys = sorted(set(a) | set(b))
                delta = np.array([a.get(k, 0) - b.get(k, 0) for k in keys])
                reference = np.array([b.get(k, 0) for k in keys])
                maxgap = float(np.max(np.abs(delta), initial=0))
                scale = float(np.max(np.abs(reference), initial=0))
                assert maxgap < 1e-7 * max(1, scale), (model, maxgap, scale)
                norm = (
                    float(np.linalg.norm(delta) / np.linalg.norm(reference))
                    if np.linalg.norm(reference) > 0
                    else None
                )
                print(model, "method max abs diff", maxgap, "normalized L2", norm)
                agreement.append(
                    {"model": model, "max_abs_diff": maxgap, "normalized_l2_diff": norm}
                )
        print(root, checked, "method cases validated")
        (root / "validation-summary.json").write_text(
            json.dumps(
                {
                    "checked_methods": checked,
                    "agreement": agreement,
                    "suite_complete": (root / "complete.json").exists(),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
