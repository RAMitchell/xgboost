"""Recompute stored accuracy metrics from raw arrays; do not assume eight points suffice."""
# pylint: disable=duplicate-code

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

for directory in sys.argv[1:]:
    out = Path(directory)
    cases = sorted(out.glob("*/metrics.json"))
    assert cases, "No completed cases"
    for path in cases:
        m = json.loads(path.read_text(encoding="utf-8"))
        folder = path.parent
        assert (
            hashlib.sha256((folder / "model.ubj").read_bytes()).hexdigest()
            == m["model_sha256"]
        )
        with np.load(folder / "raw.npz") as raw:
            ref = raw["reference"]
            check = raw["reference_double_points"]
            assert np.isfinite(ref).all() and np.isfinite(check).all()
            assert (
                ref.shape[0] == m["reference_rows"]
                and ref.shape[-1] == m["num_features"] + 1
            )
            assert raw["x_eval"].shape[0] == m["eval_rows"]
            assert len(set(raw["eval_indices"]) & set(raw["train_indices"])) == 0
            np.testing.assert_allclose(ref, check, rtol=1e-10, atol=1e-12)
            assert np.isclose(
                np.max(np.abs(ref - check)),
                m["reference_double_points_max_gap"],
                atol=1e-16,
            )
            for label, metrics in m["methods"].items():
                pred = raw[label].astype(np.float64)
                assert np.isfinite(pred).all() == metrics["finite"]
                err = pred[: len(ref), :, :-1] - ref[:, :, :-1]
                denom = np.linalg.norm(ref[:, :, :-1].ravel())
                actual = np.linalg.norm(err.ravel()) / denom if denom > 0 else None
                recorded = metrics["attribution_nrmse"]
                if actual is None or not np.isfinite(actual):
                    assert recorded is None
                else:
                    np.testing.assert_allclose(actual, recorded, rtol=1e-12, atol=1e-16)
                margin = raw["margin"][: len(pred)]
                residual = np.abs(pred.sum(axis=-1) - margin)
                finite = residual[np.isfinite(residual)]
                if len(finite):
                    np.testing.assert_allclose(
                        np.max(finite),
                        metrics["efficiency_abs_error"]["max"],
                        rtol=1e-12,
                        atol=1e-16,
                    )
                scale = np.sqrt(np.mean((margin - ref[0, :, -1]) ** 2))
                np.testing.assert_allclose(
                    scale, metrics["prediction_centered_rms"], rtol=1e-12, atol=1e-16
                )
            for kind in ["endpoint", "standard"]:
                key = f"{kind}8_implementation_nrmse"
                if key in m:
                    diff = (
                        raw[f"cpu_{kind}8_float32"][: len(ref)].astype(np.float64)
                        - raw[f"{kind}8_float64"]
                    )
                    denom = np.linalg.norm(ref[:, :, :-1].ravel())
                    if denom > 0:
                        np.testing.assert_allclose(
                            np.linalg.norm(diff[:, :, :-1].ravel()) / denom,
                            m[key],
                            rtol=1e-12,
                            atol=1e-16,
                        )
            for label, limit in [("standard8_float64", 16), ("endpoint8_float64", 8)]:
                if label in raw.files and m["max_unique_depth"] <= limit:
                    np.testing.assert_allclose(raw[label], ref, rtol=1e-10, atol=1e-12)
        print(
            folder.name,
            "validated; realized distinct-feature depth",
            m["max_unique_depth"],
        )
    print(directory, len(cases), "completed cases validated")
