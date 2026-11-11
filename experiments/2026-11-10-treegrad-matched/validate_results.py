"""Independently recompute timing moments and agreement from saved trial outputs."""
# pylint: disable=duplicate-code

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

for name in sys.argv[1:]:
    root = Path(name)
    checked = 0
    unsupported = 0
    for done in sorted(root.glob("*/complete.json")):
        marker = json.loads(done.read_text(encoding="utf-8"))
        if marker["status"] == "unsupported":
            unsupported += 1
            continue
        folder = done.parent
        m = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        assert m["precision"] == "float64"
        assert len(m["native_counts"]) == len(m["distinct_counts"]) == m["num_trees"]
        assert all(
            1 <= a <= b for a, b in zip(m["distinct_counts"], m["native_counts"])
        )
        assert max(m["distinct_counts"]) == math.ceil(m["max_distinct_depth"] / 2)
        baseline = np.load(folder / "tg_distinct-repeat0.npy")
        assert baseline.shape == (m["rows"], m["groups"], m["num_features"])
        for method, r in m["methods"].items():
            times = r["samples_s"]
            assert len(times) == len(r["row_samples_s"]) == m["niter"]
            assert all(v > 0 and math.isfinite(v) for v in times)
            assert math.isclose(statistics.mean(times), r["mean_s"])
            if len(times) > 1:
                assert math.isclose(statistics.stdev(times), r["sample_sd_s"])
            for sample, rows in zip(times, r["row_samples_s"]):
                assert len(rows) == m["rows"] and math.isclose(sum(rows), sample)
            values = np.load(folder / f"{method}-repeat0.npy")
            assert values.dtype == np.float64 and np.isfinite(values).all()
            for repeat in range(1, m["niter"]):
                np.testing.assert_array_equal(
                    values, np.load(folder / f"{method}-repeat{repeat}.npy")
                )
            if method != "q_fixed8":
                np.testing.assert_allclose(values, baseline, rtol=1e-8, atol=1e-10)
            assert math.isclose(
                np.max(np.abs(values - baseline)),
                r["max_abs_diff_vs_tg_distinct"],
                abs_tol=1e-16,
            )
        for key, a, b in [
            ("matched_native", "tg_native", "q_native_count"),
            ("matched_distinct", "tg_distinct", "q_distinct_count"),
            ("cached_native", "tg_native_cached", "q_native_count"),
            ("fixed8_vs_native", "tg_native", "q_fixed8"),
        ]:
            assert math.isclose(
                m["speedups"][key],
                m["methods"][a]["mean_s"] / m["methods"][b]["mean_s"],
            )
        checked += 1
        print(folder.name, "timings/counts/dtypes/agreement validated")
    print(
        root,
        checked,
        "supported models validated;",
        unsupported,
        "explicitly unsupported",
    )
    assert checked > 0, "No complete supported cases"
