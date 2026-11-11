"""Validate saved real pairwise per-call outcomes independently of the runner."""
# pylint: disable=duplicate-code
# pylint: disable=consider-using-generator,missing-function-docstring

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pairwise-per-prediction600"


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["interactions"] is True and manifest["nrows"] == 100
    assert (
        manifest["niter"] == 10
        and manifest["warmup"] == 3
        and manifest["nthread"] == 16
    )
    assert manifest["prediction_timeout_seconds"] == 600
    assert manifest["affinity"] == list(range(20, 36))
    for name, field in [
        ("check_pairwise_prediction600.py", "driver_sha256"),
        ("pairwise_prediction_timeout.py", "watchdog_sha256"),
    ]:
        assert (
            hashlib.sha256((OUT / "sources" / name).read_bytes()).hexdigest()
            == manifest[field]
        )
    complete = json.loads((OUT / "complete.json").read_text(encoding="utf-8"))
    assert [r["model"] for r in complete["results"]] == manifest["models"]
    for r in complete["results"]:
        name = r["model"]
        assert (
            json.loads((OUT / f"{name}-cpu-treeshap.json").read_text(encoding="utf-8"))
            == r
        )
        assert r["prediction_timeout_seconds"] == 600
        calls = r["prediction_calls"]
        samples = r["samples_s"]
        for i, c in enumerate(calls):
            assert (
                c["index"] == i and c["elapsed_s"] > 0 and math.isfinite(c["elapsed_s"])
            )
            assert c["phase"] == (
                "accuracy" if i < 2 else "warmup" if i < 5 else "timed"
            )
        if r["error"] is None:
            assert len(calls) == 15 and max(c["elapsed_s"] for c in calls) <= 600
            assert r["n_samples"] == 10 and len(samples) == 10
            assert samples == [c["elapsed_s"] for c in calls if c["phase"] == "timed"]
            assert math.isclose(statistics.mean(samples), r["mean_time_s"])
            assert math.isclose(statistics.stdev(samples), r["std_time_s"])
        else:
            assert "individual prediction exceeded 600s" in r["error"], r
            assert not samples and r["n_samples"] == 0
            for key in [
                "mean_time_s",
                "std_time_s",
                "median_time_s",
                "min_time_s",
                "max_time_s",
            ]:
                assert r.get(key) is None
            if "timed_out_call" in r:
                t = r["timed_out_call"]
                assert t["observed_elapsed_s"] > 600 and t["index"] == len(calls)
                assert max([c["elapsed_s"] for c in calls], default=0) <= 600
            else:
                assert calls[-1]["elapsed_s"] > 600
        print(name, ":", r["error"] or f"{r['mean_time_s']:.6f} seconds", "— validated")
    print("PASS: both second-order per-call outcomes and archived provenance validated")


if __name__ == "__main__":
    main()
