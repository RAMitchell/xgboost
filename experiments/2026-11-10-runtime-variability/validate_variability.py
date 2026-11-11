"""Independent checks of benchmark moments and paper table conversion."""
# pylint: disable=duplicate-code
# pylint: disable=line-too-long

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "experiments/2026-04-10-rapids-style-shap-benchmark/summarize_runtime_variability.py"
)
spec = importlib.util.spec_from_file_location("summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)

# Known independent sample: mean=2, sample SD=1, population SD=sqrt(2/3).
payload = {"niter": 3, "std_ddof": 0}
assert math.isclose(
    summary.sample_sd({"std_time_s": math.sqrt(2 / 3)}, payload)[1], 1.0
)
assert summary.sample_sd({"std_time_s": None}, {"niter": 1}) == (1, None)
assert summary.sample_sd({"std_time_s": 1}, {"niter": 3, "std_ddof": 1}) == (3, 1)

for name in (
    "results-all-rows1000-niter3-additive-v3",
    "results-all-rows100-niter3-interactions-v4",
):
    source = ROOT / f"experiments/2026-04-10-rapids-style-shap-benchmark/{name}.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    result = summary.summarize(original, source)
    assert len(result) == 48
    original_by_key = {
        (r["model"], r["device"], r["algorithm"]): r for r in original["details"]
    }
    by_key = {(r["model"], r["device"], r["algorithm"]): r for r in result}
    for row in result:
        old = original_by_key[row["model"], row["device"], row["algorithm"]]
        assert row["mean_s"] == old["mean_time_s"]
        if old["std_time_s"] is not None:
            assert math.isclose(row["sample_sd_s"] ** 2, old["std_time_s"] ** 2 * 3 / 2)
    for old in original["results_table"]:
        for device, label in [
            ("cpu", "QuadratureTreeSHAP Speedup"),
            ("cuda", "QuadratureTreeSHAP (GPU) Speedup"),
        ]:
            if old[label] is not None and math.isfinite(old[label]):
                a = by_key[old["model"], device, "treeshap"]["mean_s"]
                b = by_key[old["model"], device, "quadratureshap"]["mean_s"]
                assert math.isclose(a / b, old[label])
    print(
        f"{name}: all 48 cases preserve means, failure states, and speedups; SD conversion passed"
    )

allow_failures = "--allow-failures" in sys.argv
for name in (arg for arg in sys.argv[1:] if arg != "--allow-failures"):
    payload = json.loads(Path(name).read_text(encoding="utf-8"))
    for row in payload["details"]:
        if row["error"] is not None:
            assert allow_failures, row
            assert row["mean_time_s"] is None and row["std_time_s"] is None
            assert not row.get("samples_s")
            print(
                f"  Preserved failed case: {row['model']} {row['device']} {row['algorithm']}: {row['error']}"
            )
            continue
        samples = row["samples_s"]
        assert len(samples) == row["n_samples"] == payload["niter"]
        assert all(value > 0 and math.isfinite(value) for value in samples)
        assert math.isclose(row["mean_time_s"], statistics.mean(samples))
        assert math.isclose(row["std_time_s"], statistics.stdev(samples))
        assert math.isclose(row["median_time_s"], statistics.median(samples))
        assert (
            min(samples)
            <= row["q25_time_s"]
            <= row["median_time_s"]
            <= row["q75_time_s"]
            <= max(samples)
        )
    print(
        f"{name}: all raw timing samples and summary statistics passed independent checks"
    )
