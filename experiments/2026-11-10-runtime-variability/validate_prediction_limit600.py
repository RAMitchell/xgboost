"""Validate per-prediction reuse certificates and any new watchdog result."""
# pylint: disable=duplicate-code
# pylint: disable=line-too-long

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
out = ROOT / "first-order-per-prediction600"
old = json.loads((ROOT / "full-suite-10/first-order.json").read_text(encoding="utf-8"))
new = json.loads((out / "first-order.json").read_text(encoding="utf-8"))
original = {(r["model"], r["device"], r["algorithm"]): r for r in old["details"]}
rows = {(r["model"], r["device"], r["algorithm"]): r for r in new["details"]}
audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
assert new["prediction_timeout_seconds"] == 600 and len(rows) == len(audit) == 48
counts = {}
for item in audit:
    key = (item["model"], item["device"], item["algorithm"])
    row = rows[key]
    action = item["action"]
    counts[action] = counts.get(action, 0) + 1
    if action.startswith("reuse"):
        assert row == original[key]
        assert max(row["samples_s"]) <= 600 and item["untimed_call_upper_s"] <= 600
    elif action.startswith("retain"):
        assert row == original[key] and row["error"] is not None
    else:
        assert action.startswith("replay")
        if row.get("status") == "pending":
            continue
        if row["error"] is None:
            assert len(row["prediction_calls"]) == 14
            assert max(c["elapsed_s"] for c in row["prediction_calls"]) <= 600
            assert row["samples_s"] == [
                c["elapsed_s"] for c in row["prediction_calls"] if c["phase"] == "timed"
            ]
        else:
            assert row["mean_time_s"] is None and not row["samples_s"]
cal = rows["cal_housing-large", "cpu", "treeshap"]
assert (
    cal["mean_time_s"]
    == original["cal_housing-large", "cpu", "treeshap"]["mean_time_s"]
)
assert sum(cal["samples_s"]) > 600 > max(cal["samples_s"])
print(
    "PASS: CalHousing-large retained despite total >600; all reuse certificates and available per-call outcomes valid"
)
print(counts, "pending", new["pending_cases"])
