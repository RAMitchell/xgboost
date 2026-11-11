"""Validate/export the remaining long case automatically when the suite finishes."""
# pylint: disable=duplicate-code

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

exp = Path(__file__).resolve().parent
out = exp / "full-10-idfix"
pid = int((exp / "full-idfix.pid").read_text(encoding="utf-8"))
while not (out / "complete.json").exists():
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        (out / "finalization.json").write_text(
            json.dumps(
                {
                    "status": "runner_exited_before_completion",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
        sys.exit(1)
    time.sleep(10)
try:
    with (out / "final-validation.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(exp / "validate_results.py"), str(out)],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    subprocess.run([sys.executable, str(exp / "render_table.py"), str(out)], check=True)
    (out / "finalization.json").write_text(
        json.dumps(
            {
                "status": "validated_and_exported",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
except subprocess.CalledProcessError as error:
    (out / "finalization.json").write_text(
        json.dumps(
            {
                "status": "validation_or_export_failed",
                "exit_code": error.returncode,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    raise
