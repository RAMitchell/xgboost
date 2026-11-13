"""Fetch the unmodified authors' TreeGrad reference at the paper's pinned revision."""

import argparse
import subprocess
from pathlib import Path

URL = "https://github.com/watml/TreeGrad.git"
REVISION = "8f7fdc61decc24a2525f89c481932aa3098aa4b1"


def main():
    """Run the documented command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).parent / "dependencies/TreeGrad",
    )
    a = p.parse_args()
    if not a.directory.exists():
        subprocess.run(["git", "clone", URL, str(a.directory)], check=True)
    subprocess.run(
        ["git", "checkout", "--detach", REVISION], cwd=a.directory, check=True
    )
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=a.directory, text=True
    ).strip()
    if actual != REVISION:
        raise RuntimeError("Wrong TreeGrad reference revision")
    print(actual)


if __name__ == "__main__":
    main()
