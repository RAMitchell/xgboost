"""Build a source-only submission zip from a committed tree and pinned submodules.

Only Git-tracked files are included. Local results, models, notes, dependency
checkouts, old experiment directories and Git history are never copied.
"""

# Numerical state and orchestration are kept explicit for reproduction.
# pylint: disable=too-many-locals

import argparse
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path


def add_tree(archive, repository, revision, prefix=""):
    """Add committed sources and recursively include pinned submodules."""
    payload = subprocess.check_output(
        ["git", "archive", "--format=tar", revision], cwd=repository
    )
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as tree:
        for member in tree:
            if not member.isfile():
                continue
            name = prefix + member.name
            if name.startswith(("experiments/", "notes/")):
                raise ValueError(
                    f"Historical/local files must not be tracked in submission: {name}"
                )
            archive.writestr(name, tree.extractfile(member).read())
    entries = subprocess.check_output(
        ["git", "ls-tree", "-rz", revision], cwd=repository
    ).split(b"\0")
    for entry in entries:
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, _kind, sha = metadata.split()
        if mode != b"160000":
            continue
        subpath = name.decode()
        directory = repository / subpath
        if not directory.exists():
            raise ValueError(f"Initialize submodule {directory}")
        add_tree(archive, directory, sha.decode(), prefix + subpath + "/")


def main():
    """Run the documented command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ref", default="HEAD")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        a.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        add_tree(archive, root, a.ref)
    print(a.output)


if __name__ == "__main__":
    main()
