"""Link worktree Python sources to an isolated CPU-only experiment library."""
# pylint: disable=duplicate-code

import argparse
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--library", type=Path, default=root / "build-cpu-test/lib/libxgboost.so"
)
parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path(__file__).resolve().parent / "python-runtime-standard",
)
args = parser.parse_args()
binary = args.library.resolve()
assert binary.is_file(), binary
package = args.out_dir.resolve() / "xgboost"
package.mkdir(parents=True, exist_ok=True)
for source in (root / "python-package/xgboost").iterdir():
    if source.name in {"lib", "__pycache__"}:
        continue
    link = package / source.name
    if not link.exists():
        link.symlink_to(source, target_is_directory=source.is_dir())
lib = package / "lib"
lib.mkdir(exist_ok=True)
link = lib / "libxgboost.so"
if link.is_symlink():
    link.unlink()
link.symlink_to(binary)
print(package.parent)
