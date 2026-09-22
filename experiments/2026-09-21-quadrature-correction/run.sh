#!/usr/bin/env bash
# Activate conda environment xgboost first. This is resumable on existing artifacts.
set -euo pipefail
study_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
study_root="$(cd -- "$study_dir/../.." && pwd)"
cd "$study_root"
: "${CONDA_PREFIX:?Activate the xgboost conda environment first}"
: "${CXX:?The conda compiler environment must be active}"
git submodule update --init dmlc-core gputreeshap
cmake -S . -B build -GNinja -DUSE_CUDA=OFF -DGOOGLE_TEST=OFF
cmake --build build -j4
cmake -S . -B build-gpu -GNinja -DUSE_CUDA=ON -DUSE_NCCL=OFF \
  -DGOOGLE_TEST=OFF -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DGPU_COMPUTE_VER=120 \
  -DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=ON
cmake --build build-gpu -j4
python - "$study_root" "$study_dir" <<'PY'
import shutil, sys
from pathlib import Path
root, here = map(Path, sys.argv[1:])
dst = here/'artifacts/python-gpu/xgboost'
shutil.copytree(root/'python-package/xgboost',dst,
                ignore=shutil.ignore_patterns('__pycache__','lib'),dirs_exist_ok=True)
(dst/'lib').mkdir(exist_ok=True)
link = dst/'lib/libxgboost.so'
link.unlink(missing_ok=True)
link.symlink_to(root/'build-gpu/lib/libxgboost.so')
PY
export NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHONPATH="$study_root/python-package" python "$study_dir/check_reference.py"
PYTHONPATH="$study_root/python-package" python -u "$study_dir/sweep.py"
export PYTHONPATH="$study_dir/artifacts/python-gpu"
python "$study_dir/record_provenance.py"
python -u "$study_dir/prepare_models.py"
python -u "$study_dir/run_accuracy.py"
python -u "$study_dir/check_gpu_accumulation.py"
python -u "$study_dir/run_timings.py"
PYTHONPATH="$study_root/python-package" python -u "$study_dir/figure_accuracy.py"
python "$study_dir/validate_results.py"
python "$study_dir/report.py"
