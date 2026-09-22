"""Record the loaded GPU binary, numerical environment, source and hardware."""
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy
import numba
import sklearn
import xgboost
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
lib=Path(xgboost.core._LIB._name).resolve()
def command(args):
    return subprocess.check_output(args,text=True,cwd=ROOT).strip()
data={'source_commit':command(['git','rev-parse','HEAD']),
      'source_diff':command(['git','diff','--','src/predictor/interpretability/quadrature.h']),
      'library':str(lib),'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),
      'build_info':xgboost.build_info(),'python':platform.python_version(),
      'numpy':numpy.__version__,'numba':numba.__version__,'sklearn':sklearn.__version__,
      'cpu':command(['lscpu']),
      'gpu':command(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv']),
      'scripts_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
(HERE/'artifacts/provenance.json').write_text(json.dumps(data,indent=2)+'\n')
