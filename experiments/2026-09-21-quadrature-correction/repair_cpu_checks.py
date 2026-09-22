"""Complete CPU point-count checks rejected by the GPU build during initial runs."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import xgboost as xgb
from reference import Reference
HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts/accuracy'
for path in sorted(OUT.glob('sweep-*.json')):
    record=json.loads(path.read_text())
    bad=[r for r in record.get('native',[]) if r['device']=='cpu' and r['algorithm']=='quadratureshap' and 'error' in r]
    if not bad:
        continue
    name=record['name']
    env=dict(os.environ,PYTHONPATH=str(HERE.parents[1]/'python-package'))
    subprocess.run([sys.executable,str(HERE/'cpu_sweep_predictions.py'),name],env=env,check=True)
    exact_path=OUT/f'sweep-{name}-exact.npy'
    if exact_path.exists():
        exact=np.load(exact_path)
    else:
        model=xgb.Booster(model_file=HERE/'artifacts/sweep'/f'depth{name}.ubj')
        rows=np.load(HERE/'artifacts/sweep/rows.npy')[:100]
        ref=Reference(model)
        exact=ref.explain(rows,record['exact_points'])
        np.save(exact_path,exact)
    predictions=np.load(OUT/f'cpu-sweep-{name}.npz')
    for entry in bad:
        pred=predictions[str(entry['points'])]
        error=np.abs(pred-exact)
        entry['initial_configuration_error']=entry.pop('error')
        entry.update(max_feature_abs=float(error.max()),mean_feature_abs=float(error.mean()),finite=bool(np.isfinite(pred).all()),library='CPU-only build')
    path.write_text(json.dumps(record,indent=2)+'\n')
    print('REPAIRED CPU POINT COUNTS',name,flush=True)
