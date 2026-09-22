"""Per-feature and pairwise error against independent float64 exact quadrature."""
import argparse
import os
import subprocess
import sys
import json
import pickle
import time
from pathlib import Path
import numpy as np
import xgboost as xgb
from reference import Reference, numeric_rows, interactions

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('kind',choices=['sweep','benchmark'])
p.add_argument('name')
a=p.parse_args()
base=HERE/'artifacts'
outdir=base/'accuracy'
outdir.mkdir(exist_ok=True)
if a.kind=='sweep':
    model_path=base/'sweep'/f'depth{a.name}.ubj'
    rows=np.load(HERE/'artifacts/sweep/rows.npy')[:100]
else:
    model_path=base/'benchmarks/models'/f'{a.name}.ubj'
    with (base/'benchmarks'/f'{a.name}-rows.pkl').open('rb') as f:
        rows=pickle.load(f)
    rows=rows.iloc[:16] if hasattr(rows,'iloc') else rows[:16]
model=xgb.Booster(model_file=model_path)
model.set_param({'nthread':8,'device':'cpu'})
ref=Reference(model)
x=numeric_rows(rows)
n=max(2,(ref.max_unique_depth+1)//2)
print('REFERENCE',a.kind,a.name,len(x),ref.max_depth,ref.max_unique_depth,n,flush=True)
exact=ref.explain(x,n)
np.save(outdir/f'{a.kind}-{a.name}-exact.npy',exact)
check=ref.explain(x,n+8)
record={'name':a.name,'kind':a.kind,'rows':len(x),'max_depth':ref.max_depth,
        'max_unique_depth':ref.max_unique_depth,'exact_points':n,
        'reference_crosscheck_max_abs':float(np.max(np.abs(exact-check))),
        'float64':{},'native':[]}
assert np.isfinite(exact).all()
assert record['reference_crosscheck_max_abs'] < 1e-9,record
for points in ([4,6,8,16] if a.kind=='sweep' else [8,16]):
    q=ref.explain(x,points)
    error=np.abs(q-exact)
    record['float64'][str(points)]={'max_feature_abs':float(error.max()),'mean_feature_abs':float(error.mean()),
                                  'max_total_abs':float(error.sum(axis=-1).max())}
cpu_sweep=None
if a.kind=='sweep':
    env=dict(os.environ,PYTHONPATH=str(HERE.parents[1]/'python-package'))
    subprocess.run([sys.executable,str(HERE/'cpu_sweep_predictions.py'),a.name],env=env,check=True)
    cpu_sweep=np.load(outdir/f'cpu-sweep-{a.name}.npz')
d=xgb.DMatrix(rows,feature_names=model.feature_names,enable_categorical=True,nthread=8)
for device in ['cpu','cuda']:
    for algorithm in ['treeshap','quadratureshap']:
        for points in ([4,6,8,16] if a.kind=='sweep' and device=='cpu' and algorithm=='quadratureshap' else [8]):
            entry={'device':device,'algorithm':algorithm,'points':points}
            try:
                if cpu_sweep is not None and device=='cpu' and algorithm=='quadratureshap':
                    native=cpu_sweep[str(points)]
                else:
                    model.set_param({'device':device,'shap_algorithm':algorithm,'quadratureshap_points':points})
                    native=model.predict(d,pred_contribs=True,strict_shape=True)[:,:,:-1]
                error=np.abs(native-exact)
                entry.update(max_feature_abs=float(error.max()),mean_feature_abs=float(error.mean()),finite=bool(np.isfinite(native).all()))
            except xgb.core.XGBoostError as e:
                entry['error']=str(e).splitlines()[0]
            record['native'].append(entry)
            print(entry,flush=True)
if a.kind=='benchmark':
    # One complete feature-pair matrix per model; reference uses bounded path factors.
    print('PAIR_REFERENCE_START',n,flush=True)
    pair_start=time.perf_counter()
    pair_exact=interactions(ref,x[:1],n)
    print('PAIR_REFERENCE_DONE',time.perf_counter()-pair_start,flush=True)
    pair_q8=interactions(ref,x[:1],8)
    error=np.abs(pair_q8-pair_exact)
    record['pairwise']={'rows':1,'float64_8_max_abs':float(error.max()),'native':[]}
    assert np.isfinite(pair_exact).all()
    np.save(outdir/f'benchmark-{a.name}-pair-exact.npy',pair_exact)
    row=rows.iloc[:1] if hasattr(rows,'iloc') else rows[:1]
    d1=xgb.DMatrix(row,feature_names=model.feature_names,enable_categorical=True,nthread=8)
    for device in ['cpu','cuda']:
        for algorithm in ['treeshap','quadratureshap']:
            entry={'device':device,'algorithm':algorithm}
            print('PAIR_NATIVE_START',entry,flush=True)
            case_path=outdir/f'{a.name}-{device}-{algorithm}-pair-result.json.tmp'
            try:
                subprocess.run([sys.executable,str(HERE/'pair_accuracy_case.py'),a.name,device,algorithm,str(case_path)],check=True,timeout=600)
                entry=json.loads(case_path.read_text())
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as e:
                entry['error']=str(e)
            record['pairwise']['native'].append(entry)
            print('PAIR',entry,flush=True)
(outdir/f'{a.kind}-{a.name}.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
