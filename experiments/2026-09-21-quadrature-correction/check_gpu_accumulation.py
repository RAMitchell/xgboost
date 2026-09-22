"""Isolate native GPU accumulation error using exact model slices and float64 sums."""
import json
import pickle
from pathlib import Path
import numpy as np
import xgboost as xgb
HERE=Path(__file__).resolve().parent
A=HERE/'artifacts'
name='cal_housing-large'
model=xgb.Booster(model_file=A/'benchmarks/models'/f'{name}.ubj')
with (A/'benchmarks'/f'{name}-rows.pkl').open('rb') as f:rows=pickle.load(f)[:16]
d=xgb.DMatrix(rows,nthread=8)
exact=np.load(A/'accuracy'/f'benchmark-{name}-exact.npy')
result=[]
for size in [1000,100,10]:
    total=np.zeros_like(exact)
    for start in range(0,1000,size):
        chunk=model[start:min(start+size,1000)]
        chunk.set_param({'device':'cuda','nthread':8,'shap_algorithm':'quadratureshap','quadratureshap_points':8})
        total+=chunk.predict(d,pred_contribs=True,strict_shape=True)[:,:,:-1].astype(np.float64)
    error=np.abs(total-exact)
    r={'trees_per_slice':size,'max_feature_error':float(error.max()),'mean_feature_error':float(error.mean())}
    result.append(r);print(r,flush=True)
(A/'gpu-accumulation-check.json').write_text(json.dumps(result,indent=2)+'\n')
