"""Use the CPU-only library for point counts rejected by GPU-build configuration."""
import sys
from pathlib import Path
import numpy as np
import xgboost as xgb
HERE=Path(__file__).resolve().parent
name=sys.argv[1]
rows=np.load(HERE/'artifacts/sweep/rows.npy')[:100]
model=xgb.Booster(model_file=HERE/'artifacts/sweep'/f'depth{name}.ubj')
model.set_param({'nthread':8,'device':'cpu','shap_algorithm':'quadratureshap'})
d=xgb.DMatrix(rows,feature_names=model.feature_names,nthread=8)
result={}
for n in [4,6,8,16]:
    model.set_param('quadratureshap_points',n)
    result[str(n)]=model.predict(d,pred_contribs=True,strict_shape=True)[:,:,:-1]
np.savez(HERE/'artifacts/accuracy'/f'cpu-sweep-{name}.npz',**result)
