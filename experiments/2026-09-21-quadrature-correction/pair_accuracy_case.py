"""One native pairwise check; parent enforces a 600-second deadline."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import xgboost as xgb
HERE=Path(__file__).resolve().parent
name,device,algorithm,output=sys.argv[1:]
base=HERE/'artifacts/benchmarks'
model=xgb.Booster(model_file=base/'models'/f'{name}.ubj')
model.set_param({'nthread':8,'device':device,'shap_algorithm':algorithm,'quadratureshap_points':8})
with (base/f'{name}-rows.pkl').open('rb') as f:rows=pickle.load(f)
rows=rows.iloc[:1] if hasattr(rows,'iloc') else rows[:1]
d=xgb.DMatrix(rows,feature_names=model.feature_names,enable_categorical=True,nthread=8)
entry={'device':device,'algorithm':algorithm}
try:
    native=model.predict(d,pred_interactions=True,strict_shape=True)[:,:,:-1,:-1]
    exact=np.load(HERE/'artifacts/accuracy'/f'benchmark-{name}-pair-exact.npy')
    error=np.abs(native-exact)
    entry.update(max_pair_abs=float(error.max()),mean_pair_abs=float(error.mean()),finite=bool(np.isfinite(native).all()))
except xgb.core.XGBoostError as e:
    entry['error']=str(e).splitlines()[0]
Path(output).write_text(json.dumps(entry)+'\n')
