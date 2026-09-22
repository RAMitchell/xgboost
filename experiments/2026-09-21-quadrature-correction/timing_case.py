"""One isolated timing case, run by run_timings.py with a 600-second deadline."""
import argparse
import gc
import json
import pickle
import time
from pathlib import Path
import numpy as np
import xgboost as xgb

p=argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('device')
p.add_argument('algorithm')
p.add_argument('order',type=int)
p.add_argument('output',type=Path)
a=p.parse_args()
base=Path(__file__).resolve().parent/'artifacts/benchmarks'
with (base/f'{a.model}-rows.pkl').open('rb') as f:
    x=pickle.load(f)
x=x.iloc[:100] if a.order==2 and hasattr(x,'iloc') else x[:100] if a.order==2 else x
model=xgb.Booster(model_file=base/'models'/f'{a.model}.ubj')
model.set_param({'nthread':32,'device':a.device,'shap_algorithm':a.algorithm,'quadratureshap_points':8})
if a.device=='cuda':
    assert xgb.build_info()['USE_CUDA']
d=xgb.DMatrix(x,enable_categorical=True,nthread=32)
kwargs={'pred_interactions':True} if a.order==2 else {'pred_contribs':True}
def predict():
    # Match the original benchmark: parameter selection is inside each timed call.
    params={'device':a.device,'shap_algorithm':a.algorithm}
    if a.algorithm=='quadratureshap':
        params['quadratureshap_points']=8
    model.set_param(params)
    return np.asarray(model.predict(d,**kwargs))

pred=predict()
actual_device=json.loads(model.save_config())['learner']['generic_param']['device']
assert actual_device.startswith(a.device), (a.device,actual_device)
metrics={'finite':bool(np.isfinite(pred).all())}
if a.order==1:
    margin=model.predict(d,output_margin=True)
    error=np.abs(pred.sum(axis=-1)-margin)
    metrics.update(mean_efficiency_error=float(error.mean()),max_efficiency_error=float(error.max()))
else:
    additive=model.predict(d,pred_contribs=True)
    error=np.abs(pred.sum(axis=-1)-additive)
    metrics.update(max_row_sum_error=float(error.max()),mean_row_sum_error=float(error.mean()),
                   max_asymmetry=float(np.max(np.abs(pred-np.swapaxes(pred,-1,-2)))))
del pred
gc.collect()
samples=[]
for _ in range(3):
    t=time.perf_counter()
    pred=predict()
    samples.append(time.perf_counter()-t)
    del pred
payload={'model':a.model,'device':a.device,'algorithm':a.algorithm,'order':a.order,'rows':len(x),
         'samples_s':samples,'mean_s':float(np.mean(samples)),'std_s':float(np.std(samples)),
         'metrics':metrics,'library':xgb.core._LIB._name,'status':'ok'}
a.output.write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload),flush=True)
