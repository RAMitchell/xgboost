"""Regenerate the 12 documented GPU-trained models and persist identical inputs."""
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
import xgboost as xgb

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'artifacts/benchmarks'
OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('qbench',ROOT/'demo/guide-python/quadratureshap_rapids_benchmark.py')
b=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=b
spec.loader.exec_module(b)

def train(dataset,spec):
    params=dataset.set_params(spec.training_params())
    params.update(nthread=32, seed=0)
    return xgb.train(params,dataset.train_dmatrix(),spec.num_rounds,verbose_eval=False)
b.train_model=train
if __name__=='__main__':
    assert xgb.build_info()['USE_CUDA'], xgb.build_info()
    for size in ['small','sparse','large']:
        for dataset in ['adult','cal_housing','covtype','fashion_mnist']:
            name=f'{dataset}-{size}'
            meta=OUT/f'{name}.json'
            if meta.exists():
                continue
            print('PREPARE',name,flush=True)
            model=b.get_models(name,OUT/'models',False)[0]
            model.booster.set_param({'nthread':32})
            with (OUT/f'{name}-rows.pkl').open('wb') as f:
                pickle.dump(model.dataset.test_input(1000,432),f)
            stats=b.tree_stats(model.booster)
            stats.update(name=name,features=model.booster.num_features(),
                         num_rounds=model.spec.num_rounds,
                         parameters={**model.dataset.set_params(model.spec.training_params()),'seed':0,'nthread':32},
                         model_sha256=hashlib.sha256((OUT/'models'/f'{name}.ubj').read_bytes()).hexdigest())
            meta.write_text(json.dumps(stats,indent=2)+'\n')
            print('READY',name,stats,flush=True)
