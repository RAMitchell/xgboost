"""Rerun the documented CPU efficiency sweep, retaining models for accuracy checks."""
import importlib.util
from functools import lru_cache
import sys
from pathlib import Path
import xgboost as xgb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts/sweep'
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('sweep_original', ROOT / 'experiments/2026-04-21-fashion-mnist-efficiency-sweep/benchmark_fashion_mnist_efficiency.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
original_train = m.train_model
original_fetch = m.fetch_fashion_mnist

@lru_cache(maxsize=1)
def fetch_and_save_rows():
    x, y = original_fetch()
    x_test, _ = m.sample_rows(x, y, m.DEFAULT_TEST_ROWS, m.DEFAULT_SEED)
    np.save(OUT / 'rows.npy', np.asarray(x_test))
    return x, y

m.fetch_fashion_mnist = fetch_and_save_rows

def cached_train(x, y, depth, seed, max_leaves):
    path = OUT / f'depth{depth}.ubj'
    if depth == 8 and not path.exists():
        old = ROOT / 'experiments/2026-04-21-fashion-mnist-efficiency-sweep/results-depth8/fashion.ubj'
        if old.exists():
            path.write_bytes(old.read_bytes())
    if path.exists():
        model = xgb.Booster(model_file=path)
        model.set_param({'nthread': m.DEFAULT_THREADS, 'device': 'cpu'})
        return model
    model = original_train(x, y, depth, seed, max_leaves)
    model.save_model(path)
    return model

m.train_model = cached_train
if __name__ == '__main__':
    if not (OUT / 'rows.npy').exists():
        fetch_and_save_rows()
    sys.argv = [sys.argv[0], '--out-dir', str(OUT), '--max-leaves', '1024', '--depths', '4', '8', '12', '16', '24', '32', '48', '55', '64', '--points', '4', '6', '8', '16']
    if (OUT / 'results.json').exists():
        sys.argv += ['--reuse-json', str(OUT / 'results.json')]
    m.main()
