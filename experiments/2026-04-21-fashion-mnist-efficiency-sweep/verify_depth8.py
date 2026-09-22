"""Compare rule builds on identical saved models; run with local PYTHONPATH.

Example: python verify_depth8.py --dataset fashion --label original --out-dir results-depth8
Rebuild after changing the rule, then repeat with --label corrected (same out-dir).
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from benchmark_fashion_mnist_efficiency import (
    DEFAULT_SEED, efficiency_metrics, fetch_fashion_mnist, predict_contribs,
    sample_rows, train_model, tree_stats,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset', choices=['synthetic', 'fashion'], required=True)
p.add_argument('--label', required=True)
p.add_argument('--out-dir', type=Path, required=True)
a = p.parse_args()
a.out_dir.mkdir(parents=True, exist_ok=True)
model_path = a.out_dir / f'{a.dataset}.ubj'
rows_path = a.out_dir / f'{a.dataset}-rows.npy'
if model_path.exists():
    model = xgb.Booster(model_file=model_path)
    model.set_param({'nthread': 35})
    x_test = np.load(rows_path)
else:
    if a.dataset == 'fashion':
        x, y = fetch_fashion_mnist()
        x_test, _ = sample_rows(x, y, 512, DEFAULT_SEED)
        print('Training original depth-8 Fashion-MNIST configuration', flush=True)
        model = train_model(x, y, 8, DEFAULT_SEED, 1024)
    else:
        x_test = np.array(list(itertools.product([0., 1.], repeat=8)), dtype=np.float32)
        model = xgb.train({'objective': 'reg:squarederror', 'tree_method': 'exact',
                           'max_depth': 8, 'min_child_weight': 0, 'lambda': 0,
                           'eta': 1, 'base_score': 0, 'nthread': 2},
                          xgb.DMatrix(x_test, label=x_test.prod(axis=1)), num_boost_round=1)
    np.save(rows_path, np.asarray(x_test))
    model.save_model(model_path)
stats = tree_stats(model)
assert stats['max_max_depth'] == 8, stats
dtest = xgb.DMatrix(x_test, feature_names=model.feature_names)
margin = model.predict(dtest, output_margin=True)
reference = predict_contribs(model, dtest, 'treeshap', None)
results = [{'algorithm': 'TreeSHAP', **efficiency_metrics(reference, margin)}]
for points in [4, 6, 8, 16]:
    pred = predict_contribs(model, dtest, 'quadratureshap', points)
    row = {'algorithm': f'QuadratureSHAP-{points}', **efficiency_metrics(pred, margin),
           'max_feature_difference_vs_treeshap': float(np.max(np.abs(pred-reference)))}
    results.append(row)
    print(row, flush=True)
    np.save(a.out_dir / f'{a.dataset}-{a.label}-{points}.npy', pred)
    if a.label == 'corrected':
        np.testing.assert_allclose(pred, reference, rtol=0, atol=2e-6)
payload = {'dataset': a.dataset, 'label': a.label, 'stats': stats,
           'xgboost_library': str(xgb.core._LIB._name), 'results': results}
(a.out_dir / f'{a.dataset}-{a.label}.json').write_text(json.dumps(payload, indent=2)+'\n')
print(json.dumps(payload, indent=2), flush=True)
