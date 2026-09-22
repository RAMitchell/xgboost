"""Figure 1: maximum native feature error against sufficient-order float64 quadrature."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from reference import Reference, numeric_rows

HERE = Path(__file__).resolve().parent
A = HERE / 'artifacts'
OUT = A / 'figure1-accuracy'
OUT.mkdir(exist_ok=True)
rows_path = A / 'sweep/rows.npy'
rows = np.load(rows_path)
assert len(rows) == 512
row_hash = hashlib.sha256(rows_path.read_bytes()).hexdigest()
records = []
for depth in [4, 8, 12, 16, 24, 32, 48, 55, 64]:
    path = A / 'sweep' / f'depth{depth}.ubj'
    model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    dest = OUT / f'depth{depth}.json'
    if dest.exists():
        record = json.loads(dest.read_text())
        assert record['model_sha256'] == model_hash and record['rows_sha256'] == row_hash
        records.append(record)
        continue
    model = xgb.Booster(model_file=path)
    model.set_param({'nthread':32, 'device':'cpu'})
    ref = Reference(model)
    points = max(2, (ref.max_unique_depth + 1) // 2)
    print('Reference', depth, 'points', points, flush=True)
    exact = ref.explain(numeric_rows(rows), points)
    check = ref.explain(numeric_rows(rows), points + 8)
    crosscheck = float(np.max(np.abs(exact - check)))
    assert np.isfinite(exact).all() and np.isfinite(check).all()
    assert crosscheck < 1e-9, crosscheck
    np.save(OUT / f'depth{depth}-exact.npy', exact)
    record = {'requested_depth':depth, 'realized_depth':ref.max_depth,
              'max_unique_depth':ref.max_unique_depth, 'exact_points':points,
              'rows':len(rows), 'classes':ref.ngroups, 'features':rows.shape[1],
              'model_sha256':model_hash, 'rows_sha256':row_hash,
              'reference_crosscheck_max_abs':crosscheck, 'methods':[]}
    d = xgb.DMatrix(rows, feature_names=model.feature_names, nthread=32)
    for label, algorithm, n in [('TreeSHAP','treeshap',8)] + [
            (f'QuadratureSHAP-{n}','quadratureshap',n) for n in [4,6,8,16]]:
        model.set_param({'shap_algorithm':algorithm, 'quadratureshap_points':n})
        native = model.predict(d, pred_contribs=True, strict_shape=True)[:,:,:-1]
        assert native.shape == exact.shape and np.isfinite(native).all()
        error = np.abs(native.astype(np.float64) - exact)
        index = np.unravel_index(np.argmax(error), error.shape)
        entry = {'algorithm_label':label, 'max_feature_abs':float(error.max()),
                 'worst_row_class_feature':[int(i) for i in index]}
        record['methods'].append(entry)
        print(depth, entry, flush=True)
    dest.write_text(json.dumps(record, indent=2) + '\n')
    records.append(record)
(A / 'figure1-accuracy.json').write_text(json.dumps(records, indent=2) + '\n')
with (A / 'figure1-accuracy.csv').open('w', newline='') as f:
    fields = ['requested_depth','realized_depth','max_unique_depth','exact_points',
              'rows','classes','features','reference_crosscheck_max_abs',
              'algorithm_label','max_feature_abs']
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for record in records:
        for method in record['methods']:
            writer.writerow({key:(method[key] if key in method else record[key]) for key in fields})
print('Completed maximum feature-error figure data', flush=True)
