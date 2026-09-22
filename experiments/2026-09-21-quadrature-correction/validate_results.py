"""Audit result completeness and numerical invariants; retain expected baseline DNFs."""
import json
import numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent
A=HERE/'artifacts'
accuracy=[json.loads(p.read_text()) for p in (A/'accuracy').glob('*.json')]
assert len(accuracy)==21,len(accuracy)
for result in accuracy:
    assert 'float64' in result,result
    assert result['reference_crosscheck_max_abs']<1e-9,result
    for points,metric in result['float64'].items():
        if 2*int(points)>=result['max_unique_depth']:
            assert metric['max_feature_abs']<1e-11,(result['name'],points,metric)
    for row in result['native']:
        if row['algorithm']=='quadratureshap':
            assert row.get('finite') and 'error' not in row,(result['name'],row)
    for row in result.get('pairwise',{}).get('native',[]):
        if row['algorithm']=='quadratureshap':
            assert row.get('finite') and 'error' not in row,(result['name'],row)
timings=[json.loads(p.read_text()) for p in (A/'timings').glob('*.json')]
assert len(timings)==96,len(timings)
failures=[r for r in timings if r['status']!='ok']
for r in timings:
    if r['status']=='ok':
        assert len(r['samples_s'])==3 and all(t>0 for t in r['samples_s']),r
        assert r['metrics']['finite'],r
summary={'accuracy_records':len(accuracy),'timing_records':len(timings),
         'completed_timing_cases':len(timings)-len(failures),'timing_failures':failures,
         'quadrature_timing_failures':[r for r in failures if r['algorithm']=='quadratureshap'],
         'max_float64_q8_sweep_feature_error':max(r['float64']['8']['max_feature_abs'] for r in accuracy if r['kind']=='sweep'),
         'exactness_bound_checks':'passed','finite_native_quadrature_accuracy_checks':'passed'}
figure = json.loads((A/'figure1-accuracy.json').read_text())
assert [r['requested_depth'] for r in figure] == [4,8,12,16,24,32,48,55,64]
retained_reference_depths = []
for r in figure:
    assert (r['rows'],r['classes'],r['features']) == (512,10,784)
    assert 2*r['exact_points'] >= r['max_unique_depth']
    assert r['reference_crosscheck_max_abs'] < 1e-9
    assert {m['algorithm_label'] for m in r['methods']} == {
        'TreeSHAP','QuadratureSHAP-4','QuadratureSHAP-6','QuadratureSHAP-8','QuadratureSHAP-16'}
    assert all(np.isfinite(m['max_feature_abs']) and m['max_feature_abs'] >= 0 for m in r['methods'])
    exact = np.load(A/'figure1-accuracy'/f'depth{r["requested_depth"]}-exact.npy')
    assert exact.shape == (512,10,784) and np.isfinite(exact).all()
    old_path = A/'accuracy'/f'sweep-{r["requested_depth"]}-exact.npy'
    if old_path.exists():
        old = np.load(old_path)
        assert np.max(np.abs(exact[:100]-old)) < 1e-12
        retained_reference_depths.append(r['requested_depth'])
summary['figure1_metric'] = 'maximum absolute feature difference from sufficient-order float64 quadrature'
summary['figure1_records'] = len(figure)
summary['figure1_reference_crosscheck_max_abs'] = max(r['reference_crosscheck_max_abs'] for r in figure)
summary['figure1_original_100_row_reference_agreement'] = {'status':'passed', 'retained_reference_depths':retained_reference_depths}
(A/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
