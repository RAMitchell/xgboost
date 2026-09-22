"""Render measurements without hiding failures, missing cases, or hardware changes."""
import csv
import json
import statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
A=HERE/'artifacts'
lines=['# Corrected QuadratureTreeSHAP verification','',
       'Hardware: AMD Threadripper PRO 7975WX, NVIDIA RTX PRO 6000 Blackwell. '
       'Timing uses 32 CPU threads. These are fresh measurements, not V100 timings.','']
sweep_path=A/'sweep/results.json'
if sweep_path.exists():
    sweep=json.loads(sweep_path.read_text())['rows']
    depths=sorted({r['requested_depth'] for r in sweep})
    lines+=['## Efficiency diagnostic (previous Figure 1 metric)','','[Efficiency diagnostic (PDF)](artifacts/figure1-efficiency.pdf)','',f'Completed requested depths: {depths}.','',
            '| Depth | Realized depth | TreeSHAP mean error | Q4 | Q6 | Q8 | Q16 |',
            '| --- | --- | --- | --- | --- | --- | --- |']
    for depth in depths:
        rs={r['algorithm_label']:r for r in sweep if r['requested_depth']==depth}
        vals=[rs[k]['mean_efficiency_err'] for k in ['TreeSHAP','QuadratureSHAP-4','QuadratureSHAP-6','QuadratureSHAP-8','QuadratureSHAP-16']]
        lines.append(f'| {depth} | {rs["TreeSHAP"]["max_max_depth"]:.0f} | '+' | '.join(f'{v:.3e}' for v in vals)+' |')
    fig,ax=plt.subplots(figsize=(7.4,4.4))
    for label,color in zip(['TreeSHAP','QuadratureSHAP-4','QuadratureSHAP-6','QuadratureSHAP-8','QuadratureSHAP-16'],['tab:red','tab:orange','tab:green','tab:blue','tab:purple']):
        rs=sorted([r for r in sweep if r['algorithm_label']==label and r['requested_depth']<=55],key=lambda r:r['requested_depth'])
        ax.plot([r['requested_depth'] for r in rs],[r['mean_efficiency_err'] for r in rs],marker='o',color=color,label=label)
    ax.set_yscale('log');ax.set_xlabel('Requested max_depth');ax.set_ylabel('Mean absolute efficiency error')
    ax.set_xticks([4,8,12,16,24,32,48,55])
    ax.set_title('Fashion-MNIST: standard Gauss–Legendre quadrature')
    ax.grid(True,which='major',alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(A/'figure1-efficiency.png',dpi=200);fig.savefig(A/'figure1-efficiency.pdf');plt.close(fig)
figure_data = A / 'figure1-accuracy.json'
if figure_data.exists():
    data = json.loads(figure_data.read_text())
    lines += ['', '## Figure 1: maximum absolute feature error', '',
              '[Corrected figure (PDF)](artifacts/figure1-corrected.pdf)', '',
              'Maximum absolute difference over 512 images, 10 classes and 784 feature contributions '
              '(bias excluded), comparing native CPU float32 predictions with independent float64 '
              'Gauss–Legendre quadrature using ceil(maximum unique-feature depth / 2) points '
              '(minimum 2). Each reference is cross-checked with eight additional points. '
              'Exactness refers to polynomial integration in real arithmetic; float64 rounding remains.', '',
              '| Depth | Realized depth | Reference points | TreeSHAP | Q4 | Q6 | Q8 | Q16 |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    labels = ['TreeSHAP','QuadratureSHAP-4','QuadratureSHAP-6','QuadratureSHAP-8','QuadratureSHAP-16']
    for r in data:
        errors = {m['algorithm_label']:m['max_feature_abs'] for m in r['methods']}
        lines.append(f'| {r["requested_depth"]} | {r["realized_depth"]} | {r["exact_points"]} | ' +
                     ' | '.join(f'{errors[label]:.3e}' for label in labels) + ' |')
    fig, ax = plt.subplots(figsize=(7.4,4.4))
    shown = [r for r in data if r['requested_depth'] <= 55]
    for label, color in zip(labels, ['tab:red','tab:orange','tab:green','tab:blue','tab:purple']):
        values = [next(m['max_feature_abs'] for m in r['methods'] if m['algorithm_label']==label) for r in shown]
        ax.plot([r['requested_depth'] for r in shown], values, marker='o', color=color, label=label)
    ax.set_yscale('log')
    ax.set_xlabel('Requested max_depth')
    ax.set_ylabel('Maximum absolute SHAP-value error')
    ax.set_xticks([4,8,12,16,24,32,48,55])
    ax.grid(True,which='major',alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(A/'figure1-corrected.png',dpi=200)
    fig.savefig(A/'figure1-corrected.pdf')
    plt.close(fig)
accuracies=[json.loads(p.read_text()) for p in sorted((A/'accuracy').glob('*.json'))]
accuracies.sort(key=lambda r:(r['kind'],int(r['name']) if r['kind']=='sweep' else r['name']))
lines+=['','## Independent accuracy checks','',
        '| Model | Rows | Unique depth | Float64 Q8 max feature error | CPU Q8 max feature error | GPU Q8 max feature error | Reference cross-check |',
        '| --- | --- | --- | --- | --- | --- | --- |']
for a in accuracies:
    if 'float64' not in a:
        lines.append(f'| {a["kind"]}-{a["name"]} | ERROR: {a.get("error")} | | | | | |')
        continue
    native={(r['device'],r['algorithm'],r['points']):r for r in a['native']}
    vals=[]
    for device in ['cpu','cuda']:
        r=native.get((device,'quadratureshap',8),{})
        vals.append(f'{r["max_feature_abs"]:.3e}' if 'max_feature_abs' in r else 'ERROR')
    lines.append(f'| {a["kind"]}-{a["name"]} | {a["rows"]} | {a["max_unique_depth"]} | {a["float64"]["8"]["max_feature_abs"]:.3e} | '+ ' | '.join(vals)+f' | {a["reference_crosscheck_max_abs"]:.3e} |')
lines+=['','Pairwise comparisons use one complete feature-pair matrix per benchmark model.','',
        '| Model | Float64 Q8 max pair error | CPU Q8 max pair error | GPU Q8 max pair error |','| --- | --- | --- | --- |']
for a in accuracies:
    if 'pairwise' not in a:continue
    p=a['pairwise'];native={(r['device'],r['algorithm']):r for r in p['native']}
    vals=[]
    for device in ['cpu','cuda']:
        r=native.get((device,'quadratureshap'),{})
        vals.append(f'{r["max_pair_abs"]:.3e}' if 'max_pair_abs' in r else 'ERROR')
    lines.append(f'| {a["name"]} | {p["float64_8_max_abs"]:.3e} | '+' | '.join(vals)+' |')
timings=[json.loads(p.read_text()) for p in sorted((A/'timings').glob('*.json'))]
lines+=['','## Runtime tables','',f'{len(timings)}/96 cases recorded. A timeout is the original 600-second per-case deadline.','']
speedups=[]
for order in [1,2]:
    for device in ['cpu','cuda']:
        ratios=[]
        for model in sorted({r['model'] for r in timings}):
            pair={r['algorithm']:r for r in timings if r['model']==model and r['order']==order and r['device']==device and r['status']=='ok'}
            if len(pair)==2:
                ratios.append({'model':model,'speedup':pair['treeshap']['mean_s']/pair['quadratureshap']['mean_s']})
        if ratios:
            vals=[r['speedup'] for r in ratios]
            speedups.append({'order':order,'device':device,'paired_models':len(vals),'min':min(vals),'median':statistics.median(vals),'max':max(vals),'models':ratios})
            lines.append(f'Order {order}, {device}: {min(vals):.2f}–{max(vals):.2f}x speedup; median {statistics.median(vals):.2f}x across {len(vals)} completed pairs.')
            lines.append('')
(A/'speedups.json').write_text(json.dumps(speedups,indent=2)+'\n')

def timing_status(case):
    if case is None:
        return 'pending'
    if case['status']=='error':
        log=A/'timings'/f'{case["model"]}-{case["device"]}-{case["algorithm"]}-order{case["order"]}.log'
        if log.exists() and 'Tree depth must be < 32' in log.read_text():
            return 'unsupported depth'
    return case['status']
for order in [1,2]:
    lines += [f'### Order {order} ({1000 if order==1 else 100} rows)','',
              '| Model | CPU TreeSHAP (s) | CPU Q8 (s) | Speedup | GPU TreeSHAP (s) | GPU Q8 (s) | Speedup |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    rs=[r for r in timings if r['order']==order]
    for name in sorted({r['model'] for r in rs}):
        cases={(r['device'],r['algorithm']):r for r in rs if r['model']==name}
        vals=[]
        for device in ['cpu','cuda']:
            t=cases.get((device,'treeshap'));q=cases.get((device,'quadratureshap'))
            for c in [t,q]:
                vals.append(f'{c["mean_s"]:.6f}' if c and c['status']=='ok' else timing_status(c))
            vals.append(f'{t["mean_s"]/q["mean_s"]:.2f}x' if t and q and t['status']==q['status']=='ok' else '—')
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines.append('')
accum_path=A/'gpu-accumulation-check.json'
if accum_path.exists():
    accum=json.loads(accum_path.read_text())
    lines+=['','## Separate GPU accumulation limitation','',
            'CalHousing-large has unique-feature depth 8, so 8-point quadrature is mathematically exact. '
            'Its native GPU QuadratureTreeSHAP feature error reached 6.54e-05, compared with 2.05e-07 for GPUTreeSHAP. '
            'Splitting the same model into batches and summing their feature contributions in float64 reduces this error substantially:',
            '', '| Trees per slice | Maximum feature error |', '| --- | --- |']
    for r in accum:
        lines.append(f'| {r["trees_per_slice"]} | {r["max_feature_error"]:.3e} |')
    lines+=['','This supports cross-tree float32 accumulation as a major source of the deviation. '
            'The production algorithm was not changed for this diagnostic, and all timing tables use the standard implementation. '
            'Do not claim uniformly better native accuracy than TreeSHAP on all workloads.','']
lines+=['','## Provenance and limits','',
        '- See README.md for exact protocols and the distinction from the paper hardware.',
        '- Original model caches were unavailable; benchmark ensembles were regenerated from the tracked training script.',
        '- Float64 appendix-generating scripts were not found in the local checkout or tracked experiment history. The new checks do not establish provenance of the original appendix tables.',
        '- Full measurements, all baseline accuracy errors, model hashes, timing samples and failure logs are retained under artifacts/.',
        '- Python higher-order and TreeGrad-comparison scripts use standard Gauss–Legendre and were not rerun.','']
(HERE/'RESULTS.md').write_text('\n'.join(lines))
(A/'measurements.json').write_text(json.dumps({'accuracy':accuracies,'timings':timings},indent=2)+'\n')
print('Report updated:',len(accuracies),'accuracy cases;',len(timings),'timing cases')

# Flat tables for manuscript/spreadsheet updates; raw nested measurements remain in JSON.
def write_csv(path, rows):
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
flat_timings=[]
for r in timings:
    row={k:v for k,v in r.items() if k not in ['metrics','samples_s']}
    row.update(r.get('metrics',{}))
    for i,t in enumerate(r.get('samples_s',[]),1):row[f'sample_{i}_s']=t
    flat_timings.append(row)
write_csv(A/'timings.csv',flat_timings)
flat_accuracy=[]
for r in accuracies:
    for native in r.get('native',[]):
        flat_accuracy.append({'kind':r['kind'],'model':r['name'],'order':1,'rows':r['rows'],**native})
    for native in r.get('pairwise',{}).get('native',[]):
        flat_accuracy.append({'kind':r['kind'],'model':r['name'],'order':2,'rows':1,**native})
write_csv(A/'accuracy.csv',flat_accuracy)
models=[json.loads(p.read_text()) for p in sorted((A/'benchmarks').glob('*.json'))]
write_csv(A/'models.csv',[{k:v for k,v in m.items() if k!='parameters'} for m in models])
