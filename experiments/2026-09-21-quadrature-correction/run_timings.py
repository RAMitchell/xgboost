"""Run all 96 timing cases serially, retaining every timeout/failure."""
import json
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts/timings'
OUT.mkdir(parents=True,exist_ok=True)
for order in [1,2]:
    for size in ['small','sparse','large']:
        for dataset in ['adult','cal_housing','covtype','fashion_mnist']:
            name=f'{dataset}-{size}'
            for device in ['cpu','cuda']:
                for algorithm in ['treeshap','quadratureshap']:
                    case=f'{name}-{device}-{algorithm}-order{order}'
                    output=OUT/f'{case}.json'
                    if output.exists():
                        continue
                    print('TIMING',case,flush=True)
                    try:
                        with (OUT/f'{case}.log').open('w') as log:
                            result=subprocess.run([sys.executable,str(HERE/'timing_case.py'),name,device,algorithm,str(order),str(output)],stdout=log,stderr=subprocess.STDOUT,timeout=600)
                        if result.returncode:
                            raise RuntimeError(f'process exit {result.returncode}; see case log')
                    except (subprocess.TimeoutExpired,RuntimeError) as e:
                        output.write_text(json.dumps({'model':name,'device':device,'algorithm':algorithm,'order':order,'status':'timeout' if isinstance(e,subprocess.TimeoutExpired) else 'error','error':str(e)},indent=2)+'\n')
                    print(output.read_text(),flush=True)
