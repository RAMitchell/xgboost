"""Process available models as they finish; persist logs and failures."""
import json
import subprocess
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts/accuracy'
OUT.mkdir(exist_ok=True)
cases=[('sweep',str(d),HERE/'artifacts/sweep'/f'depth{d}.ubj') for d in [4,8,12,16,24,32,48,55,64]]
cases += [('benchmark',f'{d}-{s}',HERE/'artifacts/benchmarks'/f'{d}-{s}.json') for s in ['small','sparse','large'] for d in ['adult','cal_housing','covtype','fashion_mnist']]
while cases:
    available=[c for c in cases if c[2].exists()]
    if not available:
        time.sleep(10)
        continue
    kind,name,_=available[0]
    cases.remove(available[0])
    path=OUT/f'{kind}-{name}.json'
    if path.exists():
        continue
    print('ACCURACY',kind,name,flush=True)
    try:
        with (OUT/f'{kind}-{name}.log').open('w') as log:
            r=subprocess.run([sys.executable,str(HERE/'accuracy.py'),kind,name],stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        if r.returncode:
            raise RuntimeError(f'exit {r.returncode}')
    except (subprocess.TimeoutExpired,RuntimeError) as e:
        path.write_text(json.dumps({'name':name,'kind':kind,'status':'error','error':str(e)},indent=2)+'\n')
    print('DONE',kind,name,flush=True)
(HERE/'artifacts/accuracy.complete').touch()
