"""Wait for all preparation, then time serially without this task's background load."""
import os
import subprocess
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent
for pid in map(int,sys.argv[1:]):
    while True:
        try: os.kill(pid,0)
        except ProcessLookupError: break
        time.sleep(10)
while not (HERE/'artifacts/accuracy.complete').exists():
    time.sleep(10)
print('All training and accuracy checks finished; starting serial timing cases.',flush=True)
subprocess.run([sys.executable,str(HERE/'run_timings.py')],check=True)
subprocess.run([sys.executable,str(HERE/'report.py')],check=True)
(HERE/'artifacts/run.complete').touch()
