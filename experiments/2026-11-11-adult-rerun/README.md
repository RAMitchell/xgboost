# Canonical Adult rerun

The existing benchmark loader now expands Adult categorical columns in place.
Adult remains the dataset name. OpenML 179 supplies 48,842 rows; 122 indicators
(including missing categories) plus two numerical columns give 124 features.
Dense float32 preserves explicit zeros. No category is dropped. The fixed nominal
vocabulary is defined before accuracy-study splits; no labels enter the encoder.
All three split seeds produce matrices identical to the earlier train-fit pilot.

Checkpoint before changing the representation: `b9275d9dc`. Old model binaries
are copied, not hard-linked, into `full/historical-model-cache/` before the three
canonical cache paths are replaced. Cached feature count and dataset signatures
are checked to reject accidentally using a historical model with expanded data.

Run with the xgboost environment Python:

```
python -u experiments/2026-11-11-adult-rerun/run_suite.py
```

The queue records commands, source hashes, logs, status and completion markers
under `full/`. CPU/GPU benchmarks use 16 threads on cores 20–35 and GPU 1.
Python comparisons use one thread on core 20. All timing jobs run sequentially.
First/second order use 10 repetitions after 3 warmups and independent 600-second
prediction limits. Third order uses the same repeat counts and separate
600-second setup and prediction limits. TreeGrad repeats the matched-count suite
(6 variants, 10 rows, 3 trials), including the fixed-eight comparison. The depth
study uses 3 seeds at requested depths 4/8/12/16/24/32/40/50, standard GL only.

A zero process exit means collection finished, not that all methods succeeded:
inspect raw per-method statuses for timeouts or unsupported GPU depths. After
collection, validate and summarize raw outputs before updating the paper. Other
datasets' historical measurements are not rerun by this queue.
