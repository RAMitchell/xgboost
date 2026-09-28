###########################################
Experimental Ask/Tell Hyperparameter Search
###########################################

The :py:mod:`xgboost.hpo` module provides a sequential optimizer over a finite list
of configurations. The caller owns training, validation, early stopping, metrics,
class weights, and compute budgets. There is no training wrapper, automatic
cross-validation, or network access. This API is experimental.

Basic use
=========

A historical prior is a separate versioned artifact. The research reproduction
repository supplies a local ``prior/`` directory and ``candidates.json``:

.. code-block:: python

    import json
    import xgboost as xgb

    with open("candidates.json", encoding="utf-8") as stream:
        candidates = json.load(stream)
    prior = xgb.hpo.Prior.load("prior", candidates=candidates)
    optimizer = xgb.hpo.Optimizer(candidates, prior=prior, random_state=42)

    for _ in range(32):
        trial = optimizer.ask()
        loss = evaluate(trial.params)  # Caller-supplied validation objective.
        optimizer.tell(trial, loss)

    best_parameters = optimizer.best_trial.params

``ask()`` returns the same pending trial until it is reported. It does not reserve
parallel trials. Candidate dictionaries are copied; ``trial.params`` returns a
fresh dictionary. The same configuration is never proposed twice. Once every
candidate is completed or failed, ``ask()`` raises ``StopIteration``.

Use ``direction="maximize"`` for a metric that should increase. Historical prior
means always describe normal-rank *loss* (lower is better), even in this case.
A prior learned under a different metric is not automatically validated for the
new objective.

Failure and restart
===================

.. code-block:: python

    trial = optimizer.ask()
    try:
        loss = evaluate(trial.params)
    except EvaluationError as error:  # Your application's exception type.
        optimizer.tell(trial, status="failed", message=str(error))
    else:
        optimizer.tell(trial, loss)

    optimizer.save("optimizer.json")
    restored = xgb.hpo.Optimizer.load("optimizer.json")

A failure excludes that candidate without inserting an invented loss into the GP.
Failure messages are retained in ``history``. Non-finite losses are rejected and
leave the trial pending. A timeout may instead be reported as a completed scalar
objective when the caller explicitly defines a resource penalty. ``best_trial``
is ``None`` until an evaluation succeeds.

JSON state includes the candidate-aligned prior, history, seed, and pending trial;
it needs no model files or network on resume. Loading replays and checks all
recorded decisions. Numerical environment changes can alter decisions and cause
replay validation to fail. Use the same NumPy/BLAS environment for reproducibility.
State does not contain the user's training data or fitted objective models.

Prior and search space
======================

``Prior.load`` verifies checksums and projects two small XGBoost models onto the
candidate list. They supply historical mean and scale. A Matérn 5/2 kernel has
one offline-learned length scale per encoded coordinate, plus amplitude and noise.
Kernel parameters are fixed during a run. Projected priors are bound to candidate
order; reordering candidates requires reprojecting the prior.

The version-1 historical encoding supports:

=================== =============================================
Parameter           Supported values
=================== =============================================
learning_rate       [0.01, 0.3]
max_depth           0 (inactive), or integers 2 through 12
max_leaves          0 (inactive), or integers 8 through 256
grow_policy         depthwise, lossguide
min_child_weight    [1e-5, 100]
subsample           [0.5, 1]
colsample_bytree     [0.5, 1]
colsample_bylevel    [0.5, 1]
colsample_bynode     [0.5, 1]
reg_lambda          0, or [1e-5, 100]
reg_alpha           0, or [1e-5, 10]
gamma               0, or [1e-5, 10]
max_bin             128, 256, 512
=================== =============================================

All thirteen parameters must be explicit. At least one tree-size bound must be
active, and at most one column-sampling parameter may be below one. Other training
settings (objective, thread count, seed, etc.) are added by the caller. Values
outside the supported space are rejected, rather than clipped into the prior's
training domain. Validity inside these ranges is not a guarantee of dense
historical coverage.

``prior=None`` uses a zero-mean Matérn baseline with unit length scales, amplitude
one and observation variance 0.1, under the same encoding. For arbitrary candidate
spaces, callers can instead construct ``Prior(mean, covariance, noise_variance)``
with aligned arrays. The covariance must be positive definite, including numerical
jitter. The caller owns the meaning and provenance of a custom prior.

Algorithm and limitations
=========================

The first two successful observations are selected using independent prior
Thompson draws, excluding already attempted candidates. The posterior is updated
when at least two observed losses differ. Observations use average ranks for ties,
converted to normal quantiles. Subsequent proposals maximize expected latent
improvement over the best observed candidate, including covariance between the
candidate and incumbent. This uncertainty-driven acquisition does not add a
separate random-exploration fraction. All-tied observations leave the posterior
unchanged; uncertainty still guides subsequent proposals.

The evaluated historical prior was learned from small/medium, unweighted CPU
histogram classification (log loss) and regression (RMSE) workloads. It is not
validated for class-weighted training, other objectives, or unrestricted data
sizes. Rank normalization does not remove these differences. The implementation
is intended for small finite pools and few-shot budgets: prior storage is
quadratic in pool size and initial factorization is cubic. It performs no online
kernel fitting, dataset-characteristic conditioning, curve-guided proposals,
continuous acquisition optimization, or automatic parameter-space expansion.

The historical artifacts, data and reproduction instructions are available from
`the independent research repository <https://github.com/RAMitchell/xgboost-hpo>`_.
No artifact is bundled into the XGBoost package. A runnable caller-owned training
example is ``demo/guide-python/hpo.py``.

API
===

.. automodule:: xgboost.hpo

.. autoclass:: xgboost.hpo.Optimizer
   :members:

.. autoclass:: xgboost.hpo.Prior
   :members:

.. autoclass:: xgboost.hpo.Trial
   :members:

.. autoclass:: xgboost.hpo.Observation
   :members:
