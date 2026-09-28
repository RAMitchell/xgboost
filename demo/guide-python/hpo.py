"""Caller-owned training with the experimental sequential ask/tell optimizer.

Optionally pass --prior /path/to/xgboost-hpo/prior to use a historical artifact.
Without it, use the no-history fixed-kernel baseline. No network access required.
"""

import argparse

import numpy as np

import xgboost as xgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    x = rng.normal(size=(400, 5))
    y = (x[:, 0] * x[:, 1] + x[:, 2] > 0).astype(float)
    train = xgb.DMatrix(x[:300], label=y[:300], nthread=1)
    valid = xgb.DMatrix(x[300:], label=y[300:], nthread=1)
    candidates = [
        dict(
            learning_rate=eta,
            max_depth=depth,
            max_leaves=0,
            grow_policy="depthwise",
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=1.0,
            colsample_bylevel=1.0,
            colsample_bynode=1.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            gamma=0.0,
            max_bin=256,
        )
        for eta in [0.03, 0.1, 0.3]
        for depth in [2, 4, 6]
    ]
    prior = (
        xgb.hpo.Prior.load(args.prior, candidates=candidates) if args.prior else None
    )
    optimizer = xgb.hpo.Optimizer(candidates, prior=prior, random_state=7)
    for _ in range(6):
        trial = optimizer.ask()
        model = xgb.train(
            dict(
                trial.params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                nthread=1,
                seed=7,
            ),
            train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        optimizer.tell(trial, float(model.best_score))
        print(f"Trial {trial.number}: validation log loss {model.best_score:.5f}")
    assert optimizer.best_trial is not None
    print("Best evaluated parameters:", optimizer.best_trial.params)


if __name__ == "__main__":
    main()
