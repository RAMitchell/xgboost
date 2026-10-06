"""Reproducible baseline/candidate DART training and prediction comparison.

Run in separate processes with each XGBoost package on PYTHONPATH. Saves complete
validation curves and final margins, plus full-model-versus-cache discrepancies.
No held-out study test sets are accessed.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.datasets import make_classification, make_regression
import xgboost as xgb


class CacheCheck(xgb.callback.TrainingCallback):
    def __init__(self, data, device, threads, last):
        self.data, self.device, self.threads, self.last = data, device, threads, last
        self.max_error = 0.

    def after_iteration(self, model, epoch, evals_log):
        if epoch == self.last:
            # xgb.train resets caches on return, so check before that reset.
            cached = model.predict(self.data, output_margin=True)
            fresh = xgb.Booster(model_file=model.save_raw())
            fresh.set_param(dict(device=self.device, nthread=self.threads))
            expected = fresh.predict(self.data, output_margin=True)
            self.max_error = float(np.max(np.abs(cached - expected)))
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--rates", type=float, nargs="+", default=[0., .01, .1, .5, 1.])
    parser.add_argument("--tasks", nargs="+", default=["regression", "classification", "multiclass", "ranking"])
    parser.add_argument("--normalizations", nargs="+", default=["tree", "forest"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        if task in ["regression", "ranking"]:
            x, y = make_regression(n_samples=args.rows, n_features=20,
                                   n_informative=10, noise=20., random_state=9)
            y = (y - y.mean()) / y.std()
            if task == "ranking":
                y = np.digitize(y, [-1., -.3, .3, 1.])
            objective = "reg:squarederror" if task == "regression" else "rank:ndcg"
        else:
            classes = 2 if task == "classification" else 5
            x, y = make_classification(n_samples=args.rows, n_features=20,
                                       n_informative=10, n_classes=classes, random_state=9)
            objective = "binary:logistic" if classes == 2 else "multi:softprob"
        x = x.astype(np.float32)
        split = (len(x) * 4 // 5) // 10 * 10
        dtrain = xgb.QuantileDMatrix(x[:split], label=y[:split], nthread=args.threads)
        dvalid = xgb.QuantileDMatrix(x[split:], label=y[split:], ref=dtrain,
                                    nthread=args.threads)
        if task == "ranking":
            dtrain.set_group([10] * (split // 10))
            dvalid.set_group([10] * ((len(x) - split) // 10))
        for normalization in args.normalizations:
            for rate in args.rates:
                for seed in range(args.seeds):
                    key = f"{task}_{normalization}_p{rate}_seed{seed}"
                    params = dict(objective=objective, tree_method="hist", device=args.device,
                                  nthread=args.threads, max_depth=4, learning_rate=.1,
                                  rate_drop=rate, normalize_type=normalization, seed=seed,
                                  subsample=.8, colsample_bytree=.9)
                    if task == "multiclass":
                        params["num_class"] = 5
                    history = {}
                    check = CacheCheck(dvalid, args.device, args.threads, args.rounds - 1)
                    start = time.perf_counter()
                    model = xgb.train(params, dtrain, args.rounds,
                                      evals=[(dtrain, "train"), (dvalid, "valid")],
                                      evals_result=history, verbose_eval=False, callbacks=[check])
                    elapsed = time.perf_counter() - start
                    cached = model.predict(dvalid, output_margin=True)
                    fresh = xgb.Booster(model_file=model.save_raw())
                    fresh.set_param(dict(device=args.device, nthread=args.threads))
                    reference = fresh.predict(dvalid, output_margin=True)
                    np.savez(args.output / f"{key}.npz", margin=cached, fresh=reference)
                    result = dict(key=key, params=params, rounds=args.rounds, rows=args.rows,
                                  fit_seconds=elapsed, history=history,
                                  cache_max_abs_error=check.max_error,
                                  build=xgb.build_info())
                    (args.output / f"{key}.json").write_text(json.dumps(result, indent=2))
                    print(key, round(elapsed, 3), result["cache_max_abs_error"], flush=True)


if __name__ == "__main__":
    main()
