import xgboost as xgb
import numpy as np
import testing as tm
import unittest
from sklearn import datasets

try:
    from regression_test_utilities import run_suite, parameter_combinations, \
        assert_results_non_increasing
except ImportError:
    None


class TestDCT:
    def test_dct_basic(self):
        # Check DCT can learn to predict ones
        X = [[1], [1], [1]]
        y = np.ones(3)
        metric = "rmse"
        param = {"eta": 1.0, "eval_metric": metric, "booster": "gbdct", "max_bin": 2, "max_coefficients": 2}
        dtrain = xgb.DMatrix(X, y)
        bst = xgb.train(param, dtrain, 1)
        preds = bst.predict(dtrain)
        assert preds.all() == 1.0

    def test_dct(self):
        tm._skip_if_no_sklearn()
        variable_param = {'booster': ['gbdct'], 'eta': [0.5],
                          'nthread': [2], 'max_bin': [64, 256], 'max_coefficients': [8, 64]
                          }
        datasets = ["Boston", "Digits", "Cancer", "Sparse regression",
                    "Boston External Memory"]

        for param in parameter_combinations(variable_param):
            results = run_suite(param, 25, datasets, scale_features=False)
            assert_results_non_increasing(results, 1e-2)

    def test_dct_consistency(self):
        X, y = datasets.load_boston(True)
        param = {"eta": 1.0, "booster": "gbdct", "max_bin": 16, "max_coefficients": 4}
        dtrain = xgb.DMatrix(X, y)
        bst = xgb.train(param, dtrain)
        bst2 = xgb.train(param, dtrain)
        assert np.array_equal(bst.predict(dtrain), bst2.predict(dtrain))


'''
    def test_dct_digits(self):
        num_rounds = 50
        X, y = datasets.load_digits(return_X_y=True)
        param = {"num_class":10,"eta": 0.3, "booster": "gbdct", "max_bin": 256, "max_coefficients": 16, "objective": "multi:softmax","debug_verbose":1}
        param["num_thread"] =8
        dtrain = xgb.DMatrix(X, y)
        bst = xgb.train(param, dtrain, num_rounds, evals=[(dtrain, "train")])

    def test_dct_boston(self):
        num_rounds = 5
        X,y = load_boston(True)
        param = {"eta":1.0,"booster": "gbdct", "max_bin": 64, "max_coefficients": 16}
        dtrain = xgb.DMatrix(X, y)
        bst = xgb.train(param, dtrain, num_rounds, evals=[(dtrain,"train")])

    def test_dct_breast_cancer(self):
        num_rounds = 5
        X,y = load_breast_cancer(True)
        param = {"eta":1.0,"booster": "gbdct", "max_bin": 64, "max_coefficients": 16, "objective":"binary:logistic"}
        dtrain = xgb.DMatrix(X, y)
        bst = xgb.train(param, dtrain, num_rounds, evals=[(dtrain,"train")])


'''
