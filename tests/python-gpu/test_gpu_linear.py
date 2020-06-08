import sys
from hypothesis import strategies, given, settings, assume
import xgboost as xgb
sys.path.append("tests/python")
import testing as tm


parameter_strategy = strategies.fixed_dictionaries({
    'booster': strategies.just('gblinear'),
    'eta': strategies.floats(0.01, 0.25),
    'tolerance': strategies.floats(1e-5, 1e-2),
    'nthread': strategies.integers(1, 4),
    'alpha': strategies.floats(1e-5, 2.0),
    'lambda': strategies.floats(1e-5, 2.0),
    'feature_selector': strategies.sampled_from(['cyclic', 'shuffle',
                                                 'greedy', 'thrifty']),
    'top_k': strategies.integers(1, 10),
})

def train_result(param, dmat, num_rounds):
    result = {}
    xgb.train(param, dmat, num_rounds, [(dmat, 'train')], verbose_eval=False,
              evals_result=result)
    return result


class TestGPULinear:
    @given(parameter_strategy, strategies.integers(10, 50),
           tm.dataset_strategy)
    @settings(deadline=2000)
    def test_gpu_coordinate(self, param, num_rounds, dataset):
        assume(len(dataset.y) > 0)
        param['updater'] = 'gpu_coord_descent'
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)['train'][dataset.metric]
        assert tm.non_increasing([result[0], result[-1]])

