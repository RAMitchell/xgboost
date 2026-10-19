"""GPU counterparts of the DART cache correctness tests."""
import sys

import pytest

sys.path.append("tests/python")
from test_dart_cache import (  # noqa
    run_dart_cache_endpoints,
    run_dart_cache_reload,
    run_dart_cached_margins,
)


@pytest.mark.parametrize("normalization", ["tree", "forest"])
@pytest.mark.parametrize("sample_type", ["uniform", "weighted"])
@pytest.mark.parametrize("task", ["regression", "multiclass", "ranking", "vector", "categorical"])
def test_dart_cached_margins(normalization, sample_type, task):
    run_dart_cached_margins("cuda", normalization, sample_type, task)


@pytest.mark.parametrize("rate,skip", [(0., 0.), (0., 1.), (1., 0.), (.8, .5)])
def test_dart_cache_endpoints(rate, skip):
    run_dart_cache_endpoints("cuda", rate, skip)


def test_dart_cache_reload():
    run_dart_cache_reload("cuda")
