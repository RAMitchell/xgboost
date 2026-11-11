"""Compare both third-order methods against directly enumerated SII values."""
# pylint: disable=duplicate-code
# pylint: disable=missing-function-docstring,too-many-locals

import copy
import itertools
import math

import numpy as np
import xgboost as xgb
from adapters import canonical_trees, explain, load_old, make_explainer


def brute(trees, row):
    n = len(row)

    def value(node, t, subset):
        f = t.feature[node]
        if f < 0:
            return t.value[node]
        if f in subset:
            return value(
                t.left[node] if row[f] < t.threshold[node] else t.right[node], t, subset
            )
        return sum(
            t.child_weight[c] * value(c, t, subset)
            for c in [t.left[node], t.right[node]]
        )

    game = {
        mask: sum(value(0, t, {j for j in range(n) if mask & (1 << j)}) for t in trees)
        for mask in range(1 << n)
    }
    result = {}
    for s in itertools.combinations(range(n), 3):
        smask = sum(1 << j for j in s)
        total = 0.0
        for mask in range(1 << n):
            if mask & smask:
                continue
            k = mask.bit_count()
            weight = (
                math.factorial(k) * math.factorial(n - k - 3) / math.factorial(n - 2)
            )
            delta = 0.0
            for subset in range(1 << 3):
                bits = sum(1 << s[j] for j in range(3) if subset & (1 << j))
                delta += (-1) ** (3 - subset.bit_count()) * game[mask | bits]
            total += weight * delta
        result[s] = total
    return result


def main():
    rng = np.random.default_rng(184)
    x = rng.normal(size=(256, 4)).astype(np.float32)
    x[:, 3] = rng.integers(0, 2, len(x))
    y = x[:, 0] * x[:, 1] * x[:, 2] + x[:, 3] * x[:, 0]
    b = xgb.train(
        {"max_depth": 6, "min_child_weight": 0, "reg_lambda": 0, "nthread": 1},
        xgb.DMatrix(x, label=y, nthread=1),
        2,
    )
    old = load_old()
    trees = canonical_trees(old, b)
    methods = {
        m: make_explainer(old, trees, 4, m) for m in ["quadrature", "treeshapiq"]
    }
    rows = x[:3].astype(np.float64)
    rows[0, trees[0].feature[0]] = trees[0].threshold[0]
    for row in rows:
        exact = brute(trees, row)
        for method, obj in methods.items():
            observed = explain(old, obj, row, method)
            keys = set(exact) | set(observed)
            gap = max(abs(exact.get(k, 0) - observed.get(k, 0)) for k in keys)
            assert gap < 1e-10, (method, gap, exact, observed)
            print(method, "direct-enumeration max error", gap)
    # Noncontiguous feature IDs previously exposed a shapiq interaction-label permutation.
    mapping = np.array([0, 12, 15, 45])
    mapped = copy.deepcopy(trees)
    for tree in mapped:
        mask = tree.feature >= 0
        tree.feature[mask] = mapping[tree.feature[mask]]
    methods = {
        m: make_explainer(old, mapped, 54, m) for m in ["quadrature", "treeshapiq"]
    }
    for row in rows:
        expected = {tuple(mapping[list(k)]): v for k, v in brute(trees, row).items()}
        padded = np.zeros(54)
        padded[mapping] = row
        for method, obj in methods.items():
            observed = explain(old, obj, padded, method)
            keys = set(expected) | set(observed)
            gap = max(abs(expected.get(k, 0) - observed.get(k, 0)) for k in keys)
            assert gap < 1e-10, (method, gap)
            print(method, "noncontiguous-ID oracle error", gap)
    print("PASS: third-order SII, repeated features and threshold equality")


if __name__ == "__main__":
    main()
