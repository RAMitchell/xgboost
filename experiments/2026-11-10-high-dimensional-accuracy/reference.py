"""Independent float64 leaf-path integration; no production traversal reuse."""
# pylint: disable=duplicate-code
# pylint: disable=cell-var-from-loop,consider-using-enumerate,missing-function-docstring,too-many-arguments,too-many-locals,too-many-positional-arguments,too-many-statements

import json
import math

import numpy as np
from numba import njit, prange


def flatten(booster):
    payload = json.loads(booster.save_raw(raw_format="json"))["learner"]
    model = payload["gradient_booster"]["model"]
    nfeatures = booster.num_features()
    groups = max(1, int(payload["learner_model_param"]["num_class"]))
    raw_base = json.loads(payload["learner_model_param"]["base_score"])
    base = np.atleast_1d(raw_base).astype(np.float64)
    if len(base) == 1:
        base = np.repeat(base, groups)
    if payload["objective"]["name"] == "binary:logistic":
        base = np.log(base / (1 - base))
    offsets = [0]
    features = []
    probabilities = []
    lows = []
    highs = []
    masks = []
    missing = []
    values = []
    group_ids = []
    depths = []
    masses = []
    for tree, group in zip(model["trees"], model["tree_info"]):
        cover = np.asarray(tree["sum_hessian"], dtype=np.float32).astype(np.float64)
        cond = np.asarray(tree["split_conditions"], dtype=np.float32).astype(np.float64)
        categorical = {}
        for node, start, size in zip(
            tree["categories_nodes"],
            tree["categories_segments"],
            tree["categories_sizes"],
        ):
            cats = tree["categories"][start : start + size]
            if cats and max(cats) >= 63:
                raise ValueError("Reference category bitset supports codes 0..62")
            categorical[node] = sum(1 << int(c) for c in cats)

        def walk(node, path, depth):
            left, right = tree["left_children"][node], tree["right_children"][node]
            if left == -1:
                mass = 1.0
                for feat, (prob, lo, hi, mask, miss) in sorted(path.items()):
                    features.append(feat)
                    probabilities.append(prob)
                    lows.append(lo)
                    highs.append(hi)
                    masks.append(mask)
                    missing.append(miss)
                    mass *= prob
                offsets.append(len(features))
                values.append(cond[node])
                group_ids.append(group)
                depths.append(depth)
                masses.append(mass)
                return
            feat = tree["split_indices"][node]
            for child, is_left in [(left, True), (right, False)]:
                prob, lo, hi, mask, miss = path.get(
                    feat, (1.0, -np.inf, np.inf, (1 << 63) - 1, True)
                )
                if cover[node] <= 0 or cover[child] <= 0:
                    raise ValueError("Reference requires strictly positive node covers")
                prob *= cover[child] / cover[node]
                if node in categorical:
                    mask &= (
                        ((1 << 63) - 1) ^ categorical[node]
                        if is_left
                        else categorical[node]
                    )
                elif is_left:
                    hi = min(hi, cond[node])
                else:
                    lo = max(lo, cond[node])
                miss = miss and (bool(tree["default_left"][node]) == is_left)
                child_path = dict(path)
                child_path[feat] = (prob, lo, hi, mask, miss)
                walk(child, child_path, depth + 1)

        walk(0, {}, 0)
    arrays = (
        np.array(offsets, np.int64),
        np.array(features, np.int64),
        np.array(probabilities),
        np.array(lows),
        np.array(highs),
        np.array(masks, np.uint64),
        np.array(missing, np.bool_),
        np.array(values),
        np.array(group_ids, np.int64),
    )
    unique = np.diff(arrays[0])
    masses = np.asarray(masses)
    stats = {
        "num_trees": len(model["trees"]),
        "num_leaves": len(values),
        "max_depth": max(depths),
        "max_unique_depth": int(max(unique)),
        "leaf_fraction_unique_gt16": float(np.mean(unique > 16)),
        "cover_mass_fraction_unique_gt16": float(
            masses[unique > 16].sum() / masses.sum()
        ),
        "mean_unique_depth": float(np.mean(unique)),
        "num_features": nfeatures,
        "groups": groups,
    }
    return arrays, base, stats


@njit(parallel=True, cache=True)
def integrate(
    x,
    offsets,
    features,
    probabilities,
    lows,
    highs,
    masks,
    missing,
    values,
    group_ids,
    base,
    nodes,
    weights,
):
    result = np.zeros((len(x), len(base), x.shape[1] + 1), dtype=np.float64)
    for row in prange(len(x)):
        for group in range(len(base)):
            result[row, group, -1] = base[group]
        dmax = x.shape[1]
        z = np.empty(dmax, dtype=np.float64)
        factors = np.empty(dmax, dtype=np.float64)
        for leaf in range(len(values)):
            start, end = offsets[leaf], offsets[leaf + 1]
            group = group_ids[leaf]
            mass = 1.0
            for k in range(start, end):
                v = x[row, features[k]]
                match = missing[k] if np.isnan(v) else (lows[k] <= v < highs[k])
                if not np.isnan(v) and match:
                    code = int(v)
                    # Full mask denotes a numeric feature; otherwise category membership.
                    if masks[k] != np.uint64(9223372036854775807):
                        match = 0 <= code < 63 and bool(
                            (masks[k] >> np.uint64(code)) & np.uint64(1)
                        )
                z[k - start] = 1.0 if match else 0.0
                mass *= probabilities[k]
            result[row, group, -1] += values[leaf] * mass
            for point in range(len(nodes)):
                product = 1.0
                for k in range(start, end):
                    factor = (1 - nodes[point]) * probabilities[k] + nodes[point] * z[
                        k - start
                    ]
                    factors[k - start] = factor
                    product *= factor
                for k in range(start, end):
                    result[row, group, features[k]] += (
                        values[leaf]
                        * weights[point]
                        * (z[k - start] - probabilities[k])
                        * product
                        / factors[k - start]
                    )
    return result


def rule(n, endpoint=False):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes = (nodes + 1) / 2
    weights = weights / 2
    if endpoint:
        weights = 2 * nodes * weights
        nodes = nodes * nodes
    return nodes, weights


def predict_reference(x, arrays, base, n, endpoint=False):
    return integrate(np.asarray(x, dtype=np.float32), *arrays, base, *rule(n, endpoint))


def exhaustive_reference(x, arrays, base):
    """Small-feature coalition enumeration, independent of integration."""
    offsets, features, p, lows, highs, masks, missing, values, groups = arrays
    f = x.shape[1]
    assert f <= 8
    answer = np.zeros((len(x), len(base), f + 1))
    for row in range(len(x)):
        coalition_values = {}
        for bits in range(1 << f):
            pred = base.copy()
            for leaf in range(len(values)):
                factor = values[leaf]
                for k in range(offsets[leaf], offsets[leaf + 1]):
                    feat = features[k]
                    if bits & (1 << feat):
                        v = x[row, feat]
                        match = missing[k] if np.isnan(v) else lows[k] <= v < highs[k]
                        if not np.isnan(v) and masks[k] != np.uint64((1 << 63) - 1):
                            match = match and bool(int(masks[k]) & (1 << int(v)))
                        factor *= match
                    else:
                        factor *= p[k]
                pred[groups[leaf]] += factor
            coalition_values[bits] = pred
        answer[row, :, -1] = coalition_values[0]
        for feat in range(f):
            for bits in range(1 << f):
                if bits & (1 << feat):
                    continue
                count = bits.bit_count()
                weight = (
                    math.factorial(count)
                    * math.factorial(f - count - 1)
                    / math.factorial(f)
                )
                answer[row, :, feat] += weight * (
                    coalition_values[bits | (1 << feat)] - coalition_values[bits]
                )
    return answer
