"""Batched first-order quadrature SHAP for numeric XGBoost trees.

Two implementations share the traversal and float64 arithmetic:
* depth-dependent Gauss--Legendre rules with runtime-sized point loops;
* eight points with constant coefficients and explicit scalar temporaries.

Set NUMBA_SLP_VECTORIZE=1 before importing Numba to reproduce the benchmark.
Outputs have shape (rows, output groups, features), without the expected-value
column. Contributions are on the model's raw-margin scale. Eight points are
exact through distinct-feature depth 16; deeper trees use an approximation.
"""

# Numerical state and orchestration are kept explicit for reproduction.
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,consider-using-enumerate

import json

import numpy as np
from numba import literal_unroll, njit, types
from numba.experimental import jitclass
from numba.typed import List

# Permit reassociation/FMA, but preserve NaNs used for missing/unseen features.
_FAST_MATH = {"reassoc", "contract"}

_FIELDS = [
    ("x", types.float64[:, :]),
    ("out", types.float64[:, :, ::1]),
    ("basis", types.float64[::1]),
    ("h", types.float64[::1]),
    ("path", types.float64[::1]),
    ("old", types.float64[::1]),
    ("hot", types.boolean[::1]),
    ("feature", types.int64[::1]),
    ("left", types.int64[::1]),
    ("right", types.int64[::1]),
    ("missing", types.int64[::1]),
    ("threshold", types.float64[::1]),
    ("weight", types.float64[::1]),
    ("value", types.float64[::1]),
    ("nodes", types.float64[::1]),
    ("weights", types.float64[::1]),
    ("B", types.int64),
    ("F", types.int64),
    ("stride", types.int64),
    ("n", types.int64),
    ("group", types.int64),
]


@jitclass(_FIELDS)
class _Workspace:
    def __init__(self, x, groups, depth, stride):
        # Row-major point state; depth scratch is reused during traversal.
        self.x = x
        self.B = x.shape[0]
        self.F = x.shape[1]
        self.stride = stride
        self.n = stride
        self.group = 0
        self.out = np.zeros((self.B, groups, self.F))
        self.path = np.full(self.F * self.B, np.nan)
        self.basis = np.empty((depth + 1) * self.B * stride)
        self.h = np.empty((depth + 1) * self.B * stride)
        self.old = np.empty((depth + 1) * self.B)
        self.hot = np.empty((depth + 1) * self.B, np.bool_)
        self.feature = np.empty(0, np.int64)
        self.left = np.empty(0, np.int64)
        self.right = np.empty(0, np.int64)
        self.missing = np.empty(0, np.int64)
        self.threshold = np.empty(0)
        self.weight = np.empty(0)
        self.value = np.empty(0)
        self.nodes = np.empty(0)
        self.weights = np.empty(0)


@njit(inline="always", fastmath=_FAST_MATH, error_model="numpy")
def _propagate_variable(s, src, dst, p, old, seen):
    for j in range(s.n):
        v = s.basis[src + j] * (1 + (p - 1) * s.nodes[j])
        if seen and old != 1:
            v /= 1 + (old - 1) * s.nodes[j]
        s.basis[dst + j] = v


@njit(inline="always", fastmath=_FAST_MATH, error_model="numpy")
def _leaf_variable(s, offset, scale):
    for j in range(s.n):
        s.h[offset + j] = s.basis[offset + j] * scale


@njit(inline="always", fastmath=_FAST_MATH, error_model="numpy")
def _merge_variable(s, parent, child, first):
    for j in range(s.n):
        if first:
            s.h[parent + j] = 0.0 + s.h[child + j]
        else:
            s.h[parent + j] += s.h[child + j]


@njit(inline="always", fastmath=_FAST_MATH, error_model="numpy")
def _extract_variable(s, offset, p, up):
    if p == up:
        return 0.0
    a = p - 1.0
    b = up - 1.0
    total = 0.0
    n = s.n
    if up == 1.0:
        if p == 0.0:
            for j in range(n):
                total += s.weights[j] * s.h[offset + j] * s.weights[n + j]
        else:
            for j in range(n):
                total += s.weights[j] * s.h[offset + j] * (a / (1 + a * s.nodes[j]))
    elif p == 1.0:
        if up == 0.0:
            for j in range(n):
                total += s.weights[j] * s.h[offset + j] * -s.weights[n + j]
        else:
            for j in range(n):
                total += s.weights[j] * s.h[offset + j] * (-b / (1 + b * s.nodes[j]))
    elif p == 0.0:
        for j in range(n):
            total += (
                s.weights[j]
                * s.h[offset + j]
                * (s.weights[n + j] - b / (1 + b * s.nodes[j]))
            )
    elif up == 0.0:
        for j in range(n):
            total += (
                s.weights[j]
                * s.h[offset + j]
                * (a / (1 + a * s.nodes[j]) - s.weights[n + j])
            )
    else:
        for j in range(n):
            total += (
                s.weights[j]
                * s.h[offset + j]
                * (a / (1 + a * s.nodes[j]) - b / (1 + b * s.nodes[j]))
            )
    return total


# Standard eight-point Gauss--Legendre rule on [0, 1].
_NODES_8 = (
    0.019855071751231912,
    0.10166676129318664,
    0.2372337950418355,
    0.4082826787521751,
    0.5917173212478248,
    0.7627662049581645,
    0.8983332387068134,
    0.9801449282487681,
)
_WEIGHTS_8 = (
    0.05061426814518853,
    0.11119051722668721,
    0.15685332293894344,
    0.18134189168918083,
    0.18134189168918083,
    0.15685332293894344,
    0.11119051722668721,
    0.05061426814518853,
)
_ZERO_8 = (
    -1.0202572815294848,
    -1.113172659000729,
    -1.3110177056872192,
    -1.6899961587928183,
    -2.44928343043177,
    -4.215251034632957,
    -9.836056418834861,
    -50.36496531109009,
)


@njit(inline="always", fastmath=_FAST_MATH, error_model="numpy")
def _propagate8(s, src, dst, p, old, seen):
    a = p - 1.0
    v0 = s.basis[src + 0] * (1 + a * _NODES_8[0])
    v1 = s.basis[src + 1] * (1 + a * _NODES_8[1])
    v2 = s.basis[src + 2] * (1 + a * _NODES_8[2])
    v3 = s.basis[src + 3] * (1 + a * _NODES_8[3])
    v4 = s.basis[src + 4] * (1 + a * _NODES_8[4])
    v5 = s.basis[src + 5] * (1 + a * _NODES_8[5])
    v6 = s.basis[src + 6] * (1 + a * _NODES_8[6])
    v7 = s.basis[src + 7] * (1 + a * _NODES_8[7])
    if seen and old != 1.0:
        previous = old - 1.0
        v0 /= 1 + previous * _NODES_8[0]
        v1 /= 1 + previous * _NODES_8[1]
        v2 /= 1 + previous * _NODES_8[2]
        v3 /= 1 + previous * _NODES_8[3]
        v4 /= 1 + previous * _NODES_8[4]
        v5 /= 1 + previous * _NODES_8[5]
        v6 /= 1 + previous * _NODES_8[6]
        v7 /= 1 + previous * _NODES_8[7]
    s.basis[dst + 0] = v0
    s.basis[dst + 1] = v1
    s.basis[dst + 2] = v2
    s.basis[dst + 3] = v3
    s.basis[dst + 4] = v4
    s.basis[dst + 5] = v5
    s.basis[dst + 6] = v6
    s.basis[dst + 7] = v7


@njit(inline="always", fastmath=_FAST_MATH, error_model="numpy")
def _leaf8(s, offset, scale):
    v0 = s.basis[offset + 0] * scale
    v1 = s.basis[offset + 1] * scale
    v2 = s.basis[offset + 2] * scale
    v3 = s.basis[offset + 3] * scale
    v4 = s.basis[offset + 4] * scale
    v5 = s.basis[offset + 5] * scale
    v6 = s.basis[offset + 6] * scale
    v7 = s.basis[offset + 7] * scale
    s.h[offset + 0] = v0
    s.h[offset + 1] = v1
    s.h[offset + 2] = v2
    s.h[offset + 3] = v3
    s.h[offset + 4] = v4
    s.h[offset + 5] = v5
    s.h[offset + 6] = v6
    s.h[offset + 7] = v7


@njit(inline="always", fastmath=_FAST_MATH, error_model="numpy")
def _merge8(s, parent, child, first):
    if first:
        v0 = 0.0 + s.h[child + 0]
        v1 = 0.0 + s.h[child + 1]
        v2 = 0.0 + s.h[child + 2]
        v3 = 0.0 + s.h[child + 3]
        v4 = 0.0 + s.h[child + 4]
        v5 = 0.0 + s.h[child + 5]
        v6 = 0.0 + s.h[child + 6]
        v7 = 0.0 + s.h[child + 7]
    else:
        v0 = s.h[parent + 0] + s.h[child + 0]
        v1 = s.h[parent + 1] + s.h[child + 1]
        v2 = s.h[parent + 2] + s.h[child + 2]
        v3 = s.h[parent + 3] + s.h[child + 3]
        v4 = s.h[parent + 4] + s.h[child + 4]
        v5 = s.h[parent + 5] + s.h[child + 5]
        v6 = s.h[parent + 6] + s.h[child + 6]
        v7 = s.h[parent + 7] + s.h[child + 7]
    s.h[parent + 0] = v0
    s.h[parent + 1] = v1
    s.h[parent + 2] = v2
    s.h[parent + 3] = v3
    s.h[parent + 4] = v4
    s.h[parent + 5] = v5
    s.h[parent + 6] = v6
    s.h[parent + 7] = v7


@njit(inline="always", fastmath=_FAST_MATH, error_model="numpy")
def _extract8(s, offset, p, up):
    """Fixed coefficients with compiler unrolling; no handwritten expansion."""
    if p == up:
        return 0.0
    a = p - 1.0
    b = up - 1.0
    total = 0.0
    if up == 1.0 and p == 0.0:
        for j in literal_unroll(range(8)):
            total += _WEIGHTS_8[j] * s.h[offset + j] * _ZERO_8[j]
    elif up == 1.0:
        for j in literal_unroll(range(8)):
            total += _WEIGHTS_8[j] * s.h[offset + j] * (a / (1 + a * _NODES_8[j]))
    elif p == 1.0 and up == 0.0:
        for j in literal_unroll(range(8)):
            total += _WEIGHTS_8[j] * s.h[offset + j] * -_ZERO_8[j]
    elif p == 1.0:
        for j in literal_unroll(range(8)):
            total += _WEIGHTS_8[j] * s.h[offset + j] * (-b / (1 + b * _NODES_8[j]))
    elif p == 0.0:
        for j in literal_unroll(range(8)):
            total += (
                _WEIGHTS_8[j]
                * s.h[offset + j]
                * (_ZERO_8[j] - b / (1 + b * _NODES_8[j]))
            )
    elif up == 0.0:
        for j in literal_unroll(range(8)):
            total += (
                _WEIGHTS_8[j]
                * s.h[offset + j]
                * (a / (1 + a * _NODES_8[j]) - _ZERO_8[j])
            )
    else:
        for j in literal_unroll(range(8)):
            total += (
                _WEIGHTS_8[j]
                * s.h[offset + j]
                * (a / (1 + a * _NODES_8[j]) - b / (1 + b * _NODES_8[j]))
            )
    return total


def _compile_traversal(propagate, leaf, merge, extract):
    """Bind either kernel set once, keeping identical traversal logic."""

    @njit(fastmath=_FAST_MATH, error_model="numpy")
    def visit(s, node, depth, prob):
        feature = s.feature[node]
        B = s.B
        offset = depth * B * s.stride
        if feature < 0:
            scale = s.value[node] * prob
            for r in range(B):
                leaf(s, offset + r * s.stride, scale)
            return 0.0
        left = s.left[node]
        right = s.right[node]
        slot = depth * B
        pathslot = feature * B
        seen = not np.isnan(s.path[pathslot])
        for r in range(B):
            s.old[slot + r] = s.path[pathslot + r]
            f = s.x[r, feature]
            if np.isnan(f):
                s.hot[slot + r] = s.missing[node] == left
            else:
                s.hot[slot + r] = f < s.threshold[node]
        for child in (left, right):
            weight = s.weight[child]
            childoffset = (depth + 1) * B * s.stride
            for r in range(B):
                old = s.old[slot + r]
                satisfies = s.hot[slot + r] if child == left else not s.hot[slot + r]
                if not seen:
                    p = 1.0 / weight if satisfies else 0.0
                elif old == 0.0:
                    p = 0.0
                else:
                    p = old / weight if satisfies else 0.0
                propagate(
                    s, offset + r * s.stride, childoffset + r * s.stride, p, old, seen
                )
                s.path[pathslot + r] = p
            visit(s, child, depth + 1, prob * weight)
            for r in range(B):
                old = s.old[slot + r]
                up = old if seen else 1.0
                p = s.path[pathslot + r]
                delta = extract(s, childoffset + r * s.stride, p, up)
                s.out[r, s.group, feature] += delta
                merge(
                    s, offset + r * s.stride, childoffset + r * s.stride, child == left
                )
                s.path[pathslot + r] = old
        return 0.0

    return visit


_VARIABLE_VISIT = _compile_traversal(
    _propagate_variable, _leaf_variable, _merge_variable, _extract_variable
)


_FIXED8_VISIT = _compile_traversal(_propagate8, _leaf8, _merge8, _extract8)


def _compile_prediction(groups, depth, stride, visit):
    @njit(fastmath=_FAST_MATH, error_model="numpy")
    def predict(trees, rule_nodes, rule_weights, x):
        s = _Workspace(x, groups, depth, stride)
        for index in range(len(trees)):
            t = trees[index]
            s.feature, s.left, s.right, s.missing = t[:4]
            s.threshold, s.weight, s.value = t[4:7]
            s.n, s.group = t[7:9]
            s.nodes, s.weights = (rule_nodes[index], rule_weights[index])
            for r in range(s.B):
                for j in range(s.n):
                    s.basis[r * s.stride + j] = 1.0
            visit(s, 0, 0, 1.0)
        return s.out

    return predict


_INT_ARRAY = types.int64[::1]
_FLOAT_ARRAY = types.float64[::1]
_TREE_TYPE = types.Tuple((_INT_ARRAY,) * 4 + (_FLOAT_ARRAY,) * 3 + (types.int64,) * 3)


class QuadratureSHAPNumba:
    """Numeric, scalar-leaf gbtree models; multiclass groups are supported.

    ``fixed_eight=False`` uses ceil(min(raw tree depth, feature count) / 2)
    points per tree, with a minimum of one. This matches the depth-dependent
    comparison; it is not the tightest distinct-feature-depth bound.
    ``fixed_eight=True`` selects the specialized eight-point kernels.

    Model conversion and quadrature setup happen once in the constructor.
    Each explain call allocates fresh scratch and output arrays.
    """

    def __init__(self, booster, *, fixed_eight=False):
        learner = json.loads(booster.save_raw(raw_format="json"))["learner"]
        gradient_booster = learner["gradient_booster"]
        if gradient_booster["name"] != "gbtree":
            raise ValueError("Only gbtree models are supported")
        params = learner["learner_model_param"]
        if int(params.get("num_target", 1)) != 1:
            raise ValueError("Vector-leaf/multi-target models are not supported")
        self.nfeatures = int(params["num_feature"])
        self.groups = max(1, int(params["num_class"]))
        self.fixed_eight = bool(fixed_eight)
        model = gradient_booster["model"]
        self.trees = List.empty_list(_TREE_TYPE)
        self.rule_nodes = List.empty_list(_FLOAT_ARRAY)
        self.rule_weights = List.empty_list(_FLOAT_ARRAY)
        maximum_depth, stride = 0, 8 if fixed_eight else 1
        self.counts = []
        for tree, group in zip(model["trees"], model["tree_info"]):
            if any(tree["split_type"]):
                raise ValueError(
                    "Categorical splits must be expanded before conversion"
                )
            if int(tree["tree_param"]["size_leaf_vector"]) != 1:
                raise ValueError("Vector leaves are not supported")
            left = np.asarray(tree["left_children"], dtype=np.int64)
            right = np.asarray(tree["right_children"], dtype=np.int64)
            if left[0] < 0:
                # A constant tree contributes only to the omitted expected value.
                continue
            feature = np.where(left < 0, -1, tree["split_indices"]).astype(np.int64)
            missing = np.where(tree["default_left"], left, right).astype(np.int64)
            # XGBoost stores split/leaf/cover values in float32. Restore those
            # values exactly before doing the attribution arithmetic in float64.
            values = np.asarray(tree["split_conditions"], dtype=np.float32).astype(
                np.float64
            )
            cover = np.asarray(tree["sum_hessian"], dtype=np.float32).astype(np.float64)
            weight = np.ones(len(left), dtype=np.float64)
            depth, stack = 0, [(0, 0)]
            while stack:
                node, level = stack.pop()
                depth = max(depth, level)
                if left[node] < 0:
                    continue
                for child in (left[node], right[node]):
                    if not (0 < cover[child] <= cover[node]) or not np.isfinite(
                        cover[node]
                    ):
                        raise ValueError(
                            "Internal nodes and children need finite positive cover"
                        )
                    weight[child] = cover[child] / cover[node]
                    stack.append((child, level + 1))
            n = 8 if fixed_eight else max(1, (min(depth, self.nfeatures) + 1) // 2)
            self.counts.append(n)
            self.trees.append(
                (
                    feature,
                    left,
                    right,
                    missing,
                    values,
                    weight,
                    values,
                    n,
                    int(group),
                    depth,
                )
            )
            nodes, weights = np.polynomial.legendre.leggauss(n)
            nodes, weights = (nodes + 1) / 2, weights / 2
            self.rule_nodes.append(nodes)
            # The second half caches the p=0 fraction, -1/(1-t).
            self.rule_weights.append(np.concatenate((weights, -1 / (1 - nodes))))
            maximum_depth, stride = max(maximum_depth, depth), max(stride, n)
        visit = _FIXED8_VISIT if fixed_eight else _VARIABLE_VISIT
        self._predict = _compile_prediction(self.groups, maximum_depth, stride, visit)

    def explain(self, x):
        """Return batched first-order contributions with no bias column."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.nfeatures:
            raise ValueError(f"Expected a matrix with {self.nfeatures} features")
        if len(x) == 0:
            return np.zeros((0, self.groups, self.nfeatures), dtype=np.float64)
        return self._predict(self.trees, self.rule_nodes, self.rule_weights, x)
