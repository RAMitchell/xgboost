"""Float64 Python traversal with identical helper paths at every point count.

Derived from the Appendix A harness; the only optimization change is extending
its Numba arithmetic helpers to arbitrary counts. Traversal remains Python.
"""
# pylint: disable=duplicate-code
# pylint: disable=consider-using-enumerate,missing-class-docstring,missing-function-docstring,too-many-arguments,too-many-locals,too-many-positional-arguments

import numpy as np
from numba import njit


@njit(cache=True)
def propagate_basis(nodes, basis, p_edge, old, old_seen, child_basis):
    alpha_edge = p_edge - 1.0
    alpha_old = old - 1.0
    for i in range(len(nodes)):
        value = basis[i] * (1.0 + alpha_edge * nodes[i])
        if old_seen and old != 1.0:
            value /= 1.0 + alpha_old * nodes[i]
        child_basis[i] = value


@njit(cache=True)
def leaf(basis, leaf_value, weight_prod, out):
    scale = leaf_value * weight_prod
    for i in range(len(basis)):
        out[i] = basis[i] * scale


@njit(cache=True)
def add(lhs, rhs):
    for i in range(len(lhs)):
        lhs[i] += rhs[i]


@njit(cache=True)
def extract_order1(nodes, weights, h_child, p_edge, p_up):
    alpha_edge = p_edge - 1.0
    alpha_up = p_up - 1.0
    acc = 0.0
    for i in range(len(nodes)):
        edge_delta = alpha_edge / (1.0 + alpha_edge * nodes[i])
        edge_delta -= alpha_up / (1.0 + alpha_up * nodes[i])
        acc += weights[i] * h_child[i] * edge_delta
    return acc


class Quadrature:
    def __init__(self, trees, n_features, counts):
        self.trees = trees
        self.n_features = n_features
        self.counts = list(counts)
        assert len(self.counts) == len(trees) and all(n >= 1 for n in counts)
        self.rules = {}
        for count in set(counts):
            nodes, weights = np.polynomial.legendre.leggauss(count)
            self.rules[count] = ((nodes + 1) / 2, weights / 2)
        nodes, weights = (
            self.rules[next(iter(self.rules))]
            if self.rules
            else (np.array([0.5]), np.array([1.0]))
        )
        one = np.ones(len(nodes), dtype=np.float64)
        tmp = np.empty_like(one)
        propagate_basis(nodes, one, 1.0, 1.0, False, tmp)
        leaf(one, 1.0, 1.0, tmp)
        add(tmp, one)
        extract_order1(nodes, weights, tmp, 1.0, 1.0)

    def explain(self, row):
        out = np.zeros(self.n_features, dtype=np.float64)
        for tree, count in zip(self.trees, self.counts):
            self.nodes, self.weights = self.rules[count]
            path_prob = np.full(self.n_features, np.nan, dtype=np.float64)
            self._dfs_numba(
                tree, row, 0, np.ones(count, dtype=np.float64), 1.0, path_prob, out
            )
        return out

    def _dfs_numba(self, tree, row, node, basis, weight_prod, path_prob, out):
        feature = int(tree.feature[node])
        h_out = np.empty(len(self.nodes), dtype=np.float64)
        if feature < 0:
            leaf(basis, tree.value[node], weight_prod, h_out)
            return h_out

        left = int(tree.left[node])
        right = int(tree.right[node])
        fvalue = row[feature]
        if tree.categories[node] is not None:
            if np.isnan(fvalue) or fvalue < 0:
                hot_child = int(tree.missing[node])
            else:
                hot_child = left if int(fvalue) in tree.categories[node] else right
        elif np.isnan(fvalue):
            hot_child = int(tree.missing[node])
        else:
            hot_child = left if fvalue < tree.threshold[node] else right

        total = np.zeros(len(self.nodes), dtype=np.float64)
        for child in (left, right):
            child_weight = tree.child_weight[child]
            satisfies = child == hot_child
            old = path_prob[feature]
            old_seen = not np.isnan(old)
            if not old_seen:
                p_edge = (1.0 / child_weight) if satisfies else 0.0
                p_up = 1.0
            elif old == 0.0:
                p_edge = 0.0
                p_up = 0.0
            else:
                p_edge = (old / child_weight) if satisfies else 0.0
                p_up = old

            child_basis = np.empty(len(self.nodes), dtype=np.float64)
            propagate_basis(
                self.nodes,
                basis,
                p_edge,
                old if old_seen else 1.0,
                old_seen,
                child_basis,
            )
            path_prob[feature] = p_edge
            h_child = self._dfs_numba(
                tree,
                row,
                child,
                child_basis,
                weight_prod * child_weight,
                path_prob,
                out,
            )
            out[feature] += extract_order1(
                self.nodes, self.weights, h_child, p_edge, p_up
            )
            path_prob[feature] = old
            add(total, h_child)
        return total
