"""Scalar NumPy traversal specialized for numeric, finite benchmark inputs."""

# Numerical state and orchestration are kept explicit for reproduction.
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,cell-var-from-loop

import numpy as np


class ScalarNumericQuadrature:
    """Variant B: scalar traversal with NumPy quadrature vectors."""

    nodes: np.ndarray
    weights: np.ndarray

    def _dfs(self, tree, row, node, basis, weight_prod, path_prob, out):
        feature = int(tree.feature[node])
        if feature < 0:
            result = basis * tree.value[node]
            result *= weight_prod
            return result

        left = int(tree.left[node])
        right = int(tree.right[node])
        fvalue = row[feature]
        hot_child = left if fvalue < tree.threshold[node] else right

        total = np.zeros(self.nodes.size, dtype=np.float64)
        old = path_prob[feature]
        unseen = np.isnan(old)
        replace_basis = not unseen and old != 1.0
        for child in (left, right):
            child_weight = tree.child_weight[child]
            satisfies = child == hot_child
            if unseen:
                p_edge = (1.0 / child_weight) if satisfies else 0.0
                p_up = 1.0
            elif old == 0.0:
                p_edge = 0.0
                p_up = 0.0
            else:
                p_edge = (old / child_weight) if satisfies else 0.0
                p_up = old

            child_basis = (p_edge - 1.0) * self.nodes
            child_basis += 1.0
            child_basis *= basis
            if replace_basis:
                child_basis /= 1.0 + (old - 1.0) * self.nodes

            path_prob[feature] = p_edge
            h_child = self._dfs(
                tree,
                row,
                child,
                child_basis,
                weight_prod * child_weight,
                path_prob,
                out,
            )
            self._extract(feature, p_edge, p_up, h_child, out)
            path_prob[feature] = old
            total += h_child
        return total

    def _extract(self, feature, p_edge, p_up, h_child, out):
        alpha_edge = p_edge - 1.0
        alpha_up = p_up - 1.0
        edge_delta = alpha_edge / (1.0 + alpha_edge * self.nodes)
        edge_delta -= alpha_up / (1.0 + alpha_up * self.nodes)
        out[feature] += float(np.dot(self.weights, h_child * edge_delta))


class BatchedNumpyQuadrature:
    """Variant C: traverse a tree once for an entire input batch."""

    def __init__(self, trees, groups, nfeatures, counts):
        self.trees = trees
        self.groups = groups
        self.nfeatures = nfeatures
        self.counts = counts
        assert len(trees) == len(groups) == len(counts)
        self.rules = {}
        for n in set(counts):
            nodes, weights = np.polynomial.legendre.leggauss(n)
            self.rules[n] = ((nodes + 1) / 2, weights / 2)

    def explain(self, x):
        """Return first-order contributions with no bias column."""
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == 2
        if len(x) == 0:
            return np.zeros((0, max(self.groups, default=0) + 1, self.nfeatures))
        out = np.zeros((len(x), max(self.groups, default=0) + 1, self.nfeatures))
        for tree, group, n in zip(self.trees, self.groups, self.counts):
            if tree.feature[0] < 0:
                continue
            nodes, weights = self.rules[n]
            path = np.full((self.nfeatures, len(x)), np.nan)
            target = out[:, group, :]

            def visit(node, basis, prob):
                feature = int(tree.feature[node])
                if feature < 0:
                    return basis * (tree.value[node] * prob)
                assert tree.categories[node] is None, "Numeric splits are required"
                left = int(tree.left[node])
                right = int(tree.right[node])
                f = x[:, feature]
                hot = np.where(
                    np.isnan(f),
                    int(tree.missing[node]),
                    np.where(f < tree.threshold[node], left, right),
                )
                total = np.zeros_like(basis)
                for child in [left, right]:
                    old = path[feature].copy()
                    seen = not np.isnan(old[0])
                    p = np.where(
                        hot == child,
                        (old if seen else 1.0) / tree.child_weight[child],
                        0.0,
                    )
                    up = old if seen else np.ones(len(x))
                    child_basis = basis * (1 + (p[:, None] - 1) * nodes)
                    if seen:
                        child_basis /= 1 + (old[:, None] - 1) * nodes
                    path[feature] = p
                    h = visit(child, child_basis, prob * tree.child_weight[child])
                    delta = (p[:, None] - 1) / (1 + (p[:, None] - 1) * nodes) - (
                        up[:, None] - 1
                    ) / (1 + (up[:, None] - 1) * nodes)
                    target[:, feature] += np.sum(weights * h * delta, axis=1)
                    total += h
                    path[feature] = old
                return total

            visit(0, np.ones((len(x), n)), 1.0)
        return out
