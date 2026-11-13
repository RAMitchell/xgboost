"""Standard Gauss--Legendre quadrature for arbitrary-order SII interactions."""

# Numerical state and orchestration are kept explicit for reproduction.
# pylint: disable=missing-function-docstring,missing-class-docstring,too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches

import itertools

import numpy as np
from numba import njit

from .models import PyTree


@njit
def child_basis8(nodes, basis, p_edge, old, old_seen, child_basis):
    alpha_edge = p_edge - 1.0
    alpha_old = old - 1.0
    for i in range(8):
        value = basis[i] * (1.0 + alpha_edge * nodes[i])
        if old_seen and old != 1.0:
            value /= 1.0 + alpha_old * nodes[i]
        child_basis[i] = value


@njit
def leaf8(basis, leaf_value, weight_prod, out):
    scale = leaf_value * weight_prod
    for i in range(8):
        out[i] = basis[i] * scale


@njit
def add8(lhs, rhs):
    for i in range(8):
        lhs[i] += rhs[i]


@njit
def extract_order1_8(nodes, weights, h_child, p_edge, p_up):
    alpha_edge = p_edge - 1.0
    alpha_up = p_up - 1.0
    acc = 0.0
    for i in range(8):
        edge_delta = alpha_edge / (1.0 + alpha_edge * nodes[i])
        edge_delta -= alpha_up / (1.0 + alpha_up * nodes[i])
        acc += weights[i] * h_child[i] * edge_delta
    return acc


@njit
def extract_subset8(nodes, weights, h_child, path_prob, subset, p_edge, p_up):
    alpha_edge = p_edge - 1.0
    alpha_up = p_up - 1.0
    acc = 0.0
    for i in range(8):
        edge_delta = alpha_edge / (1.0 + alpha_edge * nodes[i])
        edge_delta -= alpha_up / (1.0 + alpha_up * nodes[i])
        gamma = 1.0
        for j in range(subset.shape[0]):
            p_partner = path_prob[subset[j]]
            if p_partner == 1.0:
                return 0.0
            alpha_partner = p_partner - 1.0
            gamma *= alpha_partner / (1.0 + alpha_partner * nodes[i])
        acc += weights[i] * h_child[i] * edge_delta * gamma
    return acc


class OrderQuadrature:
    def __init__(self, trees: list[PyTree], n_features: int, order: int, points: int):
        self.trees = trees
        self.n_features = n_features
        self.order = order
        self.nodes, self.weights = np.polynomial.legendre.leggauss(points)
        self.nodes = 0.5 * (self.nodes + 1.0)
        self.weights = 0.5 * self.weights
        self.use_numba = points == 8
        self._subset_cache: dict[
            tuple[object, ...],
            list[tuple[tuple[int, ...] | np.ndarray, tuple[int, ...]]],
        ] = {}
        if self.use_numba:
            one = np.ones(8, dtype=np.float64)
            tmp = np.empty(8, dtype=np.float64)
            child_basis8(self.nodes, one, 1.0, 1.0, False, tmp)
            leaf8(one, 1.0, 1.0, tmp)
            add8(tmp, one)
            extract_order1_8(self.nodes, self.weights, tmp, 1.0, 1.0)
            extract_subset8(
                self.nodes,
                self.weights,
                tmp,
                np.ones(max(1, n_features), dtype=np.float64),
                np.array([0], dtype=np.int64),
                1.0,
                1.0,
            )

    def explain(self, row: np.ndarray) -> dict[tuple[int, ...], float]:
        out: dict[tuple[int, ...], float] = {}
        for tree in self.trees:
            path_prob = np.full(self.n_features, np.nan, dtype=np.float64)
            active_mask = np.zeros(self.n_features, dtype=np.bool_)
            active: list[int] = []
            if self.use_numba:
                self._dfs_numba(
                    tree,
                    row,
                    0,
                    np.ones(8, dtype=np.float64),
                    1.0,
                    path_prob,
                    active_mask,
                    active,
                    out,
                )
                continue
            self._dfs(
                tree,
                row,
                0,
                np.ones_like(self.nodes),
                1.0,
                path_prob,
                active_mask,
                active,
                out,
            )
        return out

    def _dfs(
        self,
        tree: PyTree,
        row: np.ndarray,
        node: int,
        basis: np.ndarray,
        weight_prod: float,
        path_prob: np.ndarray,
        active_mask: np.ndarray,
        active: list[int],
        out: dict[tuple[int, ...], float],
    ) -> np.ndarray:
        feature = int(tree.feature[node])
        if feature < 0:
            return basis * tree.value[node] * weight_prod

        left = int(tree.left[node])
        right = int(tree.right[node])
        children = (left, right)
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

        total = np.zeros_like(self.nodes)
        for child in children:
            child_weight = tree.child_weight[child]
            satisfies = child == hot_child
            old = path_prob[feature]
            if np.isnan(old):
                p_edge = (1.0 / child_weight) if satisfies else 0.0
                p_up = 1.0
                was_active = bool(active_mask[feature])
            elif old == 0.0:
                p_edge = 0.0
                p_up = 0.0
                was_active = True
            else:
                p_edge = (old / child_weight) if satisfies else 0.0
                p_up = old
                was_active = True

            child_basis = basis * (1.0 + (p_edge - 1.0) * self.nodes)
            if not np.isnan(old) and old != 1.0:
                child_basis = child_basis / (1.0 + (old - 1.0) * self.nodes)

            path_prob[feature] = p_edge
            if not was_active:
                active_mask[feature] = True
                active.append(feature)
            h_child = self._dfs(
                tree,
                row,
                child,
                child_basis,
                weight_prod * child_weight,
                path_prob,
                active_mask,
                active,
                out,
            )
            self._extract(feature, p_edge, p_up, h_child, path_prob, active, out)
            if not was_active:
                active.pop()
                active_mask[feature] = False
            path_prob[feature] = old
            total += h_child
        return total

    def _dfs_numba(
        self,
        tree: PyTree,
        row: np.ndarray,
        node: int,
        basis: np.ndarray,
        weight_prod: float,
        path_prob: np.ndarray,
        active_mask: np.ndarray,
        active: list[int],
        out: dict[tuple[int, ...], float],
    ) -> np.ndarray:
        feature = int(tree.feature[node])
        h_out = np.empty(8, dtype=np.float64)
        if feature < 0:
            leaf8(basis, tree.value[node], weight_prod, h_out)
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

        total = np.zeros(8, dtype=np.float64)
        for child in (left, right):
            child_weight = tree.child_weight[child]
            satisfies = child == hot_child
            old = path_prob[feature]
            old_seen = not np.isnan(old)
            if not old_seen:
                p_edge = (1.0 / child_weight) if satisfies else 0.0
                p_up = 1.0
                was_active = bool(active_mask[feature])
            elif old == 0.0:
                p_edge = 0.0
                p_up = 0.0
                was_active = True
            else:
                p_edge = (old / child_weight) if satisfies else 0.0
                p_up = old
                was_active = True

            child_basis = np.empty(8, dtype=np.float64)
            child_basis8(
                self.nodes,
                basis,
                p_edge,
                old if old_seen else 1.0,
                old_seen,
                child_basis,
            )
            path_prob[feature] = p_edge
            if not was_active:
                active_mask[feature] = True
                active.append(feature)
            h_child = self._dfs_numba(
                tree,
                row,
                child,
                child_basis,
                weight_prod * child_weight,
                path_prob,
                active_mask,
                active,
                out,
            )
            self._extract_numba(feature, p_edge, p_up, h_child, path_prob, active, out)
            if not was_active:
                active.pop()
                active_mask[feature] = False
            path_prob[feature] = old
            add8(total, h_child)
        return total

    def _extract(
        self,
        feature: int,
        p_edge: float,
        p_up: float,
        h_child: np.ndarray,
        path_prob: np.ndarray,
        active: list[int],
        out: dict[tuple[int, ...], float],
    ) -> None:
        subset_data = self._subsets_for_active(feature, active)
        if not subset_data:
            return

        alpha_edge = p_edge - 1.0
        alpha_up = p_up - 1.0
        edge_delta = alpha_edge / (1.0 + alpha_edge * self.nodes)
        edge_delta -= alpha_up / (1.0 + alpha_up * self.nodes)
        base = self.weights * h_child * edge_delta
        partner_terms: dict[int, np.ndarray | None] = {}

        for subset, key in subset_data:
            gamma = np.ones_like(self.nodes)
            skip = False
            for partner in subset:
                term = partner_terms.get(partner)
                if term is None and partner not in partner_terms:
                    p_partner = path_prob[partner]
                    if p_partner == 1.0:
                        partner_terms[partner] = None
                        term = None
                    else:
                        alpha_partner = p_partner - 1.0
                        term = alpha_partner / (1.0 + alpha_partner * self.nodes)
                        partner_terms[partner] = term
                if term is None:
                    skip = True
                    break
                gamma *= term
            if skip:
                continue
            out[key] = out.get(key, 0.0) + float(np.sum(base * gamma))

    def _subsets_for_active(
        self, feature: int, active: list[int]
    ) -> list[tuple[tuple[int, ...] | np.ndarray, tuple[int, ...]]]:
        cache_key = (feature, tuple(active))
        cached = self._subset_cache.get(cache_key)
        if cached is not None:
            return cached

        partners = tuple(f for f in active if f != feature)
        if len(partners) < self.order - 1:
            cached = []
        else:
            cached = [
                (subset, tuple(sorted((feature, *subset))))
                for subset in itertools.combinations(partners, self.order - 1)
            ]
        self._subset_cache[cache_key] = cached
        return cached

    def _numba_subsets_for_active(
        self, feature: int, active: list[int]
    ) -> list[tuple[np.ndarray, tuple[int, ...]]]:
        cache_key = ("numba", feature, tuple(active))
        cached = self._subset_cache.get(cache_key)
        if cached is not None:
            return cached

        partners = tuple(f for f in active if f != feature)
        if len(partners) < self.order - 1:
            cached = []
        else:
            cached = [
                (np.asarray(subset, dtype=np.int64), tuple(sorted((feature, *subset))))
                for subset in itertools.combinations(partners, self.order - 1)
            ]
        self._subset_cache[cache_key] = cached
        return cached

    def _extract_numba(
        self,
        feature: int,
        p_edge: float,
        p_up: float,
        h_child: np.ndarray,
        path_prob: np.ndarray,
        active: list[int],
        out: dict[tuple[int, ...], float],
    ) -> None:
        subset_data = self._numba_subsets_for_active(feature, active)
        if not subset_data:
            return
        if self.order == 1:
            out[(feature,)] = out.get((feature,), 0.0) + extract_order1_8(
                self.nodes, self.weights, h_child, p_edge, p_up
            )
            return
        for subset, key in subset_data:
            delta = extract_subset8(
                self.nodes, self.weights, h_child, path_prob, subset, p_edge, p_up
            )
            if delta != 0.0:
                out[key] = out.get(key, 0.0) + delta


def dict_from_interaction_values(iv, order: int) -> dict[tuple[int, ...], float]:
    result = {}
    for key, value in iv.dict_values.items():
        key = tuple(int(k) for k in key)
        if len(key) == order:
            result[key] = float(value)
    return result
