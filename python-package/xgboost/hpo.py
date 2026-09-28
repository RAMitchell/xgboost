"""Experimental sequential ask/tell hyperparameter optimization.

Training and validation are owned by the caller. This module performs no network
access, model training, class reweighting, or automatic metric selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np

from .core import Booster, DMatrix

__all__ = ["Optimizer", "Prior", "Trial", "Observation"]


def _normal_rank_scores(y: np.ndarray) -> np.ndarray:
    """Normal scores of average ranks; ties receive the same score.

    NormalDist is part of Python's standard library. NumPy is the only
    third-party dependency, including for the GP linear algebra.
    """
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    quantiles = ((np.cumsum(counts) - 0.5 * counts) / len(y))[inverse]
    inverse_cdf = NormalDist().inv_cdf
    return np.fromiter((inverse_cdf(float(p)) for p in quantiles), float, len(y))


def _seed(*items: Any) -> int:
    return int.from_bytes(
        hashlib.sha256(json.dumps(items).encode()).digest()[:4], "little"
    )


def _log_ei(mean: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    """Stable log expected improvement for minimization; frozen study formula."""
    mean, sd = np.asarray(mean, float), np.asarray(sd, float)
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(sd)) or np.any(sd < 0):
        raise ValueError("Invalid posterior moments")
    out = np.full(mean.shape, -np.inf)
    positive = sd > 0
    z = (best - mean[positive]) / sd[positive]
    v = np.empty_like(z)
    central = z >= -1
    middle = (z >= -20) & ~central
    tail = z < -20
    x = z[central]
    cdf = np.fromiter(
        (0.5 * math.erfc(-float(t) / math.sqrt(2)) for t in x), float, len(x)
    )
    v[central] = np.log(np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi) + x * cdf)
    x = z[middle]
    scaled_erfc = np.fromiter(
        (math.exp(float(t * t) / 2) * math.erfc(-float(t) / math.sqrt(2)) for t in x),
        float,
        len(x),
    )
    v[middle] = (
        -0.5 * x * x
        - 0.5 * math.log(2 * math.pi)
        + np.log1p(x * math.sqrt(math.pi / 2) * scaled_erfc)
    )
    x = z[tail]
    u = 1 / (x * x)
    series, term = np.ones_like(x), np.ones_like(x)
    for k in range(1, 11):
        term *= -(2 * k + 1) * u
        series += term
    v[tail] = (
        -0.5 * x * x - 0.5 * math.log(2 * math.pi) - 2 * np.log(-x) + np.log(series)
    )
    out[positive] = np.log(sd[positive]) + v
    deterministic = ~positive & (best > mean)
    out[deterministic] = np.log(best - mean[deterministic])
    return out


# Versioned encoding shared with the public prior-reproduction repository.
_FEATURES = [
    "log_learning_rate",
    "depth_active",
    "depth",
    "leaves_active",
    "log_leaves",
    "lossguide",
    "log_min_child_weight",
    "subsample",
    "column_tree",
    "column_level",
    "column_node",
    "lambda_active",
    "log_lambda",
    "alpha_active",
    "log_alpha",
    "gamma_active",
    "log_gamma",
    "log_max_bin",
]
_PARAMETERS = {
    "learning_rate",
    "max_depth",
    "max_leaves",
    "grow_policy",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "colsample_bylevel",
    "colsample_bynode",
    "reg_lambda",
    "reg_alpha",
    "gamma",
    "max_bin",
}


def _canonical(params: Mapping[str, Any]) -> str:
    if not isinstance(params, Mapping) or not params:
        raise ValueError("Each candidate must be a nonempty parameter mapping")
    result = {}
    for name, value in params.items():
        if not isinstance(name, str):
            raise ValueError("Parameter names must be strings")
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError("Parameter values must be JSON scalars")
        if isinstance(value, float) and np.isfinite(value) and value.is_integer():
            value = int(value)
        result[name] = value
    try:
        return json.dumps(result, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate parameters must be finite JSON scalars") from exc


def _candidate_key(candidates: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps([_canonical(c) for c in candidates]).encode()
    ).hexdigest()


def _encode(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    def number(
        p: Mapping[str, Any],
        name: str,
        low: float,
        high: float,
        integer: bool = False,
        zero: bool = False,
    ) -> float:
        v = p[name]
        if isinstance(v, bool) or not isinstance(
            v, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name} must be numeric")
        v = float(v)
        if not np.isfinite(v) or (integer and v != int(v)):
            raise ValueError(f"Invalid {name}: {v}")
        if not ((zero and v == 0) or low <= v <= high):
            raise ValueError(
                f"{name} is outside the historical prior's supported range"
            )
        return v

    rows = []
    for p in candidates:
        if set(p) != _PARAMETERS:
            raise ValueError(
                "The default encoding requires exactly the documented 13 parameters"
            )
        eta = number(p, "learning_rate", 0.01, 0.3)
        depth = number(p, "max_depth", 2, 12, integer=True, zero=True)
        leaves = number(p, "max_leaves", 8, 256, integer=True, zero=True)
        if depth == leaves == 0:
            raise ValueError("At least one tree-size bound must be active")
        if p["grow_policy"] not in ("depthwise", "lossguide"):
            raise ValueError("Unsupported grow_policy")
        child = number(p, "min_child_weight", 1e-5, 100)
        sub = number(p, "subsample", 0.5, 1)
        cols = [
            number(p, "colsample_by" + n, 0.5, 1) for n in ("tree", "level", "node")
        ]
        if sum(v < 1 for v in cols) > 1:
            raise ValueError(
                "Historical prior supports at most one active column-sampling mode"
            )
        bins = number(p, "max_bin", 128, 512, integer=True)
        if bins not in (128, 256, 512):
            raise ValueError("Historical prior supports max_bin in {128, 256, 512}")
        row = [
            np.log(eta / 0.01) / np.log(30),
            float(depth > 0),
            (depth - 2) / 10 if depth else 0.0,
            float(leaves > 0),
            np.log(leaves / 8) / np.log(32) if leaves else 0.0,
            float(p["grow_policy"] == "lossguide"),
            np.log(child / 1e-5) / np.log(1e7),
            (sub - 0.5) / 0.5,
        ]
        row.extend((1 - v) / 0.5 for v in cols)
        for name, high in [("reg_lambda", 100), ("reg_alpha", 10), ("gamma", 10)]:
            v = number(p, name, 1e-5, high, zero=True)
            row.extend(
                [float(v > 0), np.log(v / 1e-5) / np.log(high / 1e-5) if v else 0.0]
            )
        row.append(np.log2(bins / 128) / 2)
        rows.append(row)
    return np.clip(np.asarray(rows, dtype=float), 0, 1)


def _kernel(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # Matérn 5/2 with one learned length scale per encoded coordinate.
    r = np.sqrt(
        5 * np.sum(((x[:, None] - x[None, :]) / np.exp(theta[:18])) ** 2, axis=2)
    )
    return np.exp(theta[18]) * (1 + r + r * r / 3) * np.exp(-r)


class Prior:
    """Gaussian prior aligned with an ordered, finite candidate list.

    Mean values use normal-rank loss units: lower is better, including when the
    caller maximizes its raw metric. Covariance describes latent uncertainty;
    noise_variance describes observation noise. Arrays are defensively copied.

    Parameters
    ----------
    mean : array-like of shape (n_candidates,)
        Prior expected normal-rank loss.
    covariance : array-like of shape (n_candidates, n_candidates)
        Symmetric positive definite latent covariance, including numerical jitter.
    noise_variance : float or array-like of shape (n_candidates,)
        Nonnegative observation variance.
    """

    def __init__(self, mean: Any, covariance: Any, noise_variance: Any) -> None:
        self.mean = np.array(mean, dtype=float, copy=True)
        self.covariance = np.array(covariance, dtype=float, copy=True)
        n = self.mean.size
        if self.mean.shape != (n,) or n < 1 or self.covariance.shape != (n, n):
            raise ValueError("Expected a nonempty mean vector and matching covariance")
        try:
            self.noise_variance = (
                np.broadcast_to(noise_variance, (n,)).astype(float).copy()
            )
        except ValueError as exc:
            raise ValueError("Noise must be scalar or one value per candidate") from exc
        if not all(
            np.isfinite(a).all()
            for a in (self.mean, self.covariance, self.noise_variance)
        ):
            raise ValueError("Prior arrays must be finite")
        if np.any(self.noise_variance < 0) or not np.allclose(
            self.covariance, self.covariance.T, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("Invalid noise variance or asymmetric covariance")
        self.covariance = (self.covariance + self.covariance.T) / 2
        try:
            np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Prior covariance must be positive definite (include jitter)"
            ) from exc
        self._candidate_key: str | None = None

    @classmethod
    def uninformative(cls, candidates: Sequence[Mapping[str, Any]]) -> Prior:
        """Zero-mean, fixed-kernel prior over the documented default parameter space.

        Uses the research no-history baseline: unit length scales/amplitude,
        observation variance 0.1, and diagonal jitter 1e-10.
        """
        x = _encode(candidates)
        if len(x) == 0:
            raise ValueError("At least one candidate is required")
        prior = cls(
            np.zeros(len(x)), _kernel(x, np.zeros(20)) + 1e-10 * np.eye(len(x)), 0.1
        )
        prior._candidate_key = _candidate_key(candidates)
        return prior

    @classmethod
    def load(
        cls, directory: str | Path, *, candidates: Sequence[Mapping[str, Any]]
    ) -> Prior:
        """Load a local versioned historical artifact and project it onto candidates.

        No download is performed. This loader supports encoding xgboost-hpo-v1
        only; configurations outside its parameter ranges are rejected. The
        artifact is trained on unweighted classification/regression workloads.
        Checksums detect corruption, not an untrusted artifact's authenticity.
        """
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("encoding") != "xgboost-hpo-v1"
            or manifest.get("direction") != "minimize"
        ):
            raise ValueError("Unsupported historical prior manifest")
        for name in ("model.json", "mean.ubj", "scale.ubj"):
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest[
                "files"
            ].get(name):
                raise ValueError(f"Historical prior checksum mismatch: {name}")
        model = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        theta = np.asarray(model["logtheta"], dtype=float)
        if (
            model.get("features") != _FEATURES
            or theta.shape != (20,)
            or not np.isfinite(theta).all()
            or np.any(np.abs(theta) > 30)
        ):
            raise ValueError("Unsupported historical prior encoding/kernel")
        x = _encode(candidates)
        if len(x) == 0:
            raise ValueError("At least one candidate is required")
        dm = DMatrix(x, nthread=1)
        predictions = []
        for name in ("mean", "scale"):
            booster = Booster(params={"nthread": 1})
            booster.load_model(directory / f"{name}.ubj")
            if booster.num_features() != 18:
                raise ValueError("Historical booster has the wrong feature count")
            predictions.append(booster.predict(dm).astype(float))
        mean, log_scale = predictions
        scale = np.maximum(0.05, np.exp(np.clip(log_scale, -12, 12) / 2))
        covariance = _kernel(x, theta) * scale[:, None] * scale[
            None, :
        ] + 1e-10 * np.eye(len(x))
        prior = cls(mean, covariance, scale**2 * np.exp(theta[-1]))
        prior._candidate_key = _candidate_key(candidates)
        return prior


@dataclass(frozen=True)
class Trial:
    """An optimizer-issued evaluation request. ``params`` returns a fresh dictionary."""

    number: int
    candidate_id: int
    _params_json: str = field(repr=False)
    _owner: str = field(repr=False)

    @property
    def params(self) -> dict[str, Any]:
        """XGBoost parameter configuration to evaluate."""
        return json.loads(self._params_json)


@dataclass(frozen=True)
class Observation:
    """Completed or failed trial. Failed trials have no numerical loss."""

    trial: Trial
    status: Literal["complete", "failed"]
    loss: float | None
    message: str | None = None


class Optimizer:
    """Experimental sequential, finite-pool Gaussian-process optimizer.

    The first two successful evaluations use prior Thompson proposals. Later
    proposals maximize expected latent improvement over the incumbent, with
    observations transformed to normal-rank scores. One observation, or all-tied
    observations, leaves the posterior unchanged. Failed candidates are excluded
    without entering the posterior. No additional random-exploration fraction is
    used. Caller-owned evaluations may use any scalar metric; the historical prior's
    evidence is limited to the documented unweighted training regime.

    Parameters
    ----------
    candidates : sequence of parameter mappings
        Distinct configurations in a fixed order. Candidates are copied. With a
        custom array-based Prior, parameter mappings can use an arbitrary space.
    prior : Prior or None
        Candidate-aligned prior. None uses the fixed-kernel no-history baseline
        in the default 13-parameter space. No prior is downloaded automatically.
    direction : {"minimize", "maximize"}
        Direction of the raw scalar reported to tell().
    random_state : int
        Nonnegative seed controlling prior draws and acquisition ties.
    """

    def __init__(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        prior: Prior | None = None,
        direction: Literal["minimize", "maximize"] = "minimize",
        random_state: int = 0,
    ) -> None:
        if direction not in ("minimize", "maximize"):
            raise ValueError("direction must be 'minimize' or 'maximize'")
        if (
            isinstance(random_state, bool)
            or not isinstance(random_state, (int, np.integer))
            or random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer")
        self._candidates = tuple(_canonical(c) for c in candidates)
        if not self._candidates or len(set(self._candidates)) != len(self._candidates):
            raise ValueError("Candidates must be nonempty and distinct")
        params = [json.loads(c) for c in self._candidates]
        if prior is None:
            prior = Prior.uninformative(params)
        if not isinstance(prior, Prior):
            raise TypeError("prior must be a Prior or None")
        if prior._candidate_key is not None and prior._candidate_key != _candidate_key(
            params
        ):
            raise ValueError(
                "Prior was projected onto different candidates or a different order"
            )
        self._prior = Prior(prior.mean, prior.covariance, prior.noise_variance)
        if len(self._prior.mean) != len(params):
            raise ValueError("Prior and candidate counts differ")
        self._cholesky = np.linalg.cholesky(self._prior.covariance)
        self._direction = direction
        self._random_state = int(random_state)
        self._rng = np.random.default_rng(
            _seed("fewshot-prior-start-v1", self.random_state)
        )
        self._owner = uuid.uuid4().hex
        self._history: list[Observation] = []
        self._pending: Trial | None = None

    @property
    def direction(self) -> Literal["minimize", "maximize"]:
        """Optimization direction fixed at construction."""
        return self._direction

    @property
    def random_state(self) -> int:
        """Seed fixed at construction."""
        return self._random_state

    @property
    def history(self) -> tuple[Observation, ...]:
        """All reported trials, including failures, in reporting order."""
        return tuple(self._history)

    @property
    def best_trial(self) -> Trial | None:
        """Best successfully evaluated trial, or None before any success."""
        completed = [o for o in self._history if o.status == "complete"]
        if not completed:
            return None
        sign = 1 if self.direction == "minimize" else -1
        return min(completed, key=lambda o: sign * cast(float, o.loss)).trial

    def _scores(
        self, available: np.ndarray, completed: list[Observation]
    ) -> np.ndarray:
        ids = np.array([o.trial.candidate_id for o in completed])
        losses = np.array([o.loss for o in completed], dtype=float)
        if self.direction == "maximize":
            losses = -losses
        best = ids[np.argmin(losses)]
        mean = self._prior.mean.copy()
        k = self._prior.covariance
        variance = k[best, best] + np.diag(k) - 2 * k[:, best]
        if np.any(losses != losses[0]):
            observed = k[np.ix_(ids, ids)] + np.diag(self._prior.noise_variance[ids])
            chol = np.linalg.cholesky(observed)
            rhs = np.column_stack(
                (_normal_rank_scores(losses) - mean[ids], k[:, ids].T)
            )
            solved = np.linalg.solve(chol, rhs)
            residual, cross = solved[:, 0], solved[:, 1:]
            mean += cross.T @ residual
            difference = cross[:, best, None] - cross
            variance -= np.sum(difference**2, axis=0)
        tolerance = 1e-7 * max(1.0, float(np.diag(k).max()))
        if variance.min() < -tolerance:
            raise ArithmeticError("Materially negative posterior variance")
        return _log_ei(
            mean[available] - mean[best],
            np.sqrt(np.maximum(variance[available], 0)),
            0.0,
        )

    def ask(self) -> Trial:
        """Return the next trial, or the same pending trial until tell() is called.

        Raises StopIteration when every candidate has been evaluated or failed.
        This is a sequential API; repeated ask() does not allocate parallel work.
        """
        if self._pending is not None:
            return self._pending
        available = np.setdiff1d(
            np.arange(len(self._candidates)),
            [o.trial.candidate_id for o in self._history],
        )
        if not len(available):
            raise StopIteration("Candidate pool exhausted")
        completed = [o for o in self._history if o.status == "complete"]
        if len(completed) < 2:
            draw = self._prior.mean + self._cholesky @ self._rng.standard_normal(
                len(self._candidates)
            )
            chosen = available[np.argmin(draw[available])]
        else:
            scores = self._scores(available, completed)
            rng = np.random.default_rng(
                _seed("fewshot-choice-v1", self.random_state, len(self._history))
            )
            chosen = rng.choice(available[scores == scores.max()])
        self._pending = Trial(
            len(self._history), int(chosen), self._candidates[chosen], self._owner
        )
        return self._pending

    def tell(
        self,
        trial: Trial,
        loss: float | None = None,
        *,
        status: Literal["complete", "failed"] = "complete",
        message: str | None = None,
    ) -> None:
        """Report the pending trial's scalar metric or explicit failure.

        For status='failed', loss must be None. Failed candidates are never
        proposed again and do not receive fabricated losses. A caller wishing to
        use a timeout penalty must explicitly report it as a completed objective.
        """
        if (
            not isinstance(trial, Trial)
            or self._pending is None
            or trial != self._pending
        ):
            raise ValueError("tell() must match this optimizer's pending trial")
        if status not in ("complete", "failed"):
            raise ValueError("status must be 'complete' or 'failed'")
        if message is not None and not isinstance(message, str):
            raise ValueError("message must be a string or None")
        if status == "complete":
            if isinstance(loss, (bool, np.bool_)) or not isinstance(
                loss, (int, float, np.integer, np.floating)
            ):
                raise ValueError("A completed trial requires a finite scalar loss")
            loss = float(loss)
            if not np.isfinite(loss):
                raise ValueError("A completed trial requires a finite scalar loss")
        elif loss is not None:
            raise ValueError("A failed trial must not have a numerical loss")
        self._history.append(Observation(trial, status, loss, message))
        self._pending = None

    def save(self, path: str | Path) -> None:
        """Save JSON state, including the projected prior and any pending trial.

        State is self-contained and does not require historical model files on
        resume. Reproducibility across NumPy/BLAS versions is not guaranteed.
        """
        state = dict(
            schema_version=1,
            candidates=[json.loads(c) for c in self._candidates],
            prior=dict(
                mean=self._prior.mean.tolist(),
                covariance=self._prior.covariance.tolist(),
                noise_variance=self._prior.noise_variance.tolist(),
            ),
            direction=self.direction,
            random_state=self.random_state,
            owner=self._owner,
            observations=[
                dict(
                    candidate_id=o.trial.candidate_id,
                    status=o.status,
                    loss=o.loss,
                    message=o.message,
                )
                for o in self._history
            ],
            pending=None if self._pending is None else self._pending.candidate_id,
        )
        path = Path(path)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(state, allow_nan=False), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> Optimizer:
        """Restore and validate JSON state by replaying its deterministic decisions."""
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported optimizer state version")
        optimizer = cls(
            state["candidates"],
            prior=Prior(**state["prior"]),
            direction=state["direction"],
            random_state=state["random_state"],
        )
        owner = state["owner"]
        if not isinstance(owner, str) or len(owner) != 32:
            raise ValueError("Invalid state owner")
        optimizer._owner = owner
        for observation in state["observations"]:
            trial = optimizer.ask()
            if trial.candidate_id != observation["candidate_id"]:
                raise ValueError(
                    "State replay mismatch; check state integrity "
                    "and numerical environment"
                )
            optimizer.tell(
                trial,
                observation["loss"],
                status=observation["status"],
                message=observation["message"],
            )
        if (
            state["pending"] is not None
            and optimizer.ask().candidate_id != state["pending"]
        ):
            raise ValueError("Pending trial replay mismatch")
        return optimizer
