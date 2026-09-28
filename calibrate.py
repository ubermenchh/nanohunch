"""Calibration: temperature scaling and metrics"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fmt import FORMAT_VERSION, QType

T_MIN, T_MAX = 0.05, 20.0


@dataclass(frozen=True)
class Calibration:
    model_revision: str
    format_version: str
    temperature: dict[str, float]  # key f"{qtype}:{n_perms}"
    default_perms: dict[str, int]
    fitted_on: str


def _nll(X: np.ndarray, t: np.ndarray, temp: float) -> float:
    z = X / temp  # padded slots are -inf and stay -inf
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(t)), t]))


def fit_temperature(pooled_logprobs: Sequence[np.ndarray], targets: Sequence[int]) -> float:
    k = max(len(x) for x in pooled_logprobs)
    X = np.full((len(pooled_logprobs), k), -np.inf)
    for i, x in enumerate(pooled_logprobs):
        X[i, : len(x)] = x
    t = np.asarray(targets)
    grid = np.linspace(np.log(T_MIN), np.log(T_MAX), 201)
    i = int(np.argmin([_nll(X, t, np.exp(g)) for g in grid]))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = (np.sqrt(5) - 1) / 2  # golden-section search on log T inside the best grid bracket
    for _ in range(40):
        a, b = hi - r * (hi - lo), lo + r * (hi - lo)
        if _nll(X, t, np.exp(a)) < _nll(X, t, np.exp(b)):
            hi = b
        else:
            lo = a
    return float(np.clip(np.exp((lo + hi) / 2), T_MIN, T_MAX))


def apply(cal: Calibration, qtype: QType, n_perms: int, pooled: np.ndarray) -> np.ndarray:
    if cal.format_version != FORMAT_VERSION:
        raise ValueError(f"calibration is for {cal.format_version}, engine is {FORMAT_VERSION}")
    z = np.asarray(pooled, dtype=np.float64) / cal.temperature[f"{qtype}:{n_perms}"]
    z = z - z.max()
    return np.exp(z) / np.exp(z).sum()


def accuracy(probs: Sequence[np.ndarray], targets: Sequence[int]) -> float:
    return float(np.mean([int(np.argmax(p)) == t for p, t in zip(probs, targets, strict=True)]))


def ece(conf: np.ndarray, correct: np.ndarray, *, bins: int = 15, scheme: str = "width") -> float:
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if scheme == "width":
        idx = np.minimum((conf * bins).astype(int), bins - 1)  # conf == 1.0 goes in the last bin
    else:
        idx = np.empty(len(conf), dtype=int)
        for b, chunk in enumerate(np.array_split(np.argsort(conf, kind="stable"), bins)):
            idx[chunk] = b
    return float(
        sum(
            (idx == b).mean() * abs(correct[idx == b].mean() - conf[idx == b].mean())
            for b in range(bins)
            if (idx == b).any()
        )
    )


def nll(probs: Sequence[np.ndarray], targets: Sequence[int]) -> float:
    return float(np.mean([-np.log(max(p[t], 1e-12)) for p, t in zip(probs, targets, strict=True)]))


def brier(probs: Sequence[np.ndarray], targets: Sequence[int]) -> float:
    return float(np.mean([((p - np.eye(len(p))[t]) ** 2).sum() for p, t in zip(probs, targets, strict=True)]))


def flip_rate(canon_top1: Sequence[int], perm_top1: Sequence[int]) -> float:
    if len(canon_top1) != len(perm_top1):
        raise ValueError(f"lengths differ: {len(canon_top1)} vs {len(perm_top1)}")
    return float(np.mean([a != b for a, b in zip(canon_top1, perm_top1, strict=True)]))


def paired_bootstrap(
    a_correct, b_correct, *, n: int = 10_000, seed: int = 0, groups=None
) -> tuple[float, float, float]:
    a, b = np.asarray(a_correct, dtype=float), np.asarray(b_correct, dtype=float)
    delta = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(a), size=(n, len(a)))
        d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    else:
        _, g = np.unique(groups, return_inverse=True)
        diff_sum = np.bincount(g, weights=a - b)
        size = np.bincount(g).astype(float)
        pick = rng.integers(0, len(size), size=(n, len(size)))
        d = diff_sum[pick].sum(axis=1) / size[pick].sum(axis=1)
    return delta, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
