"""Proper scoring rules, with the conventions documented in Appendix A.

Brier is the multi-class SUM over the three outcomes (range 0..2), as in
Tips8ng's /v1/record scoring (record.go scoreOf). RPS divides by r - 1 = 2.
"""
from __future__ import annotations

import math
import random

EPS = 1e-15


def log_loss(p, y) -> float:
    return -math.log(max(p[y], EPS))


def brier(p, y) -> float:
    return sum((p[k] - (1.0 if k == y else 0.0)) ** 2 for k in range(3))


def rps(p, y) -> float:
    cp = [p[0], p[0] + p[1]]
    co = [1.0 if y == 0 else 0.0, 1.0 if y <= 1 else 0.0]
    return sum((a - b) ** 2 for a, b in zip(cp, co)) / 2.0


def correct(p, y) -> float:
    return 1.0 if max(range(3), key=lambda k: p[k]) == y else 0.0


def ece(ps, ys, bins: int = 10) -> float:
    """Top-label expected calibration error, equal-width bins (lo, hi]."""
    total, n = 0.0, len(ys)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(ps) if lo < max(p) <= hi]
        if idx:
            conf = sum(max(ps[i]) for i in idx) / len(idx)
            acc = sum(correct(ps[i], ys[i]) for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return total


def mean(xs) -> float:
    return sum(xs) / len(xs)


def paired_bootstrap(a, b, n_boot: int = 1000, seed: int = 42):
    """Mean of (b - a) per match and its 95% percentile interval.

    a and b are per-match losses of two models on the SAME matches. Resampling
    matches (not models) keeps the pairing, which is what makes the interval
    narrow enough to be useful.
    """
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    means = sorted(mean([d[rng.randrange(n)] for _ in range(n)])
                   for _ in range(n_boot))
    return mean(d), (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])
