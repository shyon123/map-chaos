"""Small, dependency-free statistics for the scorecard (Wilson, McNemar, Holm, bootstrap)."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

_Z95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials (95% by default)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be in [0, n]")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts ``b`` and ``c``."""
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: (pvalues[i], i))
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def paired_bootstrap_diff_ci(
    before: Sequence[int],
    after: Sequence[int],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(after) - mean(before) over paired 0/1 outcomes."""
    n = len(before)
    if n == 0 or n != len(after):
        raise ValueError("need equal-length, non-empty paired samples")
    rng = random.Random(seed)  # noqa: S311
    diffs = [after[i] - before[i] for i in range(n)]
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[math.floor(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, math.ceil((1 - alpha / 2) * n_boot) - 1)]
    return lo, hi
