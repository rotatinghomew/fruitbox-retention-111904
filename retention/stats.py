"""Small statistical helpers with deterministic resampling."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Iterable, Sequence

import numpy as np

try:  # scipy is a declared dependency; keep a graceful fallback for docs builds
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def proportion(successes: int, trials: int) -> float:
    return successes / trials if trials else float("nan")


def cluster_bootstrap(
    rows: Sequence[dict],
    statistic: Callable[[Sequence[dict]], float],
    *,
    cluster: str,
    draws: int = 3000,
    seed: int = 20261120,
    alpha: float = 0.05,
) -> tuple[float, float, np.ndarray]:
    """Percentile bootstrap that resamples clusters (households), not rows.

    Rows from one household are dependent (shared coupon, shared member), so a
    row-level bootstrap would understate uncertainty. Clusters with a single
    row are still resampled correctly.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row[cluster]].append(row)
    keys = sorted(grouped)
    if not keys:
        return (float("nan"), float("nan"), np.empty(0))
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        picks = rng.integers(0, len(keys), size=len(keys))
        sample = [row for index in picks for row in grouped[keys[index]]]
        values[draw] = statistic(sample)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"), values)
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return (float(low), float(high), values)


def bootstrap_p_value(values: np.ndarray, null: float = 0.0) -> float:
    """Two-sided bootstrap p-value for a statistic centred on ``null``."""
    if values.size == 0:
        return float("nan")
    shifted = values - null
    extreme = np.abs(shifted) >= abs(np.mean(values) - null)
    return float((np.sum(extreme) + 1) / (values.size + 1))


def diff_in_means(rows: Iterable[dict], *, outcome: str, arm: str = "treated") -> float:
    rows = list(rows)
    treated = [r[outcome] for r in rows if r[arm]]
    control = [r[outcome] for r in rows if not r[arm]]
    if not treated or not control:
        return float("nan")
    return float(np.mean(treated) - np.mean(control))


def normal_interval(estimate: float, standard_error: float, z: float = 1.959963984540054) -> tuple[float, float]:
    return (estimate - z * standard_error, estimate + z * standard_error)


def stratified_diff(
    rows: Sequence[dict], *, outcome: str, arm: str, stratum: str
) -> tuple[float, float]:
    """Mantel-Haenszel style weighted difference and its standard error.

    Weights are the harmonic-style ``n1*n0/n`` weights. This is used as a
    sensitivity check to the unstratified difference when batch composition
    differs between arms.
    """
    numerator = 0.0
    variance = 0.0
    weights = 0.0
    for key in sorted({row[stratum] for row in rows}):
        block = [row for row in rows if row[stratum] == key]
        treated = [row[outcome] for row in block if row[arm]]
        control = [row[outcome] for row in block if not row[arm]]
        if not treated or not control:
            continue
        n1, n0 = len(treated), len(control)
        a1, a0 = float(np.sum(treated)), float(np.sum(control))
        n = n1 + n0
        weight = n1 * n0 / n
        numerator += weight * (a1 / n1 - a0 / n0)
        # variance of the weighted rate difference within the stratum
        p1, p0 = a1 / n1, a0 / n0
        variance += weight * weight * (p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        weights += weight
    if weights == 0:
        return (float("nan"), float("nan"))
    return (numerator / weights, math.sqrt(variance) / weights)


def heterogeneity_test(weekly: Sequence[tuple[int, int]]) -> tuple[float, float]:
    """Chi-square test that several binomial proportions are equal."""
    weekly = [(k, n) for k, n in weekly if n > 0]
    if len(weekly) < 2:
        return (float("nan"), float("nan"))
    total_k = sum(k for k, _ in weekly)
    total_n = sum(n for _, n in weekly)
    p = total_k / total_n
    if p <= 0 or p >= 1:
        return (float("nan"), float("nan"))
    statistic = sum((k - n * p) ** 2 / (n * p * (1 - p)) for k, n in weekly)
    df = len(weekly) - 1
    if _scipy_stats is None:  # pragma: no cover
        return (statistic, float("nan"))
    return (statistic, float(_scipy_stats.chi2.sf(statistic, df)))


def trend_test(weekly: Sequence[tuple[int, int, int]]) -> tuple[float, float]:
    """Cochran-Armitage score test for a linear trend in proportions.

    ``weekly`` items are ``(week_index, successes, trials)``.
    """
    weekly = [(w, k, n) for w, k, n in weekly if n > 0]
    if len(weekly) < 2:
        return (float("nan"), float("nan"))
    total_k = sum(k for _, k, _ in weekly)
    total_n = sum(n for _, _, n in weekly)
    p = total_k / total_n
    if p <= 0 or p >= 1:
        return (float("nan"), float("nan"))
    weights = [w for w, _, _ in weekly]
    mean_w = sum(w * n for w, _, n in weekly) / total_n
    centred = [w - mean_w for w in weights]
    score = sum(c * (k - n * p) for c, (_, k, n) in zip(centred, weekly))
    variance = p * (1 - p) * sum(n * c * c for c, (_, _, n) in zip(centred, weekly))
    if variance <= 0:
        return (float("nan"), float("nan"))
    z = score / math.sqrt(variance)
    if _scipy_stats is None:  # pragma: no cover
        return (z, float("nan"))
    return (z, float(2 * _scipy_stats.norm.sf(abs(z))))


def money(cents: float | int | None) -> str:
    if cents is None or (isinstance(cents, float) and math.isnan(cents)):
        return "n/a"
    return f"{cents / 100:,.2f} 元"


def pct(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def pp(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value * 100:+.{digits}f}pp"
