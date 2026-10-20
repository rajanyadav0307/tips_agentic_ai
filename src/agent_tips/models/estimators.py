"""Step 6: the fair-comparison estimators.

All estimators compare two groups ("arms") *within strata*. A stratum is a set
of tries that should be comparable, e.g. the same issue (and the same
evaluation day for the older AI). Strata without both arms carry no
information and are dropped.

Main estimator: the weighted average of within-stratum differences, with
weight ``n_a * n_b / (n_a + n_b)`` per stratum (the Cochran-Mantel-Haenszel
risk-difference weight). Uncertainty comes from resampling whole issues
(cluster bootstrap) and from shuffling arm labels within strata (permutation).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Estimate:
    """One estimated difference (arm a minus arm b) with its uncertainty."""

    estimate: float
    ci_lo: float
    ci_hi: float
    p_value: float
    n_strata: int
    n_issues: int
    n_tries: int
    ci90_lo: float = float("nan")
    ci90_hi: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        """Plain dictionary (for result tables)."""
        return asdict(self)


@dataclass(frozen=True)
class Strata:
    """Per-stratum counts for two arms: tries ``n`` and outcome sums ``k``."""

    issue: np.ndarray
    n_a: np.ndarray
    n_b: np.ndarray
    k_a: np.ndarray
    k_b: np.ndarray

    @property
    def weights(self) -> np.ndarray:
        """CMH risk-difference weight per stratum."""
        return np.asarray(self.n_a * self.n_b / (self.n_a + self.n_b), dtype=float)

    @property
    def diffs(self) -> np.ndarray:
        """Within-stratum difference in mean outcome (a minus b)."""
        return np.asarray(self.k_a / self.n_a - self.k_b / self.n_b, dtype=float)

    def __len__(self) -> int:
        return len(self.n_a)


def build_strata(
    df: pd.DataFrame,
    strata_cols: Sequence[str],
    arm_col: str,
    a: str,
    b: str,
    y: str,
    issue_col: str = "issue_key",
) -> Strata:
    """Count tries and sum outcomes per stratum and arm; keep strata with both arms."""
    d = df[df[arm_col].isin([a, b])]
    if d.empty:
        return Strata(*(np.array([]) for _ in range(5)))
    keys = list(dict.fromkeys([issue_col, *strata_cols]))
    g = d.groupby([*keys, arm_col], observed=True)[y].agg(["size", "sum"]).unstack(arm_col)
    g = g.fillna(0)
    if ("size", a) not in g or ("size", b) not in g:
        return Strata(*(np.array([]) for _ in range(5)))
    n_a, n_b = g[("size", a)].to_numpy(float), g[("size", b)].to_numpy(float)
    ok = (n_a > 0) & (n_b > 0)
    issue = g.index.get_level_values(issue_col).to_numpy()[ok]
    return Strata(
        issue=issue,
        n_a=n_a[ok],
        n_b=n_b[ok],
        k_a=g[("sum", a)].to_numpy(float)[ok],
        k_b=g[("sum", b)].to_numpy(float)[ok],
    )


def weighted_diff(st: Strata) -> float:
    """Weighted average of within-stratum differences."""
    if len(st) == 0:
        return float("nan")
    w = st.weights
    return float(np.sum(w * st.diffs) / np.sum(w))


def _issue_sums(st: Strata) -> tuple[np.ndarray, np.ndarray]:
    """Per-issue numerator and denominator of the weighted difference."""
    _, idx = np.unique(st.issue, return_inverse=True)
    w = st.weights
    return np.bincount(idx, weights=w * st.diffs), np.bincount(idx, weights=w)


def bootstrap(st: Strata, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Cluster-bootstrap draws of the weighted difference (resampling issues)."""
    if len(st) == 0:
        return np.full(reps, np.nan)
    num, den = _issue_sums(st)
    draws = rng.integers(0, len(num), size=(reps, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(num[draws].sum(1) / den[draws].sum(1), dtype=float)


def permutation_p(
    df: pd.DataFrame,
    strata_cols: Sequence[str],
    arm_col: str,
    a: str,
    b: str,
    y: str,
    reps: int,
    rng: np.random.Generator,
    issue_col: str = "issue_key",
) -> float:
    """Two-sided p-value from shuffling arm labels within strata.

    Answers: if the arm made no difference, how often would shuffling the
    labels give a difference at least as large as the observed one?
    """
    keys = list(dict.fromkeys([issue_col, *strata_cols]))
    d = df[df[arm_col].isin([a, b])]
    if d.empty:
        return float("nan")
    sid = d.groupby(keys, observed=True).ngroup().to_numpy()
    is_a = (d[arm_col] == a).to_numpy(float)
    yv = d[y].to_numpy(float)
    n_all, na_all = np.bincount(sid), np.bincount(sid, weights=is_a)
    keep = ((na_all > 0) & (na_all < n_all))[sid]
    if not keep.any():
        return float("nan")
    _, sid = np.unique(sid[keep], return_inverse=True)
    is_a, yv = is_a[keep], yv[keep]
    n, na = np.bincount(sid), np.bincount(sid, weights=is_a)
    nb, kt = n - na, np.bincount(sid, weights=yv)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    w = na * nb / n

    def est(ka: np.ndarray) -> float:
        return float(np.sum(w * (ka / na - (kt - ka) / nb)) / np.sum(w))

    obs = est(np.bincount(sid, weights=yv * is_a, minlength=len(n)))
    null = np.empty(reps)
    for r in range(reps):
        order = np.lexsort((rng.random(len(sid)), sid))
        s = sid[order]
        assigned = (np.arange(len(s)) - start[s]) < na[s]
        null[r] = est(np.bincount(s, weights=yv[order] * assigned, minlength=len(n)))
    return float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (reps + 1))


def fair_estimate(
    df: pd.DataFrame,
    strata_cols: Sequence[str],
    arm_col: str,
    a: str,
    b: str,
    y: str,
    rng: np.random.Generator,
    reps: int = 2000,
    perm_reps: int = 0,
    issue_col: str = "issue_key",
) -> Estimate:
    """Within-strata difference (a minus b) with 95% and 90% cluster-bootstrap CIs."""
    st = build_strata(df, strata_cols, arm_col, a, b, y, issue_col)
    est = weighted_diff(st)
    boot = bootstrap(st, reps, rng)
    lo, hi, lo90, hi90 = np.nanpercentile(boot, [2.5, 97.5, 5, 95]) if len(st) else [np.nan] * 4
    if perm_reps:
        p = permutation_p(df, strata_cols, arm_col, a, b, y, perm_reps, rng, issue_col)
    else:  # two-sided bootstrap p-value: how often the resampled sign flips (never 0)
        p = (
            float(min(1.0, 2 * (min(np.sum(boot <= 0), np.sum(boot >= 0)) + 1) / (len(boot) + 1)))
            if len(st)
            else float("nan")
        )
    return Estimate(
        estimate=est,
        ci_lo=float(lo),
        ci_hi=float(hi),
        p_value=p,
        n_strata=len(st),
        n_issues=int(len(np.unique(st.issue))) if len(st) else 0,
        n_tries=int(st.n_a.sum() + st.n_b.sum()) if len(st) else 0,
        ci90_lo=float(lo90),
        ci90_hi=float(hi90),
    )


def simple_estimate(
    df: pd.DataFrame,
    arm_col: str,
    a: str,
    b: str,
    y: str,
    rng: np.random.Generator,
    reps: int = 2000,
    issue_col: str = "issue_key",
) -> Estimate:
    """Pooled difference in means, ignoring strata (the "simple" analysis).

    The CI still resamples issues, so it is honest about clustering; only the
    comparison itself is naive.
    """
    d = df[df[arm_col].isin([a, b])]
    if d.empty or d[arm_col].nunique() < 2:
        nan = float("nan")
        return Estimate(nan, nan, nan, nan, 0, 0, 0)
    g = d.groupby([issue_col, arm_col], observed=True)[y].agg(["size", "sum"]).unstack(arm_col)
    g = g.fillna(0)
    na, nb = g[("size", a)].to_numpy(float), g[("size", b)].to_numpy(float)
    ka, kb = g[("sum", a)].to_numpy(float), g[("sum", b)].to_numpy(float)
    est = ka.sum() / na.sum() - kb.sum() / nb.sum()
    draws = rng.integers(0, len(na), size=(reps, len(na)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = ka[draws].sum(1) / na[draws].sum(1) - kb[draws].sum(1) / nb[draws].sum(1)
    lo, hi, lo90, hi90 = np.nanpercentile(boot, [2.5, 97.5, 5, 95])
    p = float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))
    return Estimate(
        float(est),
        float(lo),
        float(hi),
        p,
        0,
        len(na),
        int(na.sum() + nb.sum()),
        float(lo90),
        float(hi90),
    )


def benjamini_hochberg(p: Sequence[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values); NaNs stay NaN."""
    arr = np.asarray(p, dtype=float)
    q = np.full_like(arr, np.nan)
    ok = ~np.isnan(arr)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = arr[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    q[ok] = out
    return q
