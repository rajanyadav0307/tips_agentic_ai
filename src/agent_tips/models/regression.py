"""Step 6: regression cross-checks for the fair comparisons.

``fe_lpm`` fits a linear probability model with stratum fixed effects
(absorbed by demeaning within strata) and standard errors clustered by
issue. With a single binary arm indicator it estimates a variance-weighted
within-stratum difference, so it should agree closely with
:func:`agent_tips.models.estimators.fair_estimate`.

``conditional_logit`` is the logistic counterpart: the arm's log-odds ratio
conditional on the stratum (statsmodels ``ConditionalLogit``).

``fe_interactions`` adds arm × issue-feature terms (RQ5). Issue-level features
are constant within strata, so only their interactions with the arm are
identified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Coef:
    """One regression coefficient with a cluster-robust 95% CI."""

    name: str
    estimate: float
    se: float
    ci_lo: float
    ci_hi: float
    p_value: float


def _demean(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract group means (works column-wise for 2-D arrays)."""
    _, idx = np.unique(groups, return_inverse=True)
    if x.ndim == 1:
        means = np.bincount(idx, weights=x) / np.bincount(idx)
        return x - means[idx]
    counts = np.bincount(idx)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = x[:, j] - (np.bincount(idx, weights=x[:, j]) / counts)[idx]
    return out


def _cluster_ols(
    y: np.ndarray, x: np.ndarray, cluster: np.ndarray, n_absorbed: int, names: Sequence[str]
) -> list[Coef]:
    """OLS on already-demeaned data with CR1 cluster-robust standard errors."""
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    _, cidx = np.unique(cluster, return_inverse=True)
    g = int(cidx.max()) + 1
    scores = np.zeros((g, x.shape[1]))
    np.add.at(scores, cidx, x * resid[:, None])
    meat = scores.T @ scores
    n, k = x.shape
    dof = n - k - n_absorbed
    adj = (g / (g - 1)) * ((n - 1) / max(dof, 1))
    vcov = adj * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(vcov), 0, None))
    tcrit = stats.t.ppf(0.975, g - 1)
    out = []
    for i, name in enumerate(names):
        t = beta[i] / se[i] if se[i] > 0 else np.nan
        p = float(2 * stats.t.sf(abs(t), g - 1)) if np.isfinite(t) else float("nan")
        out.append(
            Coef(
                name,
                float(beta[i]),
                float(se[i]),
                float(beta[i] - tcrit * se[i]),
                float(beta[i] + tcrit * se[i]),
                p,
            )
        )
    return out


def _prepare(
    df: pd.DataFrame,
    strata_cols: Sequence[str],
    arm_col: str,
    a: str,
    b: str,
    issue_col: str,
) -> pd.DataFrame:
    d = df[df[arm_col].isin([a, b])].copy()
    keys = list(dict.fromkeys([issue_col, *strata_cols]))
    d["_stratum"] = d.groupby(keys, observed=True).ngroup()
    both = d.groupby("_stratum")[arm_col].transform("nunique") == 2
    d = d[both]
    d["_arm"] = (d[arm_col] == a).astype(float)
    return d


def fe_lpm(
    df: pd.DataFrame,
    strata_cols: Sequence[str],
    arm_col: str,
    a: str,
    b: str,
    y: str,
    issue_col: str = "issue_key",
) -> Coef | None:
    """Arm effect from a fixed-effects linear probability model."""
    d = _prepare(df, strata_cols, arm_col, a, b, issue_col)
    if d.empty:
        return None
    s = d["_stratum"].to_numpy()
    yy = _demean(d[y].to_numpy(float), s)
    xx = _demean(d["_arm"].to_numpy(float), s)[:, None]
    return _cluster_ols(yy, xx, d[issue_col].to_numpy(), int(d["_stratum"].nunique()), ["arm"])[0]


def conditional_logit(
    df: pd.DataFrame,
    strata_cols: Sequence[str],
    arm_col: str,
    a: str,
    b: str,
    y: str,
    issue_col: str = "issue_key",
) -> Coef | None:
    """Arm log-odds ratio from a conditional logistic regression (model-based SE).

    Only strata with both arms and both outcomes carry information, so the
    others are dropped before fitting.
    """
    from statsmodels.discrete.conditional_models import ConditionalLogit

    d = _prepare(df, strata_cols, arm_col, a, b, issue_col)
    d = d[d.groupby("_stratum")[y].transform("nunique") == 2]
    if d.empty:
        return None
    res = ConditionalLogit(
        d[y].to_numpy(float), d[["_arm"]].to_numpy(float), groups=d["_stratum"].to_numpy()
    ).fit(disp=False)
    est, se = float(res.params[0]), float(res.bse[0])
    z = float(stats.norm.ppf(0.975))
    p = float(2 * stats.norm.sf(abs(est / se)))
    return Coef("arm", est, se, est - z * se, est + z * se, p)


def fe_interactions(
    df: pd.DataFrame,
    strata_cols: Sequence[str],
    arm_col: str,
    a: str,
    b: str,
    y: str,
    features: Sequence[str],
    issue_col: str = "issue_key",
) -> list[Coef]:
    """Arm effect plus arm × feature interactions (features must be 0/1)."""
    d = _prepare(df, strata_cols, arm_col, a, b, issue_col)
    if d.empty:
        return []
    s = d["_stratum"].to_numpy()
    cols = [d["_arm"].to_numpy(float)]
    names = ["arm"]
    for f in features:
        cols.append(d["_arm"].to_numpy(float) * d[f].astype(float).to_numpy())
        names.append(f"arm x {f}")
    x = _demean(np.column_stack(cols), s)
    yy = _demean(d[y].to_numpy(float), s)
    return _cluster_ols(yy, x, d[issue_col].to_numpy(), int(d["_stratum"].nunique()), names)
