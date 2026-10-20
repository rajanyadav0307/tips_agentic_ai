"""Step 6 (RQ6): time until a try first touches a file changed by the real fix.

A try "survives" while it has not yet touched a gold file. Tries that end
without touching one are censored at their last step. We report:

- Kaplan-Meier curves per arm (share of tries that have *not* yet found the
  right file after each step), and the median step at which they find it;
- a discrete-time hazard model: a linear probability model on try-step rows,
  "finds a gold file at this step" ~ arm + step fixed effects + stratum fixed
  effects, with standard errors clustered by issue.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from agent_tips.models.regression import Coef, _cluster_ols, _demean


def kaplan_meier(time: np.ndarray, event: np.ndarray, horizon: int) -> np.ndarray:
    """Survival curve S(t) for t = 0..horizon (share not yet found)."""
    surv = np.ones(horizon + 1)
    s = 1.0
    for t in range(horizon + 1):
        at_risk = np.sum(time >= t)
        d = np.sum((time == t) & event)
        if at_risk > 0:
            s *= 1 - d / at_risk
        surv[t] = s
    return surv


def median_time(surv: np.ndarray) -> float:
    """First step at which S(t) drops to 0.5 or below (NaN if never)."""
    below = np.nonzero(surv <= 0.5)[0]
    return float(below[0]) if len(below) else float("nan")


def person_period(df: pd.DataFrame, time_col: str, event_col: str, horizon: int) -> pd.DataFrame:
    """Expand tries into one row per step at risk (up to ``horizon``)."""
    t = np.minimum(df[time_col].to_numpy(int), horizon)
    reps = t + 1
    out = df.loc[df.index.repeat(reps)].copy()
    out["t"] = np.concatenate([np.arange(r) for r in reps])
    last = out.groupby(level=0)["t"].transform("max")
    out["event_now"] = (
        (out["t"] == last)
        & out[event_col].astype(bool)
        & (df.loc[out.index, time_col].to_numpy() <= horizon)
    ).astype(float)
    return out.reset_index(drop=True)


def hazard_effect(
    df: pd.DataFrame,
    strata_cols: Sequence[str],
    arm_col: str,
    a: str,
    b: str,
    time_col: str,
    event_col: str,
    horizon: int = 40,
    issue_col: str = "issue_key",
) -> Coef | None:
    """Arm effect on the per-step chance of finding a gold file (percentage points)."""
    d = df[df[arm_col].isin([a, b])].copy()
    keys = list(dict.fromkeys([issue_col, *strata_cols]))
    d["_stratum"] = d.groupby(keys, observed=True).ngroup()
    d = d[d.groupby("_stratum")[arm_col].transform("nunique") == 2]
    if d.empty:
        return None
    pp = person_period(d, time_col, event_col, horizon)
    # absorb stratum and step effects: demean within stratum x step cells
    cell = pp["_stratum"].astype(np.int64) * (horizon + 1) + pp["t"].astype(np.int64)
    y = _demean(pp["event_now"].to_numpy(float), cell.to_numpy())
    x = _demean((pp[arm_col] == a).to_numpy(float), cell.to_numpy())[:, None]
    return _cluster_ols(y, x, pp[issue_col].to_numpy(), int(cell.nunique()), ["arm"])[0]
