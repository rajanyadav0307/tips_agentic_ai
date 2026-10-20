"""Step 6: near coin flips from tries with identical histories.

Two tries of the same issue that took exactly the same actions (same tool
calls and arguments) up to some step saw exactly the same situation. Where
they first differ, which action each took was decided by sampling chance.
Comparing outcomes at such forks is the closest thing to a randomized
experiment that logged data offers for decisions after the first step.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from agent_tips.models.estimators import Estimate


def fork_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """All pairs of tries per issue, with the step and the labels where they first differ.

    ``df`` needs columns ``issue_key``, ``y``, ``hashes`` (list) and ``kinds`` (list).
    """
    rows = []
    for issue, g in df.groupby("issue_key", sort=False):
        if len(g) < 2:
            continue
        hs = [list(h) for h in g["hashes"]]
        ks = [list(k) for k in g["kinds"]]
        ys = g["y"].to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                n = min(len(hs[i]), len(hs[j]))
                t = next((s for s in range(n) if hs[i][s] != hs[j][s]), None)
                if t is not None:
                    rows.append((issue, t, ks[i][t], ks[j][t], ys[i], ys[j]))
    return pd.DataFrame(rows, columns=["issue_key", "step", "k1", "k2", "y1", "y2"])


def fork_effect(
    pairs: pd.DataFrame,
    a: str,
    rng: np.random.Generator,
    b: Sequence[str] | None = None,
    reps: int = 2000,
) -> Estimate:
    """Outcome difference at forks where exactly one try does ``a``.

    The other try does anything else, or one of ``b`` if given.
    """
    one_a = (pairs.k1 == a) ^ (pairs.k2 == a)
    if b is not None:
        other = np.where(pairs.k1 == a, pairs.k2, pairs.k1)
        one_a &= pd.Series(other, index=pairs.index).isin(b)
    g = pairs[one_a]
    if g.empty:
        nan = float("nan")
        return Estimate(nan, nan, nan, nan, 0, 0, 0)
    ya = np.where(g.k1 == a, g.y1, g.y2).astype(float)
    yo = np.where(g.k1 == a, g.y2, g.y1).astype(float)
    est = float(ya.mean() - yo.mean())
    _, idx = np.unique(g.issue_key, return_inverse=True)
    sa, so, cnt = (np.bincount(idx, weights=v) for v in (ya, yo, np.ones(len(ya))))
    draws = rng.integers(0, len(cnt), size=(reps, len(cnt)))
    boot = (sa[draws].sum(1) - so[draws].sum(1)) / cnt[draws].sum(1)
    lo, hi, lo90, hi90 = np.percentile(boot, [2.5, 97.5, 5, 95])
    p = float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))
    return Estimate(
        est, float(lo), float(hi), p, len(g), len(cnt), 2 * len(g), float(lo90), float(hi90)
    )
