"""Step 6 (one-try study): the answer key and the one-try copies.

- :func:`answer_key`: the many-tries answer of a comparison. It is the
  weighted average of within-group differences (Paper 1's fair estimator;
  groups are bugs, or bug × day for the older AI), with a 95% CI from
  bootstrap resamples of bugs.
- :func:`one_try_copy`: keep one random try per key bug, the way a study with
  one try per bug would see the data.
- :func:`k_try_copy`: keep *k* random tries per bug (RQ4).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agent_tips.models.estimators import bootstrap, build_strata, weighted_diff
from agent_tips.onetry.comparisons import Comparison


@dataclass(frozen=True)
class AnswerKey:
    """The many-tries answer of one comparison."""

    estimate: float
    ci_lo: float
    ci_hi: float
    key_groups: int
    key_bugs: int

    @property
    def noise_floor(self) -> float:
        """Half-width of the 95% CI: how precisely the answer key itself is known."""
        return (self.ci_hi - self.ci_lo) / 2

    @property
    def conclusion(self) -> int:
        """+1 helps, −1 hurts, 0 no clear difference (95% CI contains 0)."""
        return int(np.sign(self.ci_lo) if self.ci_lo * self.ci_hi > 0 else 0)


def seed_for(cid: str, draw: int, extra: int = 0) -> int:
    """Stable seed for one comparison and draw (same on every machine)."""
    return (zlib.crc32(cid.encode()) * 1_000 + draw * 10 + extra) % (2**32)


def _with_groups(c: Comparison, d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(group=c.groups(d).to_numpy(), t=(d[c.column] == c.a).astype(int))


def key_rows(c: Comparison, tries: pd.DataFrame) -> pd.DataFrame:
    """Return the tries of the comparison's key groups (groups where both choices occur)."""
    d = _with_groups(c, c.rows(tries))
    both = d.groupby("group").t.nunique()
    return d[d.group.isin(both[both == 2].index)]


def answer_key(c: Comparison, tries: pd.DataFrame, reps: int = 2000, seed: int = 0) -> AnswerKey:
    """Compute the many-tries answer of a comparison, with its bootstrap CI."""
    d = key_rows(c, tries)
    st = build_strata(d, ["group"], c.column, c.a, c.b, "y", issue_col="bug_id")
    est = weighted_diff(st)
    boot = bootstrap(st, reps, np.random.default_rng(seed_for(c.cid, 0, 9)))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if len(st) else (np.nan, np.nan)
    return AnswerKey(
        float(est), float(lo), float(hi), int(d.group.nunique()), int(d.bug_id.nunique())
    )


def one_try_copy(
    c: Comparison, tries: pd.DataFrame, draw: int, key_only: bool = True
) -> pd.DataFrame:
    """Keep one random try per bug: bug_id, y, t (key bugs only unless ``key_only`` is False)."""
    d = key_rows(c, tries) if key_only else _with_groups(c, c.rows(tries))
    rng = np.random.default_rng(seed_for(c.cid, draw, 1 if key_only else 2))
    d = d.sort_values(["bug_id", "try_id"]).assign(u=lambda x: rng.random(len(x)))
    first = d.sort_values(["bug_id", "u"]).groupby("bug_id").head(1)
    return first[["bug_id", "y", "t"]].sort_values("bug_id").reset_index(drop=True)


def k_try_copy(c: Comparison, tries: pd.DataFrame, k: int, draw: int) -> pd.DataFrame:
    """Keep k random tries per bug, from bugs with at least k tries of either choice (RQ4)."""
    d = _with_groups(c, c.rows(tries)).sort_values(["bug_id", "try_id"])
    n = d.groupby("bug_id").bug_id.transform("size")
    d = d[n >= k]
    rng = np.random.default_rng(seed_for(c.cid, draw, 3 + k))
    u = pd.Series(rng.random(len(d)), index=d.index)
    rank = u.groupby(d.bug_id).rank(method="first")
    return d[rank <= k]


def k_try_answer(c: Comparison, d: pd.DataFrame) -> float:
    """Compute the within-group answer on a k-try copy."""
    st = build_strata(d, ["group"], c.column, c.a, c.b, "y", issue_col="bug_id")
    return float(weighted_diff(st)) if len(st) else float("nan")
