"""Step 6 (one-try study): the answer key and the one-try copies.

- :func:`answer_key`: the many-tries answer of a comparison. It is the
  weighted average of within-group differences (Paper 1's fair estimator;
  groups are bugs, or bug × day for the older AI), with a 95% CI from
  bootstrap resamples of bugs.
- :func:`one_try_copy`: keep one random try per key bug, the way a study with
  one try per bug would see the data.
- :func:`k_try_copy`: keep *k* random tries per bug (RQ4).

Amendment A8 adds:

- :func:`key_parts` / :func:`aligned_key`: the answer key with the bug
  weights each method actually uses (``cmh`` = the frozen key; ``dml`` = one
  copy per bug weighted by p(1−p), what partialling-out targets; ``ate`` =
  every bug equal, what AIPW targets);
- :func:`own_loo`: the studied AI's own fix rate on the bug from its *other*
  tries (the "oracle" rung, many-tries datasets only);
- :func:`realistic_other`: other AIs' rates from one random try per other
  setup, as a real study would have them.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from agent_tips.models.estimators import bootstrap, build_strata, weighted_diff
from agent_tips.onetry.comparisons import Comparison
from agent_tips.onetry.features import other_ai_rates

WEIGHTINGS = ("cmh", "dml", "ate")


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


def key_parts(c: Comparison, tries: pd.DataFrame, weighting: str = "cmh") -> pd.DataFrame:
    """Per key bug: numerator and denominator of the weighted within-group difference.

    Group weights: ``cmh`` n_a·n_b/n_g (the frozen key); ``dml`` the same divided
    by the bug's key tries (the drawn try comes from group g with chance n_g/N,
    times p(1−p)); ``ate`` n_g/N (each bug counts once). The answer is
    Σ num / Σ den over bugs.
    """
    st = build_strata(key_rows(c, tries), ["group"], c.column, c.a, c.b, "y", issue_col="bug_id")
    n_g = st.n_a + st.n_b
    bug = pd.Series(st.issue)
    n_bug = pd.Series(n_g).groupby(bug).transform("sum").to_numpy(float)
    w = {"cmh": st.weights, "dml": st.weights / n_bug, "ate": n_g / n_bug}[weighting]
    per = pd.DataFrame({"bug_id": st.issue, "num": w * st.diffs, "den": w})
    return per.groupby("bug_id", sort=True)[["num", "den"]].sum().reset_index()


def aligned_key(
    c: Comparison, tries: pd.DataFrame, weighting: str, reps: int = 2000
) -> tuple[float, float, float]:
    """Answer key with the given bug weights: estimate and 95% bootstrap bounds (bugs resampled)."""
    kp = key_parts(c, tries, weighting)
    if kp.empty:
        return float("nan"), float("nan"), float("nan")
    num, den = kp.num.to_numpy(), kp.den.to_numpy()
    rng = np.random.default_rng(seed_for(c.cid, 0, 9))
    idx = rng.integers(0, len(num), size=(reps, len(num)))
    boot = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def one_try_copy(
    c: Comparison, tries: pd.DataFrame, draw: int, key_only: bool = True
) -> pd.DataFrame:
    """Keep one random try per bug: bug_id, try_id, y, t (key bugs only unless ``key_only``)."""
    d = key_rows(c, tries) if key_only else _with_groups(c, c.rows(tries))
    rng = np.random.default_rng(seed_for(c.cid, draw, 1 if key_only else 2))
    d = d.sort_values(["bug_id", "try_id"]).assign(u=lambda x: rng.random(len(x)))
    first = d.sort_values(["bug_id", "u"]).groupby("bug_id").head(1)
    return first[["bug_id", "try_id", "y", "t"]].sort_values("bug_id").reset_index(drop=True)


def own_loo(c: Comparison, tries: pd.DataFrame, copy: pd.DataFrame) -> pd.DataFrame:
    """Return the studied AI's fix rate on each bug from its other tries (drawn try left out).

    Uses every try of the setup on the bug, whatever its choice. Not available
    to a real one-try study: it is the "oracle" rung (A8), a best case.
    """
    own = tries[(tries.setup == c.setup) & tries.bug_id.isin(copy.bug_id)]
    g = own.groupby("bug_id").y.agg(["sum", "size"])
    d = copy[["bug_id", "y"]].join(g, on="bug_id")
    n = d["size"] - 1
    rate = (d["sum"] - d.y) / n.where(n > 0)
    return pd.DataFrame(
        {
            "bug_id": d.bug_id,
            "own_rate": rate.fillna(-1.0).to_numpy(),
            "n_own": n.to_numpy(float),
            "own_missing": rate.isna().astype(float).to_numpy(),
        }
    )


def realistic_other(c: Comparison, tries: pd.DataFrame, bugs: pd.Index, draw: int) -> pd.DataFrame:
    """Other AIs' rates on each bug from one random try per other setup (A8)."""
    o = tries[(tries.setup != c.setup) & tries.bug_id.isin(bugs)].sort_values("try_id")
    rng = np.random.default_rng(seed_for(c.cid, draw, 40))
    o = o.assign(u=rng.random(len(o))).sort_values(["bug_id", "setup", "u"])
    one = o.groupby(["bug_id", "setup"]).head(1)
    return other_ai_rates(one, c.setup, bugs).reset_index()


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
