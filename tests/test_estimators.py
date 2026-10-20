"""Estimators on synthetic data with a known true effect."""

import numpy as np
import pandas as pd
import pytest

from agent_tips.models.estimators import (
    benjamini_hochberg,
    build_strata,
    fair_estimate,
    simple_estimate,
    weighted_diff,
)
from agent_tips.models.regression import fe_interactions, fe_lpm


def make_data(
    seed: int, n_issues: int = 400, effect: float = -0.05, confound: bool = True
) -> pd.DataFrame:
    """Issues differ in difficulty. If ``confound``, easy issues get arm A more often,
    so a simple pooled comparison is biased while the within-issue one is not."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_issues):
        base = rng.uniform(0.05, 0.9)
        p_a = 0.2 + 0.6 * base if confound else 0.5
        for _ in range(int(rng.integers(4, 12))):
            arm = "A" if rng.random() < p_a else "B"
            p = np.clip(base + (effect if arm == "A" else 0.0), 0, 1)
            rows.append((f"i{i}", arm, int(rng.random() < p), base > 0.5))
    return pd.DataFrame(rows, columns=["issue_key", "arm", "y", "easy"])


def test_weighted_diff_hand_example() -> None:
    df = pd.DataFrame(
        {
            "issue_key": ["x"] * 4 + ["y"] * 2,
            "arm": ["A", "A", "B", "B", "A", "B"],
            "y": [1, 0, 1, 1, 1, 0],
        }
    )
    st = build_strata(df, [], "arm", "A", "B", "y")
    # x: 0.5 - 1.0 = -0.5, weight 1; y: 1 - 0 = 1, weight 0.5 -> (-0.5 + 0.5) / 1.5 = 0
    assert weighted_diff(st) == pytest.approx(0.0)


def test_fair_estimate_recovers_effect_where_simple_is_biased() -> None:
    df = make_data(1, n_issues=1500)
    rng = np.random.default_rng(0)
    fair = fair_estimate(df, [], "arm", "A", "B", "y", rng, reps=500)
    simple = simple_estimate(df, "arm", "A", "B", "y", rng, reps=500)
    assert fair.ci_lo < -0.05 < fair.ci_hi
    assert simple.estimate > 0  # confounding flips the sign of the simple comparison


def test_ci_coverage_is_about_95_percent() -> None:
    hits = 0
    sims = 60
    for s in range(sims):
        df = make_data(100 + s, n_issues=300)
        e = fair_estimate(df, [], "arm", "A", "B", "y", np.random.default_rng(s), reps=300)
        hits += e.ci_lo <= -0.05 <= e.ci_hi
    assert hits / sims >= 0.85


def test_permutation_p_null_and_alternative() -> None:
    rng = np.random.default_rng(0)
    null = make_data(7, n_issues=300, effect=0.0, confound=False)
    alt = make_data(8, n_issues=600, effect=-0.2, confound=False)
    p0 = fair_estimate(null, [], "arm", "A", "B", "y", rng, reps=200, perm_reps=200).p_value
    p1 = fair_estimate(alt, [], "arm", "A", "B", "y", rng, reps=200, perm_reps=200).p_value
    assert p0 > 0.01
    assert p1 < 0.01


def test_fe_lpm_agrees_with_fair_estimate() -> None:
    df = make_data(3, n_issues=800)
    fair = fair_estimate(df, [], "arm", "A", "B", "y", np.random.default_rng(0), reps=200)
    lpm = fe_lpm(df, [], "arm", "A", "B", "y")
    assert lpm is not None
    assert lpm.estimate == pytest.approx(fair.estimate, abs=0.01)
    assert lpm.ci_hi < 0  # the (negative) effect is detected


def test_fe_lpm_ci_coverage() -> None:
    hits = 0
    sims = 60
    for s in range(sims):
        lpm = fe_lpm(make_data(300 + s, n_issues=300), [], "arm", "A", "B", "y")
        assert lpm is not None
        hits += lpm.ci_lo <= -0.05 <= lpm.ci_hi
    assert hits / sims >= 0.85


def test_interaction_detects_effect_modification() -> None:
    rng = np.random.default_rng(5)
    rows = []
    for i in range(1500):
        easy = i % 2 == 0
        eff = -0.15 if easy else 0.0
        for _ in range(8):
            arm = "A" if rng.random() < 0.5 else "B"
            p = 0.5 + (eff if arm == "A" else 0)
            rows.append((f"i{i}", arm, int(rng.random() < p), easy))
    df = pd.DataFrame(rows, columns=["issue_key", "arm", "y", "easy"])
    coefs = {c.name: c for c in fe_interactions(df, [], "arm", "A", "B", "y", ["easy"])}
    assert coefs["arm x easy"].ci_hi < 0
    assert coefs["arm"].ci_lo < 0 < coefs["arm"].ci_hi


def test_benjamini_hochberg() -> None:
    q = benjamini_hochberg([0.01, 0.04, 0.03, np.nan, 0.5])
    assert q[0] == pytest.approx(0.04)
    assert np.isnan(q[3])
    assert q[4] == pytest.approx(0.5)
    assert (q[~np.isnan(q)] <= 1).all()


def test_conditional_logit_agrees_in_sign() -> None:
    from agent_tips.models.regression import conditional_logit

    df = make_data(5, n_issues=600, effect=-0.1)
    c = conditional_logit(df, [], "arm", "A", "B", "y")
    assert c is not None
    assert c.ci_hi < 0
