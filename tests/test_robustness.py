"""Robustness checks on small synthetic data."""

import numpy as np
import pandas as pd

from agent_tips.evaluation.robustness import (
    frame_hash,
    misclassification,
    tipping_point,
    undated_scenarios,
)


def nv_data(seed: int = 0, n_issues: int = 300) -> pd.DataFrame:
    """Newer-AI-like tries with no true effect of test first vs fix first."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_issues):
        base = rng.uniform(0.1, 0.9)
        for _ in range(6):
            arm = "test first" if rng.random() < 0.6 else "fix first"
            broad = "test first" if arm == "fix first" and rng.random() < 0.2 else arm
            rows.append((f"i{i}", arm, broad, float(rng.random() < base)))
    return pd.DataFrame(rows, columns=["issue_key", "order_arm", "order_arm_broad", "y"])


def test_tipping_point_reaches_threshold() -> None:
    nv = nv_data()
    tp = tipping_point(nv, harm=-0.02)
    assert tp["extra_failure_share"] > 0
    # Adding more extra failures than the tipping point only makes the estimate lower.
    tighter = tipping_point(nv, harm=-0.04)
    assert tighter["extra_failure_share"] > tp["extra_failure_share"]


def test_misclassification_full_move_equals_broad() -> None:
    nv = nv_data()
    out = misclassification(nv, np.random.default_rng(0), shares=(0.5, 1.0), draws=5)
    full = next(v for k, v in out.items() if k.startswith("100%"))
    assert full["min"] == full["max"]
    half = next(v for k, v in out.items() if k.startswith("50%"))
    assert half["min"] <= half["median"] <= half["max"]


def test_undated_bounds_bracket_the_imputation() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for i in range(60):
        for _ in range(8):
            arm = "R" if rng.random() < 0.5 else "S"
            rows.append((f"i{i}", "d1", arm, float(rng.random() < 0.4), True))
        for _ in range(2):  # undated tries are always failures
            rows.append((f"i{i}", None, "R" if rng.random() < 0.5 else "S", 0.0, False))
    older = pd.DataFrame(rows, columns=["issue_key", "batch_day", "first_move_arm", "y", "dated"])
    out = undated_scenarios(older, "batch_day", np.random.default_rng(0), reps=20)
    worst = out["added back, worst case for the finding"]
    best = out["added back, best case for the finding"]
    assert best <= out["added back like other failures (median of draws)"] <= worst


def test_frame_hash_ignores_tiny_float_noise() -> None:
    a = pd.DataFrame({"x": [0.1 + 0.2], "s": ["k"]})
    b = pd.DataFrame({"x": [0.3], "s": ["k"]})
    assert frame_hash(a) == frame_hash(b)
    assert frame_hash(a) != frame_hash(b.assign(x=0.31))
