"""Step 6 of the one-try study: every method on made-up data with a known answer.

Easy bugs get choice *a* more often (the trap). The report text carries no
information about difficulty; the "other AIs" and IRT columns carry it exactly
(a clear clue). So:
- the simple comparison must be fooled;
- ML on the report alone must stay fooled;
- ML with the other-AI information must find the true effect;
- the many-tries answer key must find the true effect.
"""

import numpy as np
import pandas as pd
import pytest

from agent_tips.onetry import corrections as cr
from agent_tips.onetry import simulate as sm
from agent_tips.onetry.comparisons import Comparison
from agent_tips.onetry.simulate import answer_key, k_try_answer, k_try_copy, one_try_copy

WORDS = [f"word{i}" for i in range(60)]
TEXT_DIMS = 10  # smaller than the frozen 100 to keep the tests fast; the logic is the same


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def bugs_frame(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Bug features; returns the frame and each bug's hidden difficulty."""
    rng = np.random.default_rng(seed)
    diff = rng.normal(size=n)
    d = pd.DataFrame(
        {
            "bug_id": [f"b{i:05d}" for i in range(n)],
            "log_report_chars": rng.normal(6, 1, n),
            "has_code_block": rng.integers(0, 2, n).astype(float),
            "has_stack_trace": rng.integers(0, 2, n).astype(float),
            "names_file": rng.integers(0, 2, n).astype(float),
            "repo_group": rng.choice(["r1", "r2", "other"], n),
            "language": "python",
            "task_type": "unknown",
            "report": [" ".join(rng.choice(WORDS, 25)) for _ in range(n)],
            "log_fix_files": rng.normal(1, 0.3, n),
            "log_fix_lines": rng.normal(3, 0.5, n),
            "fix_missing": 0.0,
            # other AIs' fix rate and IRT difficulty: clear clues to difficulty
            "other_rate": _sigmoid(-1.5 * diff),
            "n_other": 3.0,
            "other_missing": 0.0,
            "irt_difficulty": diff,
            "irt_missing": 0.0,
        }
    )
    return d, diff


def one_try(n: int, tau: float, seed: int, trap: float = 1.5) -> pd.DataFrame:
    """One try per bug; easy bugs choose a more often (strength ``trap``); true effect ``tau``.

    Fix chances stay between 15% and 85% before the effect, so adding ``tau``
    never passes 100% and the true average effect is exactly ``tau``.
    """
    rng = np.random.default_rng(seed + 1)
    d, diff = bugs_frame(n, seed)
    t = (rng.random(n) < _sigmoid(-trap * diff)).astype(int)
    p = 0.15 + 0.7 * _sigmoid(-1.5 * diff) + tau * t
    return d.assign(t=t, y=(rng.random(n) < p).astype(float))


@pytest.fixture(scope="module")
def null_case() -> pd.DataFrame:
    return one_try(1500, 0.0, seed=1)


def test_simple_comparison_is_fooled(null_case: pd.DataFrame) -> None:
    assert cr.simple(null_case).estimate > 0.15


def test_ml_on_report_text_alone_stays_fooled(null_case: pd.DataFrame) -> None:
    r = cr.dml(null_case, 2, seed=0, text_dims=TEXT_DIMS)
    assert r.estimate > 0.10


def test_ml_with_other_ai_information_finds_zero(null_case: pd.DataFrame) -> None:
    r = cr.dml(null_case, 5, seed=0, text_dims=TEXT_DIMS)
    assert abs(r.estimate) < 0.04
    assert r.ci_lo < 0 < r.ci_hi


def test_ml_with_other_ai_information_finds_a_real_effect() -> None:
    d = one_try(1500, 0.10, seed=2)
    for r in (cr.dml(d, 5, seed=0, text_dims=TEXT_DIMS), cr.dml(d, 6, seed=0, text_dims=TEXT_DIMS)):
        assert abs(r.estimate - 0.10) < 0.05


def test_aipw_check_finds_a_real_effect_with_fair_overlap() -> None:
    # AIPW divides by the chance of each choice, so it needs both choices to be
    # reasonably likely for every bug (milder trap than above).
    d = one_try(1500, 0.10, seed=5, trap=0.8)
    r = cr.aipw(d, 5, seed=0, text_dims=TEXT_DIMS)
    assert abs(r.estimate - 0.10) < 0.06


def test_ladder_runs_every_rung_and_check(null_case: pd.DataFrame) -> None:
    out = cr.ladder(
        null_case.head(300), seed=0, rungs=(0, 1, 2, 4, 5, 6), text_dims=TEXT_DIMS, checks=True
    )
    assert {
        "rung0",
        "rung1",
        "rung2",
        "rung4",
        "rung5",
        "rung6",
        "rung5_forest",
        "rung5_aipw",
    } <= set(out)
    assert all(np.isfinite(r.estimate) and r.se > 0 for r in out.values())


def _many_tries(n_bugs: int, tries: int, tau: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    diff = rng.normal(size=n_bugs)
    rows = []
    for i in range(n_bugs):
        for j in range(tries):
            t = rng.random() < _sigmoid(-1.5 * diff[i])
            p = min(max(_sigmoid(-1.5 * diff[i]) + tau * t, 0), 1)
            rows.append((f"b{i}", f"b{i}-{j}", "s", "a" if t else "b", float(rng.random() < p)))
    d = pd.DataFrame(rows, columns=["bug_id", "try_id", "setup", "choice", "y"])
    return d.assign(eval_day=None, order="fix first")


def test_answer_key_finds_the_true_effect_despite_the_trap() -> None:
    tries = _many_tries(3000, 6, 0.08, seed=3)
    c = Comparison("s|x", "d", "s", "a vs b", "choice", "a", "b", experiment=True)
    key = answer_key(c, tries, reps=200)
    assert key.ci_lo < 0.08 < key.ci_hi
    assert key.noise_floor < 0.03
    # and the one-try copy of the same data is fooled
    one = one_try_copy(c, tries, draw=0)
    assert len(one) == key.key_bugs
    assert one.bug_id.is_unique
    assert cr.simple(one).estimate > 0.15


def test_k_try_copies_keep_k_tries_per_bug() -> None:
    tries = _many_tries(400, 6, 0.0, seed=4)
    c = Comparison("s|x", "d", "s", "a vs b", "choice", "a", "b", experiment=True)
    d = k_try_copy(c, tries, k=3, draw=0)
    assert (d.groupby("bug_id").size() == 3).all()
    assert np.isfinite(k_try_answer(c, d))
    # same draw, same copy (reproducible)
    pd.testing.assert_frame_equal(d, k_try_copy(c, tries, k=3, draw=0))


# ---------------------------------------------------------------- amendment A8


def test_leftover_form_reproduces_rungs_0_and_1(null_case: pd.DataFrame) -> None:
    for parts, fn in ((cr.simple_parts, cr.simple), (cr.linear_parts, cr.linear)):
        ry, rt = parts(null_case)
        assert cr.partial_out(ry, rt).estimate == pytest.approx(fn(null_case).estimate, abs=1e-10)


def test_cached_text_columns_replace_the_fold_text_model(null_case: pd.DataFrame) -> None:
    d = null_case.head(400).copy()
    rng = np.random.default_rng(0)
    for i in range(5):
        d[f"txt_{i}"] = rng.normal(size=len(d))
    assert cr.has_cached_text(d) and not cr.has_cached_text(null_case)
    r = cr.dml(d, 5, seed=0)
    assert np.isfinite(r.estimate) and r.se > 0


def test_aligned_keys_weight_bugs_as_the_methods_do() -> None:
    tries = _many_tries(1500, 6, 0.08, seed=6)
    c = Comparison("s|x", "d", "s", "a vs b", "choice", "a", "b", experiment=True)
    frozen = answer_key(c, tries, reps=100)
    assert sm.aligned_key(c, tries, "cmh", reps=100)[0] == pytest.approx(frozen.estimate)
    for w in ("dml", "ate"):
        est, lo, hi = sm.aligned_key(c, tries, w, reps=200)
        assert lo < 0.08 < hi  # one true effect: every weighting finds it
    # with 6 tries per bug the dml weights are the cmh weights divided by 6
    kp, kd = sm.key_parts(c, tries, "cmh"), sm.key_parts(c, tries, "dml")
    np.testing.assert_allclose(kd.den.to_numpy() * 6, kp.den.to_numpy())


def test_own_rate_leaves_the_drawn_try_out() -> None:
    tries = pd.DataFrame(
        {
            "bug_id": ["b1"] * 3 + ["b2"],
            "try_id": ["t1", "t2", "t3", "t4"],
            "setup": "s",
            "y": [1.0, 0.0, 0.0, 1.0],
        }
    )
    c = Comparison("s|x", "d", "s", "a vs b", "choice", "a", "b", experiment=True)
    copy = pd.DataFrame({"bug_id": ["b1", "b2"], "try_id": ["t1", "t4"], "y": [1.0, 1.0]})
    o = sm.own_loo(c, tries, copy).set_index("bug_id")
    assert o.loc["b1", "own_rate"] == 0.0 and o.loc["b1", "n_own"] == 2
    assert o.loc["b2", "own_missing"] == 1.0 and o.loc["b2", "own_rate"] == -1.0


def test_realistic_other_ai_uses_one_try_per_setup() -> None:
    rows = [("b1", f"o{j}", "other1", float(j % 2)) for j in range(10)]
    rows += [("b1", f"p{j}", "other2", 1.0) for j in range(3)]
    rows += [("b1", "s0", "s", 0.0)]
    tries = pd.DataFrame(rows, columns=["bug_id", "try_id", "setup", "y"])
    c = Comparison("s|x", "d", "s", "a vs b", "choice", "a", "b", experiment=True)
    r = sm.realistic_other(c, tries, pd.Index(["b1"], name="bug_id"), draw=0).iloc[0]
    assert r.n_other == 2
    assert r.other_rate in (0.5, 1.0)  # (0 or 1 from other1 + 1 from other2) / 2
