"""Step 4 of the one-try study: exploration and the comparison inventory."""

import pandas as pd

from agent_tips.config import Config
from agent_tips.onetry import clean, collect, eda
from agent_tips.onetry.comparisons import Comparison, key_counts


def _quiet(_: str) -> None:
    pass


def test_key_counts_need_both_choices_in_a_bug() -> None:
    tries = pd.DataFrame(
        {
            "setup": ["s"] * 6,
            "bug_id": ["x", "x", "y", "y", "z", "z"],
            "order": [
                "test first",
                "fix first",
                "test first",
                "test first",
                "fix first",
                "no edit",
            ],
            "eval_day": [None] * 6,
        }
    )
    c = Comparison(
        "s|test_first", "d", "s", "test vs fix", "order", "test first", "fix first", False
    )
    k = key_counts(c, tries)
    assert k["key_bugs"] == 1  # only bug x has both choices
    assert k["tries_a"] == 3 and k["tries_b"] == 2


def test_older_ai_groups_are_bug_by_day() -> None:
    tries = pd.DataFrame(
        {
            "setup": ["m"] * 4,
            "bug_id": ["x"] * 4,
            "first_move_older": ["R", "S", "R", "S"],
            "eval_day": ["2024-10-01", "2024-10-01", "2024-10-02", None],
        }
    )
    c = Comparison(
        "o|r", "older_ai", "m", "R vs S", "first_move_older", "R", "S", True, by_day=True
    )
    k = key_counts(c, tries)
    assert k["key_groups"] == 1  # day 2 has only R; the undated try is not used
    assert k["key_bugs"] == 1


def test_eda_writes_inventory_and_report(big_project: Config) -> None:
    collect.run(big_project, log=_quiet)
    clean.run(big_project, log=_quiet)
    report = eda.run(big_project, log=_quiet)
    inv = pd.read_parquet(big_project.path("onetry_results") / "inventory.parquet")
    assert {"cid", "key_bugs", "qualifies", "kind", "share_with_other"} <= set(inv.columns)
    # the fake project has 120 NVIDIA bugs per setup with both orders in many of them
    assert inv[inv.cid.str.endswith("|test_first")].key_bugs.min() > 0
    assert set(inv.kind) <= {"true experiment", "mid-run"}
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Step 4: Exploratory Data Analysis")
    assert (big_project.root / "docs" / "figures" / "onetry" / "o5_inventory.png").is_file()
