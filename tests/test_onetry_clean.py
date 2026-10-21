"""Step 3 of the one-try study: cleaning rules and outputs."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from agent_tips.config import Config
from agent_tips.onetry import clean, collect


def _quiet(_: str) -> None:
    pass


@pytest.mark.parametrize(
    "prompt",
    [
        "x\n<issue_description>\nThe bug\n</issue_description>\nmore",
        "x\n<pr_description>\nThe bug\n</pr_description>\nmore",
    ],
)
def test_report_wrapper_is_removed(prompt: str) -> None:
    got = (
        duckdb.connect()
        .execute("SELECT regexp_extract(?, ?, 1)", [prompt, clean.TAGGED_REPORT])
        .fetchone()
    )
    assert got is not None and got[0] == "The bug"


def test_funnel_applies_each_rule(tmp_path: Path) -> None:
    rows = [
        # try_id, dataset, setup, issue_id, issue_hash, resolved, outcome_known
        ("a", "newer_ai_nvidia", "s1", "Bug-1", "h1", 1, True),
        ("a", "newer_ai_nvidia", "s1", "Bug-1", "h1", 1, True),  # duplicate try ID (rule 1)
        ("b", "newer_ai_nvidia", "s1", "bug-1", "h1", 0, True),  # same bug after lower-casing
        ("c", "newer_ai_nvidia", "s1", "bug-1", "h2", 1, True),  # different bug text (rule 3)
        ("d", "newer_ai_nvidia", "s1", "bug-2", "h3", None, False),  # unknown outcome (rule 2)
        ("e", "newer_ai_nvidia", "s1", "bug-2", "h3", 0, True),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "try_id",
            "dataset",
            "setup",
            "issue_id",
            "issue_hash",
            "resolved",
            "outcome_known",
        ],
    )
    (tmp_path / "raw").mkdir()
    df.to_parquet(tmp_path / "raw" / "tries.parquet")
    f = clean.funnel(duckdb.connect(), tmp_path / "raw")
    tries = [r["tries"] for r in f]
    removed = [r["removed"] for r in f]
    assert tries == [6, 5, 4, 3]
    assert removed == [None, 1, 1, 1]
    assert f[-1]["bugs"] == 2


@pytest.fixture(scope="module")
def cleaned(big_project: Config) -> Config:
    collect.run(big_project, log=_quiet)
    clean.run(big_project, log=_quiet)
    return big_project


def test_clean_tables_and_report(cleaned: Config) -> None:
    c = cleaned.path("onetry_data") / "clean"
    tries = pd.read_parquet(c / "tries.parquet")
    bugs = pd.read_parquet(c / "bugs.parquet")
    setups = pd.read_parquet(c / "setups.parquet")
    assert tries.try_id.is_unique
    assert set(tries.y.unique()) <= {0.0, 1.0}
    older = tries[tries.dataset == "older_ai"]
    assert set(older.first_move_older.dropna()) <= {"R", "S", "other"}
    assert {"R", "S"} <= set(older.first_move_older)
    nv = tries[tries.dataset == "newer_ai_nvidia"]
    assert {"test first", "fix first"} <= set(nv["order"])
    assert nv.first_move_nvidia.notna().all()
    # rerun-after-fix is only defined for tries that edited code
    assert tries.loc[tries["order"] == "no edit", "rerun_after_fix"].isna().all()
    # the older AI's bug report sits between ISSUE: and INSTRUCTIONS:
    ob = bugs[bugs.dataset == "older_ai"]
    assert ob.wrapper_found.all()
    assert ob.report.str.startswith("bug ").all()
    assert (ob.fix_files == 1).all()
    assert setups.studied.all()
    text = (cleaned.root / "docs" / "one_try" / "03_cleaning.md").read_text(encoding="utf-8")
    assert text.startswith("# Step 3: Data Cleaning")
    assert "Rule 3: identical bug text" in text


def test_dashed_wrapper_and_repository_ids() -> None:
    con = duckdb.connect()
    got = con.execute(
        "SELECT regexp_extract(?, ?, 1)",
        ["intro\n--- BEGIN ISSUE ---\nThe bug\n--- END ISSUE ---\nrest", clean.DASHED_REPORT],
    ).fetchone()
    assert got is not None and got[0] == "The bug"
    for bug, repo in (
        ("astropy__astropy-12907", "astropy__astropy"),
        ("owner_repo_pr388", "owner_repo"),
    ):
        r = con.execute("SELECT regexp_extract(?, ?, 1)", [bug, clean.REPO_FROM_ID]).fetchone()
        assert r is not None and r[0] == repo
