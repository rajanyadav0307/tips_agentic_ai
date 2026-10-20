import json

import pandas as pd
import pytest

from agent_tips.cleaning.pipeline import run
from agent_tips.cleaning.schema import STEPS, TRIES, names
from agent_tips.config import Config


@pytest.fixture()
def cleaned(fake_project: Config) -> Config:
    run(fake_project, log=lambda _: None)
    return fake_project


def _tries(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.path("processed") / "tries.parquet")


def test_tries_have_schema_columns_and_unique_ids(cleaned: Config) -> None:
    t = _tries(cleaned)
    assert list(t.columns) == names(TRIES)
    assert t.try_id.is_unique
    assert t.groupby("dataset").size().to_dict() == {
        "newer_ai_nebius": 2,
        "newer_ai_nvidia": 3,
        "older_ai": 3,
    }


def test_flags(cleaned: Config) -> None:
    t = _tries(cleaned).set_index("try_id")
    nv = t[t.dataset == "newer_ai_nvidia"]
    assert nv.outcome_known.tolist().count(False) == 1
    assert nv.resolved.isna().sum() == 1
    older = t[t.dataset == "older_ai"]
    assert older.in_scope.all() and older.same_input.all() and older.dated.all()
    assert older.batch_day.notna().all()
    assert set(nv.subset) == {"sweagent/minimax_m25/swe-rebench-v2"}
    assert set(nv.ai) == {"minimax_m25"}


def test_older_ai_steps(cleaned: Config) -> None:
    s = pd.read_parquet(cleaned.path("processed") / "steps" / "older_ai.parquet")
    assert list(s.columns) == names(STEPS)
    first = s[s.step == 0]
    assert set(first.tool) == {"create"}
    assert set(first.path) == {"reproduce.py"}
    second = s[s.step == 1]
    assert set(second.tool) == {"edit"}
    assert set(second.open_file) == {"/r/reproduce.py"}


def test_funnel_counts(cleaned: Config) -> None:
    f = json.loads((cleaned.path("results") / "funnel.json").read_text(encoding="utf-8"))
    rows = {(r["dataset"], r["condition"]): r for r in f["funnel"]}
    assert rows[("older_ai", "TRUE")]["tries"] == 3
    assert rows[("newer_ai_nvidia", "outcome_known")]["tries"] == 2
    assert rows[("newer_ai_nvidia", "outcome_known")]["removed"] == 1
    assert f["duplicate_try_ids"] == 0
    assert f["steps"] == {"older_ai": 6, "newer_ai_nebius": 4, "newer_ai_nvidia": 3}


def test_report_and_eda_run_on_fixtures(cleaned: Config) -> None:
    from agent_tips.cleaning import report
    from agent_tips.eda import charts

    out = report.build(cleaned, out=cleaned.root / "report.md")
    text = out.read_text(encoding="utf-8")
    assert "## Funnels" in text and "Data dictionary" in text
    stats = charts.run(cleaned, log=lambda _: None)
    assert stats.is_file()
    assert (
        len(list((cleaned.root / "docs" / "figures" / "eda").glob("*.png"))) >= 6
    )  # language chart needs 500+ tries
