import pandas as pd
import pytest

from agent_tips.cleaning.pipeline import run as clean
from agent_tips.config import Config
from agent_tips.features.build import run as build


@pytest.fixture()
def featured(fake_project: Config) -> Config:
    clean(fake_project, log=lambda _: None)
    build(fake_project, log=lambda _: None)
    return fake_project


def test_try_features_arms(featured: Config) -> None:
    tf = pd.read_parquet(featured.path("processed") / "try_features.parquet").set_index("try_id")
    older = tf[tf.index.str.startswith("older_ai")]
    # Fixture: create reproduce.py, then edit it -> a test script, but no code edit.
    assert set(older.first_move_arm) == {"R"}
    assert set(older.label0) == {"repro_create"}
    assert set(older.order_arm) == {"no edit"}
    assert (older.n_repro_create == 1).all()


def test_issue_features(featured: Config) -> None:
    issues = pd.read_parquet(featured.path("processed") / "issues.parquet")
    assert set(issues.dataset) == {"older_ai", "newer_ai_nebius", "newer_ai_nvidia"}
    assert issues.issue_key.is_unique


def test_validation_and_report(featured: Config) -> None:
    from agent_tips.features import report

    assert (featured.path("interim") / "label_sample_blind.csv").is_file()
    assert (featured.path("results") / "label_validation.json").is_file()
    out = report.build(featured, out=featured.root / "r.md")
    assert "Label validation" in out.read_text(encoding="utf-8")


def test_cohen_kappa() -> None:
    from agent_tips.features.validate import cohen_kappa

    a = pd.Series(["x", "x", "y", "y"])
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, pd.Series(["y", "y", "x", "x"])) == -1.0
