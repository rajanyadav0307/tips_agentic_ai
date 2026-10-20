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


def test_analysis_runs_on_fixtures(featured: Config) -> None:
    import warnings

    from agent_tips.models import analysis

    featured.raw["analysis"].update({"bootstrap_reps": 50, "permutation_reps": 20})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = analysis.run(featured, log=lambda _: None)
    assert (featured.path("results") / "estimates.parquet").is_file()
    assert set(summary["decisions"]) == {"P1", "P2", "P3"}


def test_full_chain_on_big_fixture(big_project: Config) -> None:
    """Steps 3-8 end to end on a larger fake project: every output is written."""
    import warnings

    from agent_tips.eda import charts
    from agent_tips.evaluation import report as eval_report
    from agent_tips.evaluation import robustness
    from agent_tips.models import analysis
    from agent_tips.models import report as model_report
    from agent_tips.viz import figures, tables

    cfg = big_project

    def quiet(_: str) -> None:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean(cfg, log=quiet)
        charts.run(cfg, log=quiet)
        build(cfg, log=quiet)
        summary = analysis.run(cfg, log=quiet)
        model_report.build(cfg, out=cfg.root / "06.md")
        robustness.run(cfg, log=quiet, repro=False)
        eval_report.build(cfg, out=cfg.root / "07.md")
        figures.run(cfg, log=quiet)
        tables.run(cfg, log=quiet)
    assert set(summary["decisions"]) == {"P1", "P2", "P3"}
    assert (cfg.root / "06.md").read_text(encoding="utf-8").startswith("# Step 6")
    assert (cfg.root / "07.md").read_text(encoding="utf-8").startswith("# Step 7")
    assert len(list((cfg.path("figures") / "paper").glob("*.pdf"))) == 8
    numbers = (cfg.root / "outputs" / "tables" / "numbers.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\POneEst}" in numbers
