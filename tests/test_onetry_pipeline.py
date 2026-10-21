"""The whole one-try study (Steps 2-7) end to end on a copy of the fake project."""

import copy
import json
import shutil
from pathlib import Path

import pandas as pd

from agent_tips.config import Config
from agent_tips.onetry import clean, collect, eda, evaluate, features, model, share


def _quiet(_: str) -> None:
    pass


def test_steps_2_to_7_end_to_end(big_project: Config, tmp_path: Path) -> None:
    # own copy of the raw data, so lowering the size rule does not affect other tests
    shutil.copytree(big_project.path("raw"), tmp_path / "data" / "raw")
    raw = copy.deepcopy(big_project.raw)
    raw["onetry"].update(
        {
            "min_bugs": 15,
            "draws": 2,
            "precision_draws": 3,
            "check_draws": 1,
            "secondary_draws": 1,
            "jobs": 1,
            "bootstrap_reps": 50,
            "k_tries": [2, 3],
            "forest_draws": 1,
            "machines": {
                "here": {"speed": 1.0, "jobs": 1, "home": True},
                "partner": {"speed": 1.0, "jobs": 1},
            },
        }
    )
    cfg = Config(root=tmp_path, raw=raw)
    collect.run(cfg, log=_quiet)
    clean.run(cfg, log=_quiet)
    eda.run(cfg, log=_quiet)
    features.run(cfg, log=_quiet)
    model.run(cfg, log=_quiet)

    # Step 7 shared across two computers (amendment A9): plan, run each share,
    # move the partner's results through a zip, check them here, then score.
    plan = share.plan(cfg, log=_quiet)
    assert sorted(sum(plan["comparisons"].values(), [])) == sorted(
        pd.read_parquet(cfg.path("onetry_results") / "answer_keys.parquet").cid
    )
    for m in ("here", "partner"):
        share.run(cfg, m, log=_quiet)
        assert share.missing(cfg, share.assigned(cfg, m)) == []
    res = cfg.path("onetry_results")
    packed = share.pack(cfg, "partner", log=_quiet)
    before = {f: f.read_bytes() for f in res.glob("*/*/*.parquet")}
    for cid in share.assigned(cfg, "partner"):
        for folder in ("units", "leftovers"):
            shutil.rmtree(res / folder / evaluate._slug(cid), ignore_errors=True)
    share.merge(cfg, packed, log=_quiet)
    assert {f: f.read_bytes() for f in res.glob("*/*/*.parquet")} == before
    if share.assigned(cfg, "partner"):
        assert share.crosscheck(cfg, log=_quiet)["passed"]
    report = evaluate.run(cfg, log=_quiet)

    keys = pd.read_parquet(res / "answer_keys.parquet")
    assert len(keys) > 0 and keys.noise_floor.notna().all()
    units = evaluate.load_units(cfg)
    for m in (
        "rung0",
        "rung1",
        "rung2",
        "rung4",
        "rung5",
        "rung6",
        "rung5_forest",
        "rung2_aipw",
        "rung5_real",
        "rung7",
        "rung0_allbugs_trim",
    ):
        assert m in set(units.method), m
    assert units[units.method == "rung0"].draw.max() == 2  # precision draws included
    ev = json.loads((res / "evaluation.json").read_text(encoding="utf-8"))
    assert set(ev["hypotheses"]) == {"H1", "H2", "H3", "H4"}
    assert ev["repro"]["all_identical"]
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Step 7: Evaluation")
    assert "| H3:" in text
    assert "### Noise-corrected bias" in text and "### Paired contrasts" in text
    assert (res / "a8_floors.parquet").is_file()
