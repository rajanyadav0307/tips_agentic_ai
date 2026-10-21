"""The whole one-try study (Steps 2-7) end to end on a copy of the fake project."""

import copy
import json
import shutil
from pathlib import Path

import pandas as pd

from agent_tips.config import Config
from agent_tips.onetry import clean, collect, eda, evaluate, features, model


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
        }
    )
    cfg = Config(root=tmp_path, raw=raw)
    collect.run(cfg, log=_quiet)
    clean.run(cfg, log=_quiet)
    eda.run(cfg, log=_quiet)
    features.run(cfg, log=_quiet)
    model.run(cfg, log=_quiet)
    report = evaluate.run(cfg, log=_quiet)

    res = cfg.path("onetry_results")
    keys = pd.read_parquet(res / "answer_keys.parquet")
    assert len(keys) > 0 and keys.noise_floor.notna().all()
    units = pd.concat(pd.read_parquet(f) for f in (res / "units").glob("*.parquet"))
    for m in ("rung0", "rung1", "rung2", "rung4", "rung5", "rung6", "rung5_forest", "rung2_aipw"):
        assert m in set(units.method), m
    assert units[units.method == "rung0"].draw.max() == 2  # precision draws included
    ev = json.loads((res / "evaluation.json").read_text(encoding="utf-8"))
    assert set(ev["hypotheses"]) == {"H1", "H2", "H3", "H4"}
    assert ev["repro"]["all_identical"]
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Step 7: Evaluation")
    assert "| H3:" in text
