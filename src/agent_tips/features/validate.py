"""Step 5: validate the action labels without human effort.

Two checks:

(a) **Observation check:** a label should agree with what the environment
    replied. For example, a "create" step should be followed by a
    "file created" reply. This estimates the precision of each label among the
    steps where the reply makes it checkable.
(b) **Blind second labeling:** a stratified random sample of steps is written
    *without* the rule labels. The assistant labels it independently, and
    agreement with the rules is measured with Cohen's kappa. This is reported
    as AI-assisted, not human, validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from agent_tips.config import Config
from agent_tips.features.labels import LABELS

CREATED = r"(File created successfully|\[File: [^\]]*\(\d+ lines total\)\])"
EDITED = (
    r"(has been edited|File updated|lines total after edit|\[File: [^\]]*\(\d+ lines total\)\])"
)
# The editor refused the edit (the agent did try to edit; the label is still right).
EDIT_REJECTED = (
    r"(Your proposed edit has introduced new syntax error|edit was not applied|"
    r"No replacement was performed|did not appear verbatim|Invalid `?(new_str|old_str)`?)"
)
SHELL_ERROR = (
    r"(No such file or directory|Permission denied|syntax error|cannot create|command not found)"
)
TEST_OUTPUT = (
    r"(?i)(test session starts|collected \d+|passed|failed|=== RUN|--- (PASS|FAIL)|^ok\s|\nok\s|"
    r"FAIL|PASS|Tests?:|Test Suites:|test result:|Ran \d+ tests?|BUILD (SUCCESS|FAILURE)|"
    r"no tests? ran|no test files|running \d+ tests?|Running |error|={8,}|deselected|\d+ selected)"
)
EDITOR_TOOLS = ("str_replace_editor", "create", "edit")
SAMPLE_PER_LABEL = 20


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def observation_check(cfg: Config, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Precision of each label, judged by the environment's reply.

    Returns one row per dataset × label with the number of steps, the number
    that are checkable, the number confirmed, and precision = confirmed / checkable.
    """
    proc = cfg.path("processed")
    rows = []
    for ds_path in sorted((proc / "labels").glob("*.parquet")):
        ds = ds_path.stem
        steps = (proc / "steps" / f"{ds}.parquet").as_posix()
        editor = ", ".join(_q(t) for t in EDITOR_TOOLS)
        df = con.execute(f"""
          WITH j AS (
            SELECT l.label, s.tool, s.obs_head, s.obs_tail
            FROM read_parquet('{ds_path.as_posix()}') l
            JOIN read_parquet('{steps}') s USING (try_id, step))
          SELECT label, count(*) AS steps,
            count(*) FILTER (WHERE label IN ('repro_create', 'create_other', 'edit_code',
                                             'edit_created', 'test_suite')) AS checkable,
            count(*) FILTER (WHERE
              (label IN ('repro_create', 'create_other') AND tool IN ({editor})
                 AND regexp_matches(obs_head, {_q(CREATED)}))
              OR (label IN ('repro_create', 'create_other') AND tool NOT IN ({editor})
                 AND NOT regexp_matches(obs_head, {_q(SHELL_ERROR)}))
              OR (label IN ('edit_code', 'edit_created') AND tool IN ({editor})
                 AND (regexp_matches(obs_head, {_q(EDITED)})
                      OR regexp_matches(obs_head, {_q(EDIT_REJECTED)})))
              OR (label IN ('edit_code', 'edit_created') AND tool NOT IN ({editor})
                 AND NOT regexp_matches(obs_head, {_q(SHELL_ERROR)}))
              OR (label = 'test_suite' AND (regexp_matches(obs_head, {_q(TEST_OUTPUT)})
                                            OR regexp_matches(obs_tail, {_q(TEST_OUTPUT)})))
            ) AS confirmed,
            count(*) FILTER (WHERE label IN ('edit_code', 'edit_created') AND tool IN ({editor})
              AND regexp_matches(obs_head, {_q(EDIT_REJECTED)})) AS edit_rejected
          FROM j GROUP BY label ORDER BY label""").df()
        df.insert(0, "dataset", ds)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["precision"] = np.where(
        out.checkable > 0, out.confirmed / out.checkable.clip(lower=1), np.nan
    )
    return out


def draw_sample(
    cfg: Config,
    con: duckdb.DuckDBPyConnection,
    per_label: int = SAMPLE_PER_LABEL,
    labels: tuple[str, ...] | None = None,
    round_id: int = 1,
) -> tuple[Path, Path]:
    """Write a blind sample (no labels) and a separate answer key.

    Up to ``per_label`` steps per label per dataset, drawn with a fixed seed.
    Each validation round uses a different seed, so a later round never reuses
    the steps that were used to fix the rules.
    """
    seed = f"{cfg.seed}-r{round_id}"
    only = "" if not labels else "WHERE label IN (" + ", ".join(_q(x) for x in labels) + ")"
    suffix = "" if round_id == 1 else f"_r{round_id}"
    proc, interim = cfg.path("processed"), cfg.path("interim")
    interim.mkdir(parents=True, exist_ok=True)
    parts = []
    for ds_path in sorted((proc / "labels").glob("*.parquet")):
        ds = ds_path.stem
        steps = (proc / "steps" / f"{ds}.parquet").as_posix()
        parts.append(f"""
          SELECT '{ds}' AS dataset, l.try_id, l.step, l.label AS rule_label,
                 s.tool, s.action, left(s.command, 300) AS command, s.path, s.open_file,
                 left(replace(s.obs_head, chr(10), ' | '), 150) AS reply
          FROM (SELECT *, row_number() OVER (PARTITION BY label
                  ORDER BY hash(try_id || ':' || step || ':{seed}')) AS rk
                FROM read_parquet('{ds_path.as_posix()}') {only}) l
          JOIN read_parquet('{steps}') s USING (try_id, step)
          WHERE l.rk <= {per_label}""")
    df = con.execute(" UNION ALL ".join(parts) + " ORDER BY dataset, try_id, step").df()
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    df.insert(0, "id", np.arange(1, len(df) + 1))
    key = interim / f"label_sample{suffix}_key.csv"
    blind = interim / f"label_sample{suffix}_blind.csv"
    df[["id", "dataset", "try_id", "step", "rule_label"]].to_csv(key, index=False)
    df.drop(columns=["rule_label"]).to_csv(blind, index=False)
    return blind, key


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two labelings of the same items."""
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a.to_numpy() == b.to_numpy()).mean())
    pa = a.value_counts(normalize=True).reindex(cats, fill_value=0)
    pb = b.value_counts(normalize=True).reindex(cats, fill_value=0)
    pe = float((pa * pb).sum())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def agreement(annotated: Path) -> dict[str, Any]:
    """Agreement between rule labels and the second labeling (``annotator_label``)."""
    df = pd.read_csv(annotated)
    bad = set(df.annotator_label) - set(LABELS)
    if bad:
        raise ValueError(f"Unknown labels in annotations: {sorted(bad)}")
    res: dict[str, Any] = {
        "n": int(len(df)),
        "kappa": round(cohen_kappa(df.rule_label, df.annotator_label), 4),
        "agreement": round(float((df.rule_label == df.annotator_label).mean()), 4),
        "by_dataset": {},
        "by_label": {},
    }
    for ds, g in df.groupby("dataset"):
        res["by_dataset"][str(ds)] = {
            "n": int(len(g)),
            "kappa": round(cohen_kappa(g.rule_label, g.annotator_label), 4),
            "agreement": round(float((g.rule_label == g.annotator_label).mean()), 4),
        }
    for lab in LABELS:
        rule = df.rule_label == lab
        ann = df.annotator_label == lab
        tp = int((rule & ann).sum())
        res["by_label"][lab] = {
            "rule_count": int(rule.sum()),
            "precision": round(tp / int(rule.sum()), 4) if rule.sum() else None,
            "recall": round(tp / int(ann.sum()), 4) if ann.sum() else None,
        }
    disagreements = df[df.rule_label != df.annotator_label]
    res["disagreements"] = disagreements[
        ["id", "dataset", "rule_label", "annotator_label", "command"]
    ].to_dict("records")
    return res


def write_results(cfg: Config, obs: pd.DataFrame, agree: dict[str, Any] | None) -> Path:
    """Save validation results to ``outputs/results/label_validation.json``."""
    out = cfg.path("results") / "label_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"observation_check": obs.to_dict("records"), "second_labeling": agree}
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out
