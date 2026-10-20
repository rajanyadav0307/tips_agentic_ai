"""Step 3: build the clean ``tries`` and ``steps`` tables and the cleaning funnel.

Outputs (under ``data/processed/``):

- ``tries.parquet``: one row per try for all three datasets, with flags and batches.
- ``steps/<dataset>.parquet``: one row per AI action, for in-scope tries.

And ``outputs/results/funnel.json``: row and issue counts after each filter.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb

from agent_tips.cleaning import sources
from agent_tips.cleaning.schema import TRIES, names
from agent_tips.config import Config
from agent_tips.data.load import has_files, raw_globs

DATASETS = ("older_ai", "newer_ai_nebius", "newer_ai_nvidia")


def _src(cfg: Config, key: str) -> str:
    return "[" + ", ".join("'" + g.replace("'", "''") + "'" for g in raw_globs(cfg, key)) + "]"


def _queries(cfg: Config, key: str) -> tuple[str, str]:
    src = _src(cfg, key)
    if key == "older_ai":
        model = cfg.raw["datasets"]["older_ai"]["model_name"]
        return sources.older_ai_tries(src, model), sources.older_ai_steps(src, model)
    if key == "newer_ai_nebius":
        return sources.nebius_tries(src), sources.nebius_steps(src)
    return sources.nvidia_tries(src), sources.nvidia_steps(src)


FLAGS_SQL = """
CREATE OR REPLACE TABLE tries AS
WITH c AS (
  SELECT issue_key, issue_hash, count(*) AS n FROM raw_tries WHERE in_scope GROUP BY 1, 2),
m AS (
  SELECT issue_key, issue_hash FROM (
    SELECT *, row_number() OVER (PARTITION BY issue_key ORDER BY n DESC, issue_hash) AS rk
    FROM c) WHERE rk = 1),
f AS (
  SELECT t.*,
    coalesce(t.issue_hash = m.issue_hash, FALSE) AS same_input,
    CASE WHEN t.dataset = 'older_ai' THEN t.eval_ts IS NOT NULL ELSE TRUE END AS dated
  FROM raw_tries t LEFT JOIN m USING (issue_key)),
g AS (
  SELECT *,
    CASE WHEN eval_ts IS NOT NULL THEN eval_ts - lag(eval_ts) OVER w END AS gap
  FROM f WINDOW w AS (PARTITION BY issue_key, eval_ts IS NOT NULL ORDER BY eval_ts, try_id))
SELECT * EXCLUDE (gap),
  date_trunc('day', eval_ts) AS batch_day,
  time_bucket(INTERVAL 6 HOUR, eval_ts) AS batch_6h,
  date_trunc('hour', eval_ts) AS batch_hour,
  CASE WHEN eval_ts IS NOT NULL THEN CAST(sum(CASE WHEN gap IS NULL OR gap > INTERVAL 60 MINUTE
       THEN 1 ELSE 0 END) OVER w2 AS INTEGER) END AS batch_gap60,
  CASE WHEN eval_ts IS NOT NULL THEN CAST(sum(CASE WHEN gap IS NULL OR gap > INTERVAL 10 MINUTE
       THEN 1 ELSE 0 END) OVER w2 AS INTEGER) END AS batch_gap10
FROM g
WINDOW w2 AS (PARTITION BY issue_key, eval_ts IS NOT NULL ORDER BY eval_ts, try_id
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
"""

FUNNEL_STAGES: dict[str, list[tuple[str, str]]] = {
    "older_ai": [
        ("All tries in the dataset", "TRUE"),
        ("In scope (swe-agent-llama-70b)", "in_scope"),
        ("Same issue text as the most common one", "in_scope AND same_input"),
        ("Dated (evaluation timestamp found)", "in_scope AND same_input AND dated"),
    ],
    "newer_ai_nebius": [
        ("All tries in the dataset", "TRUE"),
        ("Same issue text as the most common one", "same_input"),
    ],
    "newer_ai_nvidia": [
        ("All tries in the 7 subsets", "TRUE"),
        ("Outcome known (resolved is 0 or 1)", "outcome_known"),
        ("Same issue text as the most common one", "outcome_known AND same_input"),
    ],
}


def _funnel(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ds, stages in FUNNEL_STAGES.items():
        prev: int | None = None
        for label, cond in stages:
            n, issues, solved = con.execute(
                f"SELECT count(*), count(DISTINCT issue_key), avg(resolved) "
                f"FROM tries WHERE dataset = '{ds}' AND {cond}"
            ).fetchone() or (0, 0, None)
            rows.append(
                {
                    "dataset": ds,
                    "stage": label,
                    "condition": cond,
                    "tries": int(n),
                    "issues": int(issues),
                    "removed": None if prev is None else int(prev - n),
                    "resolved_rate": None if solved is None else round(float(solved), 4),
                }
            )
            prev = int(n)
    return rows


def run(cfg: Config, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Build the clean tables and the funnel; return a summary."""
    out = cfg.path("processed")
    (out / "steps").mkdir(parents=True, exist_ok=True)
    results = cfg.path("results")
    results.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    tmp_files: list[Path] = []
    step_counts: dict[str, int] = {}
    for key in DATASETS:
        if not has_files(cfg, key):
            log(f"skip {key}: no raw files (run `agent-tips download`)")
            continue
        tries_sql, steps_sql = _queries(cfg, key)
        tmp = out / f"_tries_{key}.parquet"
        log(f"{key}: building tries")
        con.execute(f"COPY ({tries_sql}) TO '{tmp.as_posix()}' (FORMAT parquet)")
        tmp_files.append(tmp)
        steps_path = out / "steps" / f"{key}.parquet"
        log(f"{key}: building steps")
        con.execute(
            f"COPY ({steps_sql}) TO '{steps_path.as_posix()}' (FORMAT parquet, ROW_GROUP_SIZE 500000)"
        )
        row = con.execute(
            f"SELECT count(*) FROM read_parquet('{steps_path.as_posix()}')"
        ).fetchone()
        step_counts[key] = int(row[0]) if row else 0
    if not tmp_files:
        raise FileNotFoundError("No raw data found; run `agent-tips download` first")
    parts = ", ".join(f"'{p.as_posix()}'" for p in tmp_files)
    con.execute(
        f"CREATE TABLE raw_tries AS SELECT * FROM read_parquet([{parts}], union_by_name = true)"
    )
    con.execute(FLAGS_SQL)
    cols = ", ".join(names(TRIES))
    tries_path = out / "tries.parquet"
    con.execute(
        f"COPY (SELECT {cols} FROM tries ORDER BY try_id) TO '{tries_path.as_posix()}' (FORMAT parquet)"
    )
    funnel = _funnel(con)
    dup = con.execute("SELECT count(*) - count(DISTINCT try_id) FROM tries").fetchone()
    summary = {
        "funnel": funnel,
        "steps": step_counts,
        "duplicate_try_ids": int(dup[0]) if dup else 0,
    }
    (results / "funnel.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for p in tmp_files:
        p.unlink()
    log(f"wrote {tries_path} and {out / 'steps'}")
    return summary
