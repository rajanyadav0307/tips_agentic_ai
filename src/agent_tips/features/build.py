"""Step 5: build all feature tables.

Outputs (under ``data/processed/``):

- ``labels/<dataset>.parquet``: one row per step with its label.
- ``try_features.parquet``: one row per in-scope try. It holds:
  - when the try first did each thing (test script, code edit, run, test suite);
  - counts of these actions, before and after the first code edit;
  - the arms used by the hypotheses (``first_move_arm``, ``order_arm``);
  - localization: the first step that touches a file changed by the real fix.
- ``issues.parquet``: one row per issue with simple text features.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb

from agent_tips.config import Config
from agent_tips.data.load import has_files, raw_globs
from agent_tips.features import validate
from agent_tips.features.labels import RERUN, labels_query

DATASETS = ("older_ai", "newer_ai_nebius", "newer_ai_nvidia")
SEARCH_TOOLS = ("search_dir", "search_file", "find_file", "ls", "grep", "find")

# Issue text features (computed on the issue text only, not the agent prompt).
CODE_BLOCK = r"```"
STACK_TRACE = (
    r"(Traceback \(most recent call last\)|Exception in thread|panic:|"
    r"\n\s+at [\w.$<>]+\(.*:\d+\)|thread '.*' panicked|Error: .*\n\s+at )"
)
FILE_MENTION = r"[\w/.-]+\.(py|go|rs|ts|tsx|js|jsx|java|php|rb|c|cc|cpp|h|hpp|cs|kt|scala)\b"
GOLD_FILE = r"diff --git a/(\S+) b/"


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _src(cfg: Config, key: str) -> str:
    return "[" + ", ".join("'" + g.replace("'", "''") + "'" for g in raw_globs(cfg, key)) + "]"


def build_labels(cfg: Config, con: duckdb.DuckDBPyConnection, log: Callable[[str], None]) -> Path:
    """Label every step and write ``labels/<dataset>.parquet``."""
    out = cfg.path("processed") / "labels"
    out.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        steps = cfg.path("processed") / "steps" / f"{ds}.parquet"
        if not steps.is_file():
            continue
        log(f"{ds}: labelling steps")
        q = labels_query(f"read_parquet('{steps.as_posix()}')")
        con.execute(
            f"COPY (SELECT try_id, step, label, target, "
            f"regexp_matches(command, '{RERUN}') AS is_rerun FROM ({q})) "
            f"TO '{(out / f'{ds}.parquet').as_posix()}' (FORMAT parquet, ROW_GROUP_SIZE 500000)"
        )
    return out


def _try_features_sql(labels: str) -> tuple[str, str]:
    search = ", ".join(f"'{t}'" for t in SEARCH_TOOLS)
    return (
        f"""
    WITH l AS (SELECT * FROM read_parquet('{labels}/*.parquet')),
    f AS (
      SELECT try_id,
        min(step) FILTER (WHERE label = 'repro_create') AS first_repro,
        min(step) FILTER (WHERE label IN ('repro_create', 'create_other')) AS first_create,
        min(step) FILTER (WHERE label = 'edit_code') AS first_edit_code,
        min(step) FILTER (WHERE label IN ('run', 'test_suite')) AS first_run,
        min(step) FILTER (WHERE label = 'test_suite') AS first_suite,
        count(*) FILTER (WHERE label = 'repro_create') AS n_repro_create,
        count(*) FILTER (WHERE label = 'edit_code') AS n_edit_code,
        count(*) FILTER (WHERE label IN ('run', 'test_suite')) AS n_runs,
        count(*) FILTER (WHERE label = 'test_suite') AS n_suite,
        count(*) FILTER (WHERE label = 'cleanup') AS n_cleanup
      FROM l GROUP BY try_id),
    s0 AS (SELECT try_id, label AS label0 FROM l WHERE step = 0),
    a AS (
      SELECT f.try_id,
        count(*) FILTER (WHERE l.step > f.first_edit_code AND l.label IN ('run', 'test_suite')) AS n_runs_after_fix,
        count(*) FILTER (WHERE l.step > f.first_edit_code AND l.label IN ('run', 'test_suite') AND l.is_rerun) AS n_reruns_after_fix,
        count(*) FILTER (WHERE l.step > f.first_edit_code AND l.label = 'test_suite') AS n_suite_after_fix,
        count(*) FILTER (WHERE l.step < f.first_edit_code AND l.label = 'test_suite') AS n_suite_before_fix,
        count(*) FILTER (WHERE l.step > f.first_edit_code AND l.label = 'repro_create') AS n_tests_after_fix
      FROM f JOIN l USING (try_id) GROUP BY f.try_id)
    SELECT f.*, s0.label0, a.* EXCLUDE (try_id),
      CASE WHEN f.first_edit_code IS NULL THEN 'no edit'
           WHEN f.first_repro < f.first_edit_code THEN 'test first'
           ELSE 'fix first' END AS order_arm
    FROM f LEFT JOIN s0 USING (try_id) LEFT JOIN a USING (try_id)
    """,
        search,
    )


def _issue_text_sql(cfg: Config, ds: str) -> str | None:
    if not has_files(cfg, ds):
        return None
    src = _src(cfg, ds)
    tags = (
        r"(?s)<(issue_description|pr_description)>\s*(.*?)\s*</(issue_description|pr_description)>"
    )
    if ds == "older_ai":
        model = cfg.raw["datasets"]["older_ai"]["model_name"]
        return f"""SELECT DISTINCT 'older_ai::' || instance_id AS issue_key,
                     md5(coalesce(trajectory[2].text, '')) AS issue_hash,
                     regexp_extract(trajectory[2].text, '(?s)ISSUE:\\n(.*?)\\n\\nINSTRUCTIONS:', 1) AS text
                   FROM read_parquet({src}) WHERE model_name = '{model}'"""
    if ds == "newer_ai_nebius":
        return f"""SELECT DISTINCT 'newer_ai_nebius::' || instance_id AS issue_key,
                     md5(coalesce(trajectory[2].content, '')) AS issue_hash,
                     coalesce(nullif(regexp_extract(trajectory[2].content, '{tags}', 2), ''),
                              trajectory[2].content) AS text
                   FROM read_parquet({src})"""
    f = r"replace(filename, '\', '/')"
    path_re = r"data/(sweagent|openhands|minisweagent)/([^/]+)/([^/]+)/"
    sub = (
        f"regexp_extract({f}, '{path_re}', 1) || '/' || regexp_extract({f}, '{path_re}', 2)"
        f" || '/' || regexp_extract({f}, '{path_re}', 3)"
    )
    first_user = "list_filter(messages, m -> m.role = 'user')[1].content"
    return f"""SELECT DISTINCT {sub} || '::' || instance_id AS issue_key,
                 md5(coalesce({first_user}, '')) AS issue_hash,
                 coalesce(nullif(regexp_extract({first_user}, '{tags}', 2), ''), {first_user}) AS text,
                 any_value(metadata.reference_patch.patch) OVER (PARTITION BY {sub}, instance_id) AS gold_patch
               FROM read_parquet({src}, filename = true, union_by_name = true)"""


def build_issues(cfg: Config, con: duckdb.DuckDBPyConnection, log: Callable[[str], None]) -> Path:
    """Write ``issues.parquet``: text features and gold files for every issue."""
    tries = (cfg.path("processed") / "tries.parquet").as_posix()
    parts = []
    for ds in DATASETS:
        q = _issue_text_sql(cfg, ds)
        if q is None:
            continue
        log(f"{ds}: issue features")
        gold = "gold_patch" if ds == "newer_ai_nvidia" else "CAST(NULL AS VARCHAR) AS gold_patch"
        cols = "issue_key, issue_hash, text, " + gold
        parts.append(f"SELECT '{ds}' AS dataset, {cols} FROM ({q})")
    con.execute("CREATE OR REPLACE TABLE issue_texts AS " + " UNION ALL ".join(parts))
    if has_files(cfg, "gold_extra") or has_files(cfg, "gold_dev"):
        gsrc = []
        for key in ("gold_extra", "gold_dev"):
            if has_files(cfg, key):
                gsrc.append(f"SELECT instance_id, patch FROM read_parquet({_src(cfg, key)})")
        con.execute("CREATE OR REPLACE TABLE gold AS " + " UNION ALL ".join(gsrc))
        con.execute(
            "UPDATE issue_texts SET gold_patch = g.patch FROM gold g "
            "WHERE issue_texts.dataset = 'older_ai' AND issue_texts.issue_key = 'older_ai::' || g.instance_id"
        )
    out = cfg.path("processed") / "issues.parquet"
    con.execute(f"""
      COPY (
        WITH modal AS (SELECT DISTINCT issue_key, issue_hash FROM read_parquet('{tries}')
                       WHERE in_scope AND same_input)
        SELECT t.dataset, t.issue_key,
          regexp_matches(t.text, {_q(CODE_BLOCK)}) AS has_code_block,
          regexp_matches(t.text, {_q(STACK_TRACE)}) AS has_stack_trace,
          regexp_matches(t.text, {_q(FILE_MENTION)}) AS names_file,
          CAST(length(t.text) AS INTEGER) AS text_length,
          list_distinct(regexp_extract_all(coalesce(t.gold_patch, ''), '{GOLD_FILE}', 1)) AS gold_files
        FROM issue_texts t JOIN modal m USING (issue_key, issue_hash)
      ) TO '{out.as_posix()}' (FORMAT parquet)""")
    return out


def build_try_features(
    cfg: Config, con: duckdb.DuckDBPyConnection, log: Callable[[str], None]
) -> Path:
    """Write ``try_features.parquet`` (arms, counts, localization)."""
    proc = cfg.path("processed")
    sql, search = _try_features_sql((proc / "labels").as_posix())
    log("try features")
    con.execute(f"CREATE OR REPLACE TABLE tf AS {sql}")
    # First move of the older AI's tries (raw tool name at step 0).
    steps = (proc / "steps").as_posix()
    con.execute(f"""CREATE OR REPLACE TABLE first_tool AS
      SELECT try_id, tool AS tool0,
        CASE WHEN tool = 'create' THEN 'R' WHEN tool IN ({search}) THEN 'S' ELSE 'O' END AS first_move_arm
      FROM read_parquet('{steps}/*.parquet') WHERE step = 0""")
    # Localization: first step whose path, open file or command mentions a gold file.
    log("localization")
    tries = (proc / "tries.parquet").as_posix()
    issues = (proc / "issues.parquet").as_posix()
    con.execute(f"""CREATE OR REPLACE TABLE loc AS
      WITH g AS (
        SELECT t.try_id, i.gold_files FROM read_parquet('{tries}') t
        JOIN read_parquet('{issues}') i USING (issue_key)
        WHERE len(i.gold_files) > 0)
      SELECT s.try_id,
        min(s.step) FILTER (WHERE len(list_filter(g.gold_files, f ->
            ends_with(s.path, f) OR ends_with(s.open_file, f) OR contains(s.command, f))) > 0) AS first_gold_step
      FROM read_parquet('{steps}/*.parquet') s JOIN g USING (try_id)
      GROUP BY s.try_id""")
    out = proc / "try_features.parquet"
    con.execute(f"""COPY (
        SELECT tf.*, ft.tool0, ft.first_move_arm, loc.first_gold_step,
               loc.try_id IS NOT NULL AS gold_known
        FROM tf LEFT JOIN first_tool ft USING (try_id) LEFT JOIN loc USING (try_id)
        ORDER BY try_id) TO '{out.as_posix()}' (FORMAT parquet)""")
    return out


def run(cfg: Config, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Build labels, issue features and try features."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    labels = build_labels(cfg, con, log)
    issues = build_issues(cfg, con, log)
    tf = build_try_features(cfg, con, log)
    log("label validation")
    obs = validate.observation_check(cfg, con)
    blind, _ = validate.draw_sample(cfg, con)
    results = validate.write_results(cfg, obs, None)
    return {
        "labels": str(labels),
        "issues": str(issues),
        "try_features": str(tf),
        "blind_sample": str(blind),
        "validation": str(results),
    }
