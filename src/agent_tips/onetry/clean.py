"""Step 3 (one-try study): data cleaning and preparation.

Reads only this study's raw extracts (``data/onetry/raw/``, Step 2) and writes
``data/onetry/clean/``:

- ``labels/*.parquet``: one label per action, from Paper 1's validated labeler;
- ``tries.parquet``: kept tries with outcome, flags and every behaviour choice;
- ``bugs.parquet``: one row per dataset × bug with the cleaned bug report,
  report facts and the size of the human fix;
- ``setups.parquet``: one row per AI setup.

Every removal comes from a written rule and is counted in a funnel
(``outputs/onetry/cleaning.json``); the report is ``docs/one_try/03_cleaning.md``.

Rules (in order):

1. Duplicate try IDs: keep one.
2. Unknown outcome: drop (``resolved`` must be 0 or 1).
3. Identical input: keep only tries whose bug text is the most common text for
   that AI setup and bug (ties: smallest fingerprint).

Flags (kept, not removed): studied setup, step-count outlier, older-AI
evaluation day (hidden model-version proxy), bug report wrapper not found.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb

from agent_tips.config import Config
from agent_tips.features.build import (
    CODE_BLOCK,
    FILE_MENTION,
    SEARCH_TOOLS,
    STACK_TRACE,
    _try_features_sql,
)
from agent_tips.features.labels import RERUN, labels_query
from agent_tips.onetry.db import connect, count, sql_list, value

# Bug-report wrappers added by the agent frameworks (the real report sits inside).
OLDER_REPORT = r"(?s)ISSUE:\n(.*?)\n\nINSTRUCTIONS:"
TAGGED_REPORT = r"(?s)<(?:issue|pr)_description>\n?(.*?)\n?</(?:issue|pr)_description>"
DASHED_REPORT = r"(?s)--- BEGIN ISSUE ---\n?(.*?)\n?--- END ISSUE ---"
# Bug IDs: owner__repo-123 (SWE-bench style) or owner_repo_pr123 (scale-swe)
REPO_FROM_ID = r"^(.*?)(?:-|_pr)[0-9]+$"
CHOICE_COLUMNS = {
    "first_move_older": "First action of the older AI: R = create a file, S = search, other",
    "first_tool_older": "First tool of the older AI (search_dir, find_file, ls, open, grep, create, other)",
    "first_move_nebius": "First action of Nebius: think, view (any look), other",
    "first_move_nvidia": "First tool:action of an NVIDIA setup, e.g. bash:find",
    "order": "test first / fix first / no edit (test or reproduction file before the first code edit)",
    "rerun_after_fix": "Re-ran own test script after the first code edit (yes/no; empty if no edit)",
    "suite_after_fix": "Ran the project's test suite after the first code edit (yes/no; empty if no edit)",
    "run_after_fix": "Ran any code after the first code edit (yes/no; empty if no edit)",
    "tests_after_fix": "Wrote extra tests after the first code edit (yes/no; empty if no edit)",
    "suite_before_fix": "Ran the project's test suite before the first code edit (yes/no; empty if no edit)",
    "cleanup": "Removed files it had created (yes/no)",
}


def _q(pattern: str) -> str:
    """Escape a regex for use inside a SQL string literal."""
    return pattern.replace("'", "''")


def _paths(cfg: Config) -> tuple[Path, Path, Path]:
    base = cfg.path("onetry_data")
    return base / "raw", base / "clean", base / "_tmp"


# ---------------------------------------------------------------- labels


def build_labels(
    con: duckdb.DuckDBPyConnection, raw: Path, out: Path, log: Callable[[str], None]
) -> Path:
    """Label every action with Paper 1's labeler; one flat file per steps file."""
    labels = out / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    for f in sorted((raw / "steps").rglob("*.parquet")):
        name = f.relative_to(raw / "steps").with_suffix("").as_posix().replace("/", "__")
        log(f"label actions: {name}")
        q = labels_query(f"read_parquet('{f.as_posix()}')")
        con.execute(
            f"COPY (SELECT try_id, step, label, target, regexp_matches(command, '{RERUN}') AS is_rerun "
            f"FROM ({q}) ORDER BY try_id, step) TO '{(labels / f'{name}.parquet').as_posix()}' "
            "(FORMAT parquet, ROW_GROUP_SIZE 500000)"
        )
    return labels


# ---------------------------------------------------------------- tries


def tries_sql(raw: Path, clean: Path, studied_older: str) -> str:
    """Kept tries with flags and behaviour choices (rules 1-3 applied)."""
    search = ", ".join(f"'{t}'" for t in SEARCH_TOOLS)
    feats, _ = _try_features_sql((clean / "labels").as_posix())
    steps = sql_list([p for p in (raw / "steps").rglob("*.parquet")])
    return f"""
    WITH t0 AS (
      SELECT *, lower(trim(issue_id)) AS bug_id,
             row_number() OVER (PARTITION BY try_id ORDER BY issue_hash) AS dup
      FROM read_parquet('{(raw / "tries.parquet").as_posix()}')),
    t1 AS (SELECT * EXCLUDE (dup) FROM t0 WHERE dup = 1),
    t2 AS (SELECT * FROM t1 WHERE outcome_known AND resolved IN (0, 1)),
    modal AS (
      SELECT setup, bug_id, issue_hash FROM (
        SELECT setup, bug_id, issue_hash,
               row_number() OVER (PARTITION BY setup, bug_id ORDER BY count(*) DESC, issue_hash) AS rk
        FROM t2 GROUP BY setup, bug_id, issue_hash) WHERE rk = 1),
    t3 AS (SELECT t2.* FROM t2 JOIN modal USING (setup, bug_id, issue_hash)),
    cut AS (SELECT dataset, quantile_cont(n_steps, 0.999) AS hi FROM t3 GROUP BY dataset),
    s0 AS (SELECT try_id, tool AS tool0, action AS action0
           FROM read_parquet({steps}) WHERE step = 0),
    f AS ({feats})
    SELECT t3.try_id, t3.dataset, t3.setup, t3.ai, t3.scaffold, t3.bug_id,
      CAST(t3.resolved AS DOUBLE) AS y,
      t3.dataset <> 'older_ai' OR t3.setup = '{studied_older}' AS studied,
      t3.exit_status, t3.n_steps, t3.has_patch, t3.language, t3.category, t3.issue_hash,
      t3.chars_in, t3.chars_out,
      t3.eval_ts, CAST(date_trunc('day', t3.eval_ts) AS DATE) AS eval_day,
      (t3.n_steps = 0 OR t3.n_steps > cut.hi) AS steps_outlier,
      CASE WHEN t3.dataset = 'older_ai' THEN
        CASE WHEN s0.tool0 = 'create' THEN 'R' WHEN s0.tool0 IN ({search}) THEN 'S' ELSE 'other' END
      END AS first_move_older,
      CASE WHEN t3.dataset = 'older_ai' THEN
        CASE WHEN s0.tool0 IN ({search}, 'create', 'open') THEN s0.tool0 ELSE 'other' END
      END AS first_tool_older,
      CASE WHEN t3.dataset = 'newer_ai_nebius' THEN
        CASE WHEN s0.tool0 = 'think' THEN 'think' WHEN f.label0 = 'look' THEN 'view' ELSE 'other' END
      END AS first_move_nebius,
      CASE WHEN t3.dataset = 'newer_ai_nvidia' THEN s0.tool0 || ':' || s0.action0 END AS first_move_nvidia,
      f.order_arm AS "order",
      CASE WHEN f.first_edit_code IS NOT NULL THEN
        CASE WHEN f.n_reruns_after_fix > 0 THEN 'yes' ELSE 'no' END END AS rerun_after_fix,
      CASE WHEN f.first_edit_code IS NOT NULL THEN
        CASE WHEN f.n_suite_after_fix > 0 THEN 'yes' ELSE 'no' END END AS suite_after_fix,
      CASE WHEN f.first_edit_code IS NOT NULL THEN
        CASE WHEN f.n_runs_after_fix > 0 THEN 'yes' ELSE 'no' END END AS run_after_fix,
      CASE WHEN f.first_edit_code IS NOT NULL THEN
        CASE WHEN f.n_tests_after_fix > 0 THEN 'yes' ELSE 'no' END END AS tests_after_fix,
      CASE WHEN f.first_edit_code IS NOT NULL THEN
        CASE WHEN f.n_suite_before_fix > 0 THEN 'yes' ELSE 'no' END END AS suite_before_fix,
      CASE WHEN f.try_id IS NULL THEN NULL WHEN f.n_cleanup > 0 THEN 'yes' ELSE 'no' END AS cleanup
    FROM t3 JOIN cut USING (dataset)
    LEFT JOIN s0 USING (try_id) LEFT JOIN f USING (try_id)"""


def funnel(con: duckdb.DuckDBPyConnection, raw: Path) -> list[dict[str, Any]]:
    """Rows and bugs left after each rule, per dataset."""
    src = f"read_parquet('{(raw / 'tries.parquet').as_posix()}')"
    stages = [
        ("All extracted tries", f"SELECT *, lower(trim(issue_id)) AS bug_id FROM {src}"),
        (
            "Rule 1: duplicate try IDs removed",
            f"SELECT * FROM (SELECT *, lower(trim(issue_id)) AS bug_id, row_number() OVER "
            f"(PARTITION BY try_id ORDER BY issue_hash) AS d FROM {src}) WHERE d = 1",
        ),
    ]
    stages.append(
        (
            "Rule 2: outcome known",
            f"SELECT * FROM ({stages[-1][1]}) WHERE outcome_known AND resolved IN (0, 1)",
        )
    )
    s2 = stages[-1][1]
    stages.append(
        (
            "Rule 3: identical bug text (most common per setup and bug)",
            f"""WITH t2 AS ({s2}),
        m AS (SELECT setup, bug_id, issue_hash FROM (SELECT setup, bug_id, issue_hash,
              row_number() OVER (PARTITION BY setup, bug_id ORDER BY count(*) DESC, issue_hash) AS rk
              FROM t2 GROUP BY setup, bug_id, issue_hash) WHERE rk = 1)
        SELECT t2.* FROM t2 JOIN m USING (setup, bug_id, issue_hash)""",
        )
    )
    rows: list[dict[str, Any]] = []
    prev: dict[str, int] = {}
    for label, sql in stages:
        for ds, n, bugs, rate in con.execute(
            f"SELECT dataset, count(*), count(DISTINCT bug_id), "
            f"avg(CASE WHEN resolved IN (0, 1) THEN resolved END) FROM ({sql}) GROUP BY 1 ORDER BY 1"
        ).fetchall():
            rows.append(
                {
                    "dataset": ds,
                    "stage": label,
                    "tries": int(n),
                    "bugs": int(bugs),
                    "removed": None if ds not in prev else prev[ds] - int(n),
                    "fix_rate": None if rate is None else round(float(rate), 4),
                }
            )
            prev[ds] = int(n)
    return rows


# ---------------------------------------------------------------- bugs


def bugs_sql(raw: Path, clean: Path) -> str:
    """One row per dataset × bug: cleaned report, report facts and human-fix size."""
    tries = (clean / "tries.parquet").as_posix()
    return f"""
    WITH h AS (  -- the bug text seen by the studied setups (most common across them)
      SELECT dataset, bug_id, issue_hash FROM (
        SELECT dataset, bug_id, issue_hash,
               row_number() OVER (PARTITION BY dataset, bug_id ORDER BY count(*) DESC, issue_hash) AS rk
        FROM read_parquet('{tries}') GROUP BY dataset, bug_id, issue_hash) WHERE rk = 1),
    r AS (SELECT dataset, lower(trim(issue_id)) AS bug_id, issue_hash, text
          FROM read_parquet('{(raw / "reports.parquet").as_posix()}')),
    rep AS (
      SELECT h.dataset, h.bug_id, r.text AS prompt,
        CASE WHEN h.dataset = 'older_ai' THEN regexp_extract(r.text, '{_q(OLDER_REPORT)}', 1)
             ELSE coalesce(nullif(regexp_extract(r.text, '{_q(TAGGED_REPORT)}', 1), ''),
                           regexp_extract(r.text, '{_q(DASHED_REPORT)}', 1)) END AS inner_text
      FROM h JOIN r USING (dataset, bug_id, issue_hash)),
    meta AS (
      SELECT dataset, bug_id, mode(language) AS language, mode(category) AS category
      FROM read_parquet('{tries}') GROUP BY dataset, bug_id),
    fx AS (
      SELECT lower(trim(issue_id)) AS bug_id, any_value(source) AS fix_source,
        any_value(len(regexp_extract_all(patch, 'diff --git '))) AS fix_files,
        any_value(len(list_filter(string_split(patch, chr(10)), x ->
            (starts_with(x, '+') AND NOT starts_with(x, '+++'))
            OR (starts_with(x, '-') AND NOT starts_with(x, '---'))))) AS fix_lines,
        any_value(files_reported) AS fix_files_reported, any_value(lines_reported) AS fix_lines_reported
      FROM read_parquet('{(raw / "fixes.parquet").as_posix()}') WHERE patch IS NOT NULL
      GROUP BY 1)
    SELECT rep.dataset, rep.bug_id,
      regexp_extract(rep.bug_id, '{_q(REPO_FROM_ID)}', 1) AS repository,
      rep.inner_text <> '' AS wrapper_found,
      CASE WHEN rep.inner_text <> '' THEN trim(rep.inner_text) ELSE rep.prompt END AS report,
      length(CASE WHEN rep.inner_text <> '' THEN rep.inner_text ELSE rep.prompt END) AS report_chars,
      regexp_matches(rep.inner_text, '{_q(CODE_BLOCK)}') AS has_code_block,
      regexp_matches(rep.inner_text, '{_q(STACK_TRACE)}') AS has_stack_trace,
      regexp_matches(rep.inner_text, '{_q(FILE_MENTION)}') AS names_file,
      meta.language, meta.category,
      fx.fix_source, fx.fix_files, fx.fix_lines, fx.fix_files_reported, fx.fix_lines_reported
    FROM rep LEFT JOIN meta USING (dataset, bug_id) LEFT JOIN fx USING (bug_id)
    ORDER BY rep.dataset, rep.bug_id"""


# ---------------------------------------------------------------- report


def _dictionary(con: duckdb.DuckDBPyConnection, path: Path) -> list[tuple[str, str]]:
    return [
        (r[0], r[1])
        for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')").fetchall()
    ]


def write_report(cfg: Config, summary: dict[str, Any], path: Path | None = None) -> Path:
    """Write ``docs/one_try/03_cleaning.md`` from the cleaning summary."""
    fun = summary["funnel"]
    kept = {r["dataset"]: r for r in fun if r["stage"].startswith("Rule 3")}
    first = {r["dataset"]: r for r in fun if r["stage"] == "All extracted tries"}
    total_in = sum(r["tries"] for r in first.values())
    total_kept = sum(r["tries"] for r in kept.values())
    lines = [
        "# Step 3: Data Cleaning",
        "",
        "*Generated by `agent-tips onetry clean`. Paper 2, the one-try study.*",
        "",
        "## In plain words",
        "",
        f"- We started with {total_in:,} recorded tries and kept {total_kept:,} "
        f"({total_kept / total_in:.0%}). Every removed try is explained by one of three written rules below.",
        "- We also cleaned the bug reports: each AI framework wraps the real bug report in its own "
        "instructions. We cut the wrapper off so only the report itself remains "
        f"(found in {summary['wrapper_rate']:.1%} of bugs).",
        '- For every try we recorded the AI\'s choices (for example "test first" or "fix first") '
        "with the labeler validated in Paper 1.",
        "- Nothing about the study's results is computed here.",
        "",
        "## Rules",
        "",
        "| Rule | What it removes | Why |",
        "| --- | --- | --- |",
        "| 1. Duplicate try IDs | Second copies of the same try | A try must count once |",
        "| 2. Outcome known | Tries without a clear fixed / not-fixed result (NVIDIA marks some as unknown) | The outcome is what we measure |",
        "| 3. Identical bug text | Tries that saw a different version of the bug text than the most common one for that AI setup and bug | Tries of the same bug must have the same input, otherwise they are not repeats |",
        "",
        "## Funnel",
        "",
        "| Dataset | Stage | Tries | Bugs | Removed | Fix rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in fun:
        rem = "—" if r["removed"] is None else f"{r['removed']:,}"
        fr = "—" if r["fix_rate"] is None else f"{r['fix_rate']:.1%}"
        lines.append(
            f"| `{r['dataset']}` | {r['stage']} | {r['tries']:,} | {r['bugs']:,} | {rem} | {fr} |"
        )
    lines += [
        "",
        "## Flags (kept, not removed)",
        "",
        "| Flag | Tries flagged | Meaning |",
        "| --- | --- | --- |",
    ]
    fl = summary["flags"]
    lines += [
        f"| Studied setup | {fl['studied']:,} | Tries of the AI setups we study (the older AI's 8B and 405B models are kept only as other AIs: {fl['other_only']:,} tries) |",
        f"| Step-count outlier | {fl['steps_outlier']:,} | Zero actions, or more actions than 99.9% of tries in that dataset; checked in EDA |",
        f"| Older AI evaluation day | {fl['dated']:,} dated, {fl['undated']:,} undated | Hidden model-version proxy (Paper 1); used for the older AI's answer key |",
        f"| Bug report wrapper not found | {fl['no_wrapper']:,} bugs | The full prompt is kept as the report for these |",
        "",
        "## Human-fix size: check against NVIDIA's own counts",
        "",
        f"For {summary['fix_check']['n']:,} NVIDIA bugs, NVIDIA reports how many files the human fix changed. "
        f"Our count from the patch agrees for {summary['fix_check']['files_agree']:.1%} of them.",
        "",
        "## AI setups",
        "",
        "| Dataset | AI setup | Studied | Tries | Bugs | Fix rate | Tries per bug (median) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summary["setups"]:
        lines.append(
            f"| `{s['dataset']}` | `{s['setup']}` | {'yes' if s['studied'] else 'other AI only'} | "
            f"{s['tries']:,} | {s['bugs']:,} | {s['fix_rate']:.1%} | {s['median_tries']:.0f} |"
        )
    lines += [
        "",
        "## Behaviour choices recorded per try",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
    ]
    lines += [f"| `{k}` | {v} |" for k, v in CHOICE_COLUMNS.items()]
    lines += ["", "## Data dictionary", ""]
    for table, cols in summary["dictionary"].items():
        lines += [f"**`{table}`**", "", "| Column | Type |", "| --- | --- |"]
        lines += [f"| `{c}` | {t} |" for c, t in cols]
        lines.append("")
    path = path or cfg.root / "docs" / "one_try" / "03_cleaning.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def run(cfg: Config, log: Callable[[str], None] = print) -> Path:
    """Apply the cleaning rules, build the clean tables and write the report."""
    raw, clean, tmp = _paths(cfg)
    clean.mkdir(parents=True, exist_ok=True)
    con = connect(cfg, tmp)
    build_labels(con, raw, clean, log)
    log("clean tries")
    studied_older = cfg.raw["datasets"]["older_ai"]["model_name"]
    tries = clean / "tries.parquet"
    con.execute(
        f"COPY ({tries_sql(raw, clean, studied_older)} ORDER BY try_id) TO '{tries.as_posix()}' (FORMAT parquet)"
    )
    log("clean bugs")
    bugs = clean / "bugs.parquet"
    con.execute(f"COPY ({bugs_sql(raw, clean)}) TO '{bugs.as_posix()}' (FORMAT parquet)")
    setups = clean / "setups.parquet"
    con.execute(f"""COPY (
        WITH per_bug AS (SELECT dataset, setup, studied, bug_id, count(*) AS n
                         FROM read_parquet('{tries.as_posix()}') GROUP BY ALL)
        SELECT p.dataset, p.setup, p.studied, CAST(sum(p.n) AS BIGINT) AS tries, count(*) AS bugs,
               (SELECT avg(y) FROM read_parquet('{tries.as_posix()}') t WHERE t.setup = p.setup) AS fix_rate,
               median(p.n) AS median_tries
        FROM per_bug p GROUP BY ALL ORDER BY p.dataset, p.setup) TO '{setups.as_posix()}' (FORMAT parquet)""")
    t = f"read_parquet('{tries.as_posix()}')"
    b = f"read_parquet('{bugs.as_posix()}')"
    summary: dict[str, Any] = {
        "funnel": funnel(con, raw),
        "flags": {
            "studied": count(con, f"SELECT count(*) FROM {t} WHERE studied"),
            "other_only": count(con, f"SELECT count(*) FROM {t} WHERE NOT studied"),
            "steps_outlier": count(con, f"SELECT count(*) FROM {t} WHERE steps_outlier"),
            "dated": count(
                con, f"SELECT count(*) FROM {t} WHERE dataset = 'older_ai' AND eval_day IS NOT NULL"
            ),
            "undated": count(
                con, f"SELECT count(*) FROM {t} WHERE dataset = 'older_ai' AND eval_day IS NULL"
            ),
            "no_wrapper": count(con, f"SELECT count(*) FROM {b} WHERE NOT wrapper_found"),
        },
        "wrapper_rate": value(con, f"SELECT avg(wrapper_found::INT) FROM {b}"),
        "fix_check": {
            "n": count(con, f"SELECT count(*) FROM {b} WHERE fix_files_reported IS NOT NULL"),
            "files_agree": value(
                con,
                f"SELECT avg((fix_files = fix_files_reported)::INT) FROM {b} "
                "WHERE fix_files_reported IS NOT NULL",
            ),
        },
        "setups": con.execute(f"SELECT * FROM read_parquet('{setups.as_posix()}')")
        .df()
        .to_dict(orient="records"),
        "dictionary": {
            name: _dictionary(con, clean / name)
            for name in ("tries.parquet", "bugs.parquet", "setups.parquet")
        },
    }
    con.close()
    shutil.rmtree(tmp, ignore_errors=True)
    res = cfg.path("onetry_results")
    res.mkdir(parents=True, exist_ok=True)
    (res / "cleaning.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return write_report(cfg, summary)
