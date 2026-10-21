"""Step 2 (one-try study): data collection.

1. Verify the pinned downloads this study needs against ``data/raw/MANIFEST.json``
   (repository, revision, size and SHA-256).
2. Extract only the fields this study needs into ``data/onetry/raw/``:

   - ``tries.parquet``: one row per try of every AI setup (all three older-AI
     model sizes included, because the smaller and larger ones serve as
     "other AIs");
   - ``steps/<dataset>.parquet``: one row per action, for the studied setups;
   - ``reports.parquet``: one row per distinct bug-report text;
   - ``fixes.parquet``: the human fix (gold or reference patch) per bug.

3. Write ``MANIFEST.json`` for the extracts, ``outputs/onetry/collection.json``
   and the report ``docs/one_try/02_data_collection.md``.

The SQL that reads each raw format is shared with Paper 1
(:mod:`agent_tips.cleaning.sources`); every table here is rebuilt from raw.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import duckdb

from agent_tips.cleaning import sources
from agent_tips.config import Config
from agent_tips.data.download import sha256_of
from agent_tips.data.load import has_files, raw_globs

TRAJECTORIES = ("older_ai", "newer_ai_nebius", "newer_ai_nvidia")
GOLD = ("gold_extra", "gold_dev")
LICENCES = {
    "older_ai": "CC BY 4.0 (model outputs also under the Llama 3.1 license)",
    "newer_ai_nebius": "CC BY 4.0",
    "newer_ai_nvidia": "CC BY 4.0",
    "gold_extra": "CC BY 4.0",
    "gold_dev": "No license on the card; SWE-bench is MIT-licensed on GitHub",
}
WHY = {
    "older_ai": "Studied AI (70B, 13 tries per bug); 8B and 405B serve as other AIs",
    "newer_ai_nebius": "Studied AI (Qwen3-Coder, 10 tries per bug, failures kept); other AI for the older AI",
    "newer_ai_nvidia": "Seven studied AI setups; other AIs for each other",
    "gold_extra": "Human fixes for the older AI's bugs",
    "gold_dev": "Human fixes for the older AI's bugs",
}

SETUP_SQL = """CASE dataset WHEN 'older_ai' THEN ai WHEN 'newer_ai_nebius' THEN 'newer_ai_nebius'
               ELSE subset END"""
TRY_COLUMNS = (
    "try_id, dataset, " + SETUP_SQL + " AS setup, ai, scaffold, subset, issue_id, issue_hash, "
    "resolved, outcome_known, exit_status, n_steps, has_patch, eval_ts, language, chars_in, chars_out"
)


def _src(globs: Iterable[str]) -> str:
    return "[" + ", ".join("'" + g.replace("'", "''") + "'" for g in globs) + "]"


# ---------------------------------------------------------------- verification


def local_files(cfg: Config, key: str) -> list[Path]:
    """Local raw files of one dataset key that match its configured patterns."""
    base = cfg.path("raw") / key
    patterns: list[str] = cfg.raw["datasets"][key]["patterns"]
    found = {p for pat in patterns for p in base.glob(pat) if p.is_file()}
    return sorted(found)


def verify(
    cfg: Config, keys: Iterable[str], rehash: bool = True, log: Callable[[str], None] = print
) -> list[dict[str, Any]]:
    """Check every local file against the download manifest.

    A file passes if the manifest has it at the configured repository and
    revision with the same size and (when ``rehash``) the same SHA-256.
    """
    manifest_path = cfg.path("raw") / "MANIFEST.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    rows = []
    for key in keys:
        spec = cfg.raw["datasets"][key]
        files = local_files(cfg, key)
        log(f"verify {key}: {len(files)} files")
        for f in files:
            rel = f.relative_to(cfg.path("raw") / key).as_posix()
            entry = manifest.get(f"{key}/{rel}")
            size = f.stat().st_size
            ok_entry = bool(
                entry
                and entry.get("repo_id") == spec["hf_id"]
                and entry.get("revision") == spec["revision"]
                and entry.get("size") == size
            )
            ok_hash = ok_entry and (not rehash or sha256_of(f) == entry["sha256"])  # type: ignore[index]
            rows.append(
                {
                    "dataset": key,
                    "file": rel,
                    "bytes": size,
                    "in_manifest": entry is not None,
                    "verified": bool(ok_hash),
                }
            )
    return rows


# ---------------------------------------------------------------- extraction


def _tries_sql(cfg: Config, key: str, globs: list[str]) -> str:
    src = _src(globs)
    if key == "older_ai":
        sql = sources.older_ai_tries(src, cfg.raw["datasets"]["older_ai"]["model_name"])
    elif key == "newer_ai_nebius":
        sql = sources.nebius_tries(src)
    else:
        sql = sources.nvidia_tries(src)
    return f"SELECT {TRY_COLUMNS} FROM ({sql})"


def _steps_sql(cfg: Config, key: str, globs: list[str]) -> str:
    src = _src(globs)
    if key == "older_ai":
        return sources.older_ai_steps(src, cfg.raw["datasets"]["older_ai"]["model_name"])
    if key == "newer_ai_nebius":
        return sources.nebius_steps(src)
    return sources.nvidia_steps(src)


def _reports_sql(key: str, globs: list[str]) -> str:
    """One row per distinct bug-report text (the agent's first user message)."""
    src = _src(globs)
    if key == "older_ai":
        text = "coalesce(trajectory[2].text, '')"
        frm = f"read_parquet({src})"
    elif key == "newer_ai_nebius":
        text = "coalesce(trajectory[2].content, '')"
        frm = f"read_parquet({src})"
    else:
        text = "coalesce(list_filter(messages, m -> m.role = 'user')[1].content, '')"
        frm = f"read_parquet({src}, union_by_name = true)"
    return f"""
      SELECT '{key}' AS dataset, instance_id AS issue_id, md5({text}) AS issue_hash,
             any_value({text}) AS text, count(*) AS n_tries
      FROM {frm} GROUP BY ALL"""


def _fixes_sql(cfg: Config) -> str:
    parts = []
    for key in GOLD:
        if has_files(cfg, key):
            parts.append(
                f"SELECT '{key}' AS source, instance_id AS issue_id, any_value(patch) AS patch, "
                "CAST(NULL AS INTEGER) AS files_reported, CAST(NULL AS INTEGER) AS lines_reported "
                f"FROM read_parquet({_src(raw_globs(cfg, key))}) GROUP BY ALL"
            )
    return " UNION ALL ".join(parts)


def _nvidia_fixes_sql(glob: str) -> str:
    return (
        "SELECT 'newer_ai_nvidia' AS source, instance_id AS issue_id, "
        "any_value(metadata.reference_patch.patch) AS patch, "
        "CAST(any_value(metadata.reference_patch.num_modified_files) AS INTEGER) AS files_reported, "
        "CAST(any_value(metadata.reference_patch.num_modified_lines) AS INTEGER) AS lines_reported "
        f"FROM read_parquet({_src([glob])}, union_by_name = true) GROUP BY ALL"
    )


def _part_name(glob: str) -> str:
    """``.../data/sweagent/qwen36_27b/scale-swe/*.parquet`` -> ``sweagent__qwen36_27b__scale-swe``."""
    parts = glob.replace("\\", "/").split("/")
    return "__".join(parts[-4:-1])


def _count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    return int(row[0]) if row else 0


def _source_rows(con: duckdb.DuckDBPyConnection, cfg: Config, key: str) -> int:
    opts = ", union_by_name = true" if key == "newer_ai_nvidia" else ""
    sql = f"SELECT count(*) FROM read_parquet({_src(raw_globs(cfg, key))}{opts})"
    return _count(con, sql)


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def extract(cfg: Config, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Write the study's raw extracts and return their row counts."""
    out = cfg.path("onetry_data") / "raw"
    (out / "steps").mkdir(parents=True, exist_ok=True)
    tmp = cfg.path("onetry_data") / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.execute("SET preserve_insertion_order = false")
    # Keep memory bounded on a laptop: spill sorts to disk instead of filling RAM.
    con.execute(f"SET memory_limit = '{cfg.raw['onetry'].get('memory_limit', '8GB')}'")
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    con.execute(f"SET threads = {int(cfg.raw['onetry'].get('threads', 4))}")
    present = [k for k in TRAJECTORIES if has_files(cfg, k)]
    files: dict[str, Path] = {}

    # Big sources are processed one AI setup (one raw glob) at a time to bound memory.
    try_parts: list[Path] = []
    report_parts: list[Path] = []
    fix_parts: list[Path] = []
    for k in present:
        globs = raw_globs(cfg, k)
        pieces = [[g] for g in globs] if k == "newer_ai_nvidia" else [globs]
        for i, piece in enumerate(pieces):
            name = f"{k}/{_part_name(piece[0])}" if k == "newer_ai_nvidia" else k
            log(f"extract tries: {name}")
            tp = tmp / f"tries_{k}_{i}.parquet"
            con.execute(f"COPY ({_tries_sql(cfg, k, piece)}) TO '{tp.as_posix()}' (FORMAT parquet)")
            try_parts.append(tp)
            log(f"extract steps: {name}")
            p = out / "steps" / f"{name}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            con.execute(
                f"COPY (SELECT * FROM ({_steps_sql(cfg, k, piece)}) ORDER BY try_id, step) "
                f"TO '{p.as_posix()}' (FORMAT parquet, ROW_GROUP_SIZE 500000)"
            )
            files[f"steps/{name}.parquet"] = p
            log(f"extract bug reports: {name}")
            rp = tmp / f"reports_{k}_{i}.parquet"
            con.execute(f"COPY ({_reports_sql(k, piece)}) TO '{rp.as_posix()}' (FORMAT parquet)")
            report_parts.append(rp)
            if k == "newer_ai_nvidia":
                fp = tmp / f"fixes_{i}.parquet"
                con.execute(
                    f"COPY ({_nvidia_fixes_sql(piece[0])}) TO '{fp.as_posix()}' (FORMAT parquet)"
                )
                fix_parts.append(fp)
    log("combine tries, bug reports and human fixes")
    files["tries.parquet"] = out / "tries.parquet"
    con.execute(
        f"COPY (SELECT * FROM read_parquet({_src([p.as_posix() for p in try_parts])}, "
        f"union_by_name = true) ORDER BY try_id) TO '{files['tries.parquet'].as_posix()}' "
        "(FORMAT parquet)"
    )
    files["reports.parquet"] = out / "reports.parquet"
    con.execute(
        f"COPY (SELECT dataset, issue_id, issue_hash, any_value(text) AS text, "
        f"sum(n_tries) AS n_tries FROM read_parquet({_src([p.as_posix() for p in report_parts])}) "
        "GROUP BY ALL ORDER BY dataset, issue_id, issue_hash) "
        f"TO '{files['reports.parquet'].as_posix()}' (FORMAT parquet)"
    )
    fixes = [f for f in [_fixes_sql(cfg)] if f]
    if fix_parts:
        fixes.append(
            "SELECT source, issue_id, any_value(patch) AS patch, any_value(files_reported) AS "
            "files_reported, any_value(lines_reported) AS lines_reported FROM read_parquet("
            f"{_src([p.as_posix() for p in fix_parts])}) GROUP BY ALL"
        )
    if fixes:
        files["fixes.parquet"] = out / "fixes.parquet"
        con.execute(
            f"COPY (SELECT * FROM ({' UNION ALL '.join(fixes)}) ORDER BY source, issue_id) "
            f"TO '{files['fixes.parquet'].as_posix()}' (FORMAT parquet)"
        )
    for p in [*try_parts, *report_parts, *fix_parts]:
        p.unlink()

    manifest = {}
    for name, p in files.items():
        rows = _count(con, f"SELECT count(*) FROM read_parquet('{p.as_posix()}')")
        manifest[name] = {"rows": rows, "bytes": p.stat().st_size, "sha256": _file_digest(p)}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)  # DuckDB spill files and temporary parts

    by_ds = con.execute(
        f"SELECT dataset, setup, count(*) AS tries, count(DISTINCT issue_id) AS bugs "
        f"FROM read_parquet('{files['tries.parquet'].as_posix()}') GROUP BY ALL ORDER BY ALL"
    ).df()
    source_rows = {k: _source_rows(con, cfg, k) for k in present}
    extracted = {str(ds): int(n) for ds, n in by_ds.groupby("dataset")["tries"].sum().items()}
    return {
        "manifest": manifest,
        "setups": by_ds.to_dict(orient="records"),
        "source_rows": source_rows,
        "extracted_rows": extracted,
        "rows_match": all(source_rows[k] == extracted.get(k) for k in present),
    }


# ---------------------------------------------------------------- report


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def write_report(cfg: Config, summary: dict[str, Any], path: Path | None = None) -> Path:
    """Write ``docs/one_try/02_data_collection.md`` from the collection summary."""
    v = summary["verification"]
    ex = summary["extraction"]
    lines = [
        "# Step 2: Data Collection",
        "",
        "*Generated by `agent-tips onetry collect`. Paper 2, the one-try study.*",
        "",
        "## In plain words",
        "",
        "- **Sources:** three public sets of AI coding recordings, plus the human fixes for the bugs.",
        "- **Integrity:** every file was checked against the exact version we pinned, with its "
        "fingerprint (SHA-256): "
        + (
            "**all files passed**."
            if summary["all_verified"]
            else "**some files failed; see below**."
        ),
        "- **Extraction:** only the fields this study needs, into its own folder `data/onetry/raw/`.",
        "- **Row check:** the number of tries extracted matches the source datasets "
        + ("exactly." if ex["rows_match"] else "**not exactly; see below**."),
        "",
        "## Sources",
        "",
        "| Key | Hugging Face dataset | Revision | Licence | Files | Size | Verified | Why this study needs it |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in summary["sources"]:
        spec = cfg.raw["datasets"][key]
        mine = [r for r in v if r["dataset"] == key]
        ok = sum(r["verified"] for r in mine)
        size = sum(r["bytes"] for r in mine)
        lines.append(
            f"| `{key}` | {spec['hf_id']} | `{spec['revision'][:10]}` | {LICENCES[key]} | "
            f"{len(mine)} | {_gb(size)} | {ok}/{len(mine)} | {WHY[key]} |"
        )
    lines += [
        "",
        "## What was extracted",
        "",
        "| File | Rows | Size | SHA-256 (first 12) | Contents |",
        "| --- | --- | --- | --- | --- |",
    ]
    contents = {
        "tries.parquet": "One row per try: bug ID, AI setup, outcome, exit status, steps, patch present, evaluation time (older AI), bug-text fingerprint",
        "reports.parquet": "One row per distinct bug-report text (the AI's first user message)",
        "fixes.parquet": "Human fix per bug: gold patches (older AI) and reference patches (NVIDIA)",
    }
    for name, m in ex["manifest"].items():
        what = contents.get(
            name, "One row per action: tool, command, file path, start and end of the reply"
        )
        lines.append(
            f"| `{name}` | {m['rows']:,} | {_gb(m['bytes'])} | `{m['sha256'][:12]}` | {what} |"
        )
    lines += [
        "",
        "## Row check: source datasets vs extracted tries",
        "",
        "| Dataset | Rows in source | Tries extracted | Match |",
        "| --- | --- | --- | --- |",
    ]
    for k, n in ex["source_rows"].items():
        got = ex["extracted_rows"].get(k, 0)
        lines.append(f"| `{k}` | {n:,} | {got:,} | {'yes' if n == got else 'no'} |")
    lines += [
        "",
        "## AI setups found",
        "",
        "| Dataset | AI setup | Tries | Bugs |",
        "| --- | --- | --- | --- |",
    ]
    for r in ex["setups"]:
        lines.append(f"| `{r['dataset']}` | `{r['setup']}` | {r['tries']:,} | {r['bugs']:,} |")
    lines += [
        "",
        "## Notes for later steps",
        "",
        "- **Older AI:** all three model sizes are kept. Only the 70B model is studied; the 8B and "
        "405B models count as other AIs.",
        "- **Nebius:** no human fixes, because its bugs come from SWE-rebench. Human-fix features "
        "will be missing there (with a flag).",
        "- **NVIDIA:** NVIDIA removed tries that hit the step limit, ended early, had empty patches, "
        "or changed tests (arXiv 2606.16038). This cannot be undone and is a stated limitation.",
        "- **Not cleaned yet:** nothing has been removed so far. Cleaning rules are applied and "
        "counted in Step 3.",
        "",
    ]
    path = path or cfg.root / "docs" / "one_try" / "02_data_collection.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def run(cfg: Config, rehash: bool = True, log: Callable[[str], None] = print) -> Path:
    """Verify sources, extract the study's raw tables and write the report."""
    keys = [k for k in cfg.raw["onetry"]["sources"] if has_files(cfg, k)]
    ver = verify(cfg, keys, rehash=rehash, log=log)
    ext = extract(cfg, log=log)
    summary = {
        "sources": keys,
        "verification": ver,
        "all_verified": bool(ver) and all(r["verified"] for r in ver),
        "extraction": ext,
    }
    res = cfg.path("onetry_results")
    res.mkdir(parents=True, exist_ok=True)
    (res / "collection.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return write_report(cfg, summary)
