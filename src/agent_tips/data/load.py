"""Step 2: expose the downloaded raw files as DuckDB views.

Views (one per dataset key in the config):

- ``raw_older_ai``: nebius/SWE-agent-trajectories rows, plus ``filename``.
- ``raw_newer_ai_nebius``: nebius/SWE-rebench-openhands-trajectories rows.
- ``raw_newer_ai_nvidia``: nvidia/Open-SWE-Traces rows, plus ``scaffold``,
  ``teacher`` and ``task_source`` parsed from the file path.
- ``raw_gold``: gold patches with ``instance_id``, ``patch``, ``created_at``
  and ``source`` (``extra`` or ``dev``).

DuckDB reads Parquet lazily, so creating views is cheap; queries read only the
columns they need.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from agent_tips.config import Config

NVIDIA_PATH_RE = r"data/(sweagent|openhands|minisweagent)/([^/]+)/([^/]+)/"


def connect(threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection (optionally limiting threads)."""
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    return con


def raw_globs(cfg: Config, key: str) -> list[str]:
    """Return the local glob patterns (POSIX style) for one dataset key."""
    base = cfg.path("raw") / key
    patterns: list[str] = cfg.raw["datasets"][key]["patterns"]
    return [(base / p).as_posix() for p in patterns]


def has_files(cfg: Config, key: str) -> bool:
    """True if at least one local file exists for the dataset key."""
    base = cfg.path("raw") / key
    patterns: list[str] = cfg.raw["datasets"][key]["patterns"]
    return any(any(base.glob(p)) for p in patterns)


def _sql_list(items: list[str]) -> str:
    return "[" + ", ".join("'" + s.replace("'", "''") + "'" for s in items) + "]"


def register_raw_views(con: duckdb.DuckDBPyConnection, cfg: Config) -> list[str]:
    """Create one view per available dataset and return the view names."""
    created: list[str] = []
    for key in ("older_ai", "newer_ai_nebius"):
        if has_files(cfg, key):
            con.execute(
                f"CREATE OR REPLACE VIEW raw_{key} AS SELECT * FROM read_parquet("
                f"{_sql_list(raw_globs(cfg, key))}, filename = true)"
            )
            created.append(f"raw_{key}")
    if has_files(cfg, "newer_ai_nvidia"):
        # On Windows DuckDB may report file names with backslashes; normalize first.
        fname = "replace(filename, '\\', '/')"
        con.execute(
            "CREATE OR REPLACE VIEW raw_newer_ai_nvidia AS SELECT *, "
            f"regexp_extract({fname}, '{NVIDIA_PATH_RE}', 1) AS scaffold, "
            f"regexp_extract({fname}, '{NVIDIA_PATH_RE}', 2) AS teacher, "
            f"regexp_extract({fname}, '{NVIDIA_PATH_RE}', 3) AS task_source "
            f"FROM read_parquet({_sql_list(raw_globs(cfg, 'newer_ai_nvidia'))}, "
            "filename = true, union_by_name = true)"
        )
        created.append("raw_newer_ai_nvidia")
    gold_parts = [
        f"SELECT instance_id, patch, CAST(created_at AS VARCHAR) AS created_at, '{src}' AS source "
        f"FROM read_parquet({_sql_list(raw_globs(cfg, key))})"
        for key, src in (("gold_extra", "extra"), ("gold_dev", "dev"))
        if has_files(cfg, key)
    ]
    if gold_parts:
        con.execute("CREATE OR REPLACE VIEW raw_gold AS " + " UNION ALL ".join(gold_parts))
        created.append("raw_gold")
    return created


def row_counts(con: duckdb.DuckDBPyConnection, views: list[str]) -> dict[str, int]:
    """Count rows in each view."""
    counts: dict[str, int] = {}
    for v in views:
        row = con.execute(f"SELECT count(*) FROM {v}").fetchone()
        counts[v] = int(row[0]) if row else 0
    return counts


def as_posix(path: Path) -> str:
    """Path as a forward-slash string, as DuckDB expects on every platform."""
    return path.as_posix()
