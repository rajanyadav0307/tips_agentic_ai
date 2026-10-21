"""Shared DuckDB settings for the one-try study (bounded memory, on-disk spilling)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb

from agent_tips.config import Config


def connect(cfg: Config, tmp: Path) -> duckdb.DuckDBPyConnection:
    """Open a connection that caps memory and spills large jobs to ``tmp``."""
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET memory_limit = '{cfg.raw['onetry'].get('memory_limit', '8GB')}'")
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    con.execute(f"SET threads = {int(cfg.raw['onetry'].get('threads', 4))}")
    return con


def sql_list(paths: Iterable[str | Path]) -> str:
    """Return a SQL list literal of file paths or globs (POSIX style)."""
    items = [p.as_posix() if isinstance(p, Path) else p for p in paths]
    return "[" + ", ".join("'" + s.replace("'", "''") + "'" for s in items) + "]"


def count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Run a single-number query (e.g. ``SELECT count(*) ...``)."""
    row = con.execute(sql).fetchone()
    return int(row[0]) if row else 0


def value(con: duckdb.DuckDBPyConnection, sql: str) -> float:
    """Run a single-value query and return it as a float (0.0 if empty)."""
    row = con.execute(sql).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0
