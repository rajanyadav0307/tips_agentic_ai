"""Step 2 of the one-try study: verification and extraction."""

import json

import duckdb

from agent_tips.config import Config
from agent_tips.data.download import sha256_of
from agent_tips.onetry import collect


def _quiet(_: str) -> None:
    pass


def test_verify_checks_revision_size_and_hash(fake_project: Config) -> None:
    files = collect.local_files(fake_project, "older_ai")
    assert len(files) == 1
    f = files[0]
    rel = f.relative_to(fake_project.path("raw") / "older_ai").as_posix()
    spec = fake_project.raw["datasets"]["older_ai"]
    entry = {
        "repo_id": spec["hf_id"],
        "revision": spec["revision"],
        "size": f.stat().st_size,
        "sha256": sha256_of(f),
    }
    manifest = fake_project.path("raw") / "MANIFEST.json"

    def check(e: dict[str, object]) -> bool:
        manifest.write_text(json.dumps({f"older_ai/{rel}": e}), encoding="utf-8")
        return collect.verify(fake_project, ["older_ai"], log=_quiet)[0]["verified"]

    assert check(entry)
    assert not check({**entry, "revision": "other"})
    assert not check({**entry, "sha256": "0" * 64})
    manifest.unlink()
    assert collect.verify(fake_project, ["older_ai"], log=_quiet)[0]["in_manifest"] is False


def test_extract_matches_sources_and_writes_report(big_project: Config) -> None:
    report = collect.run(big_project, log=_quiet)
    raw = big_project.path("onetry_data") / "raw"
    summary = json.loads(
        (big_project.path("onetry_results") / "collection.json").read_text(encoding="utf-8")
    )
    assert summary["extraction"]["rows_match"]
    manifest = json.loads((raw / "MANIFEST.json").read_text(encoding="utf-8"))
    assert {"tries.parquet", "reports.parquet", "fixes.parquet"} <= set(manifest)
    assert "steps/older_ai.parquet" in manifest
    assert "steps/newer_ai_nvidia/sweagent__minimax_m25__swe-rebench-v2.parquet" in manifest
    con = duckdb.connect()
    setups = {
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT setup FROM read_parquet('{(raw / 'tries.parquet').as_posix()}')"
        ).fetchall()
    }
    assert "swe-agent-llama-70b" in setups and "newer_ai_nebius" in setups
    assert "sweagent/minimax_m25/swe-rebench-v2" in setups
    # one report text per bug in these fixtures (every try of a bug sees the same text)
    n_reports = con.execute(
        f"SELECT count(*) FROM read_parquet('{(raw / 'reports.parquet').as_posix()}')"
    ).fetchone()
    assert n_reports is not None and n_reports[0] > 0
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Step 2: Data Collection")
    assert "| `older_ai` |" in text
