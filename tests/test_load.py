from agent_tips.config import Config
from agent_tips.data.load import connect, has_files, register_raw_views, row_counts


def test_all_views_register_and_count(fake_project: Config) -> None:
    con = connect()
    views = register_raw_views(con, fake_project)
    assert set(views) == {
        "raw_older_ai",
        "raw_newer_ai_nebius",
        "raw_newer_ai_nvidia",
        "raw_gold",
    }
    assert row_counts(con, views) == {
        "raw_older_ai": 3,
        "raw_newer_ai_nebius": 2,
        "raw_newer_ai_nvidia": 3,
        "raw_gold": 2,
    }


def test_nvidia_path_parts_are_parsed(fake_project: Config) -> None:
    con = connect()
    register_raw_views(con, fake_project)
    row = con.execute(
        "SELECT DISTINCT scaffold, teacher, task_source FROM raw_newer_ai_nvidia"
    ).fetchall()
    assert row == [("sweagent", "minimax_m25", "swe-rebench-v2")]


def test_gold_sources(fake_project: Config) -> None:
    con = connect()
    register_raw_views(con, fake_project)
    got = dict(con.execute("SELECT instance_id, source FROM raw_gold").fetchall())
    assert got == {"a__b-1": "extra", "c__d-2": "dev"}


def test_missing_dataset_is_skipped(fake_project: Config) -> None:
    for p in (fake_project.path("raw") / "newer_ai_nebius").glob("*.parquet"):
        p.unlink()
    assert not has_files(fake_project, "newer_ai_nebius")
    assert "raw_newer_ai_nebius" not in register_raw_views(connect(), fake_project)
