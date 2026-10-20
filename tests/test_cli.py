import pytest

from agent_tips.cli import build_parser, main
from agent_tips.config import Config


def test_env_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env"]) == 0
    assert "device" in capsys.readouterr().out


def test_every_step_has_a_command() -> None:
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    steps = {"download", "clean", "eda", "features", "model", "evaluate", "figures", "run-all"}
    assert steps <= set(sub.choices)  # type: ignore[arg-type]


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_run_all_on_big_fixture(
    big_project: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``run-all`` runs every step in order (download is stubbed: the raw files exist)."""
    import warnings

    from agent_tips import cli

    monkeypatch.setattr(cli, "load_config", lambda: big_project)
    monkeypatch.setattr(cli, "_download", lambda _: 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(["run-all"]) == 0
    out = capsys.readouterr().out
    for step in ("clean", "eda", "features", "model", "evaluate", "figures"):
        assert f"== {step}" in out
    rob = (big_project.path("results") / "robustness.json").read_text(encoding="utf-8")
    assert '"identical": true' in rob
