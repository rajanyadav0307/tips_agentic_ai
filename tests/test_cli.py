import pytest

from agent_tips.cli import STEPS, main


def test_env_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env"]) == 0
    assert "device" in capsys.readouterr().out


@pytest.mark.parametrize("command", sorted(STEPS))
def test_unbuilt_steps_exit_with_code_2(command: str) -> None:
    assert main([command]) == 2


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
