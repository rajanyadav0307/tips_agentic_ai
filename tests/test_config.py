from pathlib import Path

import pytest

from agent_tips.config import find_project_root, load_config


def test_project_root_has_pyproject() -> None:
    assert (find_project_root() / "pyproject.toml").is_file()


def test_default_config_loads_with_absolute_paths() -> None:
    cfg = load_config()
    assert cfg.seed == 0
    results = cfg.path("results")
    assert isinstance(results, Path)
    assert results.is_absolute()
    assert results.parts[-2:] == ("outputs", "results")


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_project_root(tmp_path)
