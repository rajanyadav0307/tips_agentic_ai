"""Load project settings from ``configs/*.yaml`` and resolve paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent directory that contains ``pyproject.toml``.

    Parameters
    ----------
    start
        Directory to start searching from. Defaults to the current directory.

    Raises
    ------
    FileNotFoundError
        If no ``pyproject.toml`` is found up to the filesystem root.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"No pyproject.toml found above {here}")


@dataclass(frozen=True)
class Config:
    """Project settings with paths resolved against the project root."""

    root: Path
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        """Global random seed."""
        return int(self.raw.get("seed", 0))

    def path(self, name: str) -> Path:
        """Return an absolute path from the ``paths`` section, e.g. ``path("results")``."""
        return self.root / Path(self.raw["paths"][name])


def load_config(path: Path | None = None) -> Config:
    """Load a YAML config (``configs/default.yaml`` by default)."""
    root = find_project_root()
    cfg_path = path or root / "configs" / "default.yaml"
    with cfg_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path} must contain a mapping")
    return Config(root=root, raw=data)
