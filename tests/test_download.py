import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_tips.config import Config
from agent_tips.data.download import RemoteFile, download

CONTENT = {"data/a.parquet": b"alpha", "data/b.parquet": b"bravo", "README.md": b"card"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FakeApi:
    def __init__(self, bad: str | None = None) -> None:
        self.bad = bad

    def list_repo_tree(self, repo_id: str, **_: object) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(path=p, size=len(b), lfs=SimpleNamespace(sha256=_sha(b)))
            for p, b in CONTENT.items()
        ]


def make_fetch(calls: list[str], corrupt: bool = False):  # type: ignore[no-untyped-def]
    def fetch(f: RemoteFile, dest: Path) -> Path:
        calls.append(f.path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = CONTENT[f.path]
        dest.write_bytes(data[::-1] if corrupt else data)
        return dest

    return fetch


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    raw = {
        "paths": {"raw": "data/raw"},
        "datasets": {"toy": {"hf_id": "o/toy", "revision": "abc", "patterns": ["data/*.parquet"]}},
    }
    return Config(root=tmp_path, raw=raw)


def test_downloads_only_matching_files_and_writes_manifest(cfg: Config) -> None:
    calls: list[str] = []
    paths = download(cfg, api=FakeApi(), fetch=make_fetch(calls), log=lambda _: None)
    assert sorted(calls) == ["data/a.parquet", "data/b.parquet"]
    assert all(p.is_file() for p in paths)
    manifest = json.loads((cfg.path("raw") / "MANIFEST.json").read_text())
    assert manifest["toy/data/a.parquet"]["sha256"] == _sha(b"alpha")
    assert manifest["toy/data/a.parquet"]["revision"] == "abc"


def test_second_run_downloads_nothing(cfg: Config) -> None:
    calls: list[str] = []
    download(cfg, api=FakeApi(), fetch=make_fetch(calls), log=lambda _: None)
    calls.clear()
    download(cfg, api=FakeApi(), fetch=make_fetch(calls), log=lambda _: None)
    assert calls == []


def test_corrupted_download_is_rejected(cfg: Config) -> None:
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(cfg, api=FakeApi(), fetch=make_fetch([], corrupt=True), log=lambda _: None)


def test_changed_file_is_downloaded_again(cfg: Config) -> None:
    calls: list[str] = []
    paths = download(cfg, api=FakeApi(), fetch=make_fetch(calls), log=lambda _: None)
    paths[0].write_bytes(b"tampered")
    calls.clear()
    download(cfg, api=FakeApi(), fetch=make_fetch(calls), log=lambda _: None)
    assert calls == ["data/a.parquet"]


def test_dry_run_does_not_download(cfg: Config) -> None:
    calls: list[str] = []
    download(cfg, api=FakeApi(), fetch=make_fetch(calls), dry_run=True, log=lambda _: None)
    assert calls == []


def test_unknown_dataset_key(cfg: Config) -> None:
    with pytest.raises(KeyError):
        download(cfg, only=["nope"], api=FakeApi(), fetch=make_fetch([]), log=lambda _: None)
