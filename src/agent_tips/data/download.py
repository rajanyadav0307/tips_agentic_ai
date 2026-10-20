"""Step 2: download pinned dataset files from Hugging Face and verify checksums.

Every file is pinned to a dataset revision (git commit on the Hub) and checked
against the SHA-256 that the Hub publishes for it. A manifest
(``data/raw/MANIFEST.json``) records what was downloaded, so re-running skips
files that are already verified.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_tips.config import Config

CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class RemoteFile:
    """A file in a Hugging Face dataset repository at a pinned revision."""

    dataset: str
    repo_id: str
    revision: str
    path: str
    size: int
    sha256: str


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def dataset_specs(cfg: Config, only: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    """Return the dataset sections of the config, optionally filtered by key."""
    specs: dict[str, dict[str, Any]] = cfg.raw["datasets"]
    keys = set(only) if only else set(specs)
    unknown = keys - set(specs)
    if unknown:
        raise KeyError(f"Unknown dataset key(s): {sorted(unknown)}")
    return {k: v for k, v in specs.items() if k in keys}


def list_remote_files(
    cfg: Config, only: Iterable[str] | None = None, api: Any | None = None
) -> list[RemoteFile]:
    """List the pinned files to download, with their sizes and SHA-256 checksums."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    out: list[RemoteFile] = []
    for key, spec in dataset_specs(cfg, only).items():
        entries = api.list_repo_tree(
            spec["hf_id"], repo_type="dataset", revision=spec["revision"], recursive=True
        )
        patterns: list[str] = spec["patterns"]
        for e in entries:
            lfs = getattr(e, "lfs", None)
            if lfs is None or not any(fnmatch.fnmatch(e.path, p) for p in patterns):
                continue
            size = int(getattr(e, "size", 0))
            out.append(RemoteFile(key, spec["hf_id"], spec["revision"], e.path, size, lfs.sha256))
    return sorted(out, key=lambda f: (f.dataset, f.path))


def local_path(cfg: Config, f: RemoteFile) -> Path:
    """Where a remote file is stored locally: ``data/raw/<dataset>/<path in repo>``."""
    return cfg.path("raw") / f.dataset / Path(f.path)


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if path.is_file():
        data: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return data
    return {}


def _save_manifest(path: Path, manifest: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _is_verified(entry: dict[str, Any] | None, f: RemoteFile, dest: Path) -> bool:
    """A file is verified if the manifest has it with the expected checksum and
    the file on disk still has the size and modification time recorded then."""
    if not entry or not dest.is_file():
        return False
    st = dest.stat()
    return bool(
        entry.get("sha256") == f.sha256
        and entry.get("size") == st.st_size == f.size
        and entry.get("mtime_ns") == st.st_mtime_ns
    )


def download(
    cfg: Config,
    only: Iterable[str] | None = None,
    verify: bool = False,
    dry_run: bool = False,
    api: Any | None = None,
    fetch: Callable[[RemoteFile, Path], Path] | None = None,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Download (or verify) all pinned files and update the manifest.

    Parameters
    ----------
    cfg
        Project config.
    only
        Dataset keys to process (default: all in the config).
    verify
        Re-hash files even if the manifest says they are verified.
    dry_run
        Only report what would be downloaded.
    api, fetch
        Injection points for tests (Hub listing client and file fetcher).
    log
        Where progress messages go.

    Returns
    -------
    list of Path
        Local paths of all files (downloaded or already present).

    Raises
    ------
    ValueError
        If a file's checksum does not match the Hub's after download.
    """
    fetch = fetch or _hf_fetch
    files = list_remote_files(cfg, only, api)
    manifest_path = cfg.path("raw") / "MANIFEST.json"
    manifest = _load_manifest(manifest_path)
    total = sum(f.size for f in files)
    log(f"{len(files)} files, {total / 1e9:.2f} GB in total")
    paths: list[Path] = []
    for f in files:
        dest = local_path(cfg, f)
        key = f"{f.dataset}/{f.path}"
        paths.append(dest)
        if not verify and _is_verified(manifest.get(key), f, dest):
            continue
        if dest.is_file() and dest.stat().st_size == f.size and sha256_of(dest) == f.sha256:
            log(f"verified existing {key}")
        elif dry_run:
            log(f"would download {key} ({f.size / 1e6:.1f} MB)")
            continue
        else:
            log(f"downloading {key} ({f.size / 1e6:.1f} MB)")
            got = fetch(f, dest)
            if got.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(got, dest)
            if sha256_of(dest) != f.sha256:
                dest.unlink()
                raise ValueError(f"Checksum mismatch for {key}; file removed")
        st = dest.stat()
        manifest[key] = {
            "repo_id": f.repo_id,
            "revision": f.revision,
            "path": f.path,
            "size": st.st_size,
            "sha256": f.sha256,
            "mtime_ns": st.st_mtime_ns,
        }
        _save_manifest(manifest_path, manifest)
    return paths


def _hf_fetch(f: RemoteFile, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    root = dest.parents[len(Path(f.path).parts) - 1]
    got = hf_hub_download(
        repo_id=f.repo_id,
        filename=f.path,
        repo_type="dataset",
        revision=f.revision,
        local_dir=root,
    )
    return Path(got)
