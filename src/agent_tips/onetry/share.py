"""Step 7 (one-try study): share the full run across several computers (amendment A9).

The study team has three laptops (``onetry.machines`` in the config). The work
is split by **comparison**: every task of a comparison runs on the same
machine. The split is fixed in advance by :func:`plan` and saved to
``docs/one_try/share_plan.json``.

Flow:

1. ``agent-tips onetry plan``: split the comparisons (the sealed ones stay on
   the machine that made them; the rest by expected cost and machine speed).
2. ``agent-tips onetry bundle --machine NAME``: one zip with the code, the
   study data and the plan, so a partner needs no download and no GitHub
   access.
3. On each machine: ``agent-tips onetry run --machine NAME``.
4. ``agent-tips onetry pack --machine NAME``: one zip of that machine's results
   with SHA-256 hashes; ``agent-tips onetry merge FILE`` checks and adds it.
5. ``agent-tips onetry crosscheck``: re-run a few tasks of every other
   machine here; the numbers must agree within the A9 tolerance.

Every result folder records the machine and the library versions that made it
(``_machine.json``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata as md
import io
import json
import platform
import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_tips.config import Config
from agent_tips.onetry import data
from agent_tips.onetry import evaluate as ev
from agent_tips.onetry.comparisons import NVIDIA

PLAN = Path("docs") / "one_try" / "share_plan.json"
TOLERANCE = 0.001  # A9: 0.1 point, any estimate, across machines
LIBS = ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "threadpoolctl")


def machines(cfg: Config) -> dict[str, dict[str, Any]]:
    """Return the study's computers: ``{name: {speed, jobs, home?}}`` from the config."""
    return dict(cfg.raw["onetry"]["machines"])


def home(cfg: Config) -> str:
    """Return the machine that made the sealed first run (keeps the sealed comparisons)."""
    return next(k for k, v in machines(cfg).items() if v.get("home"))


# ---------------------------------------------------------------- plan


def task_cost(s: data.StudyData, cid: str, key_bugs: int, p: ev.Params) -> float:
    """Estimate the work of one comparison, in bug × model-fit units (for balancing only)."""
    c = next(x for x in s.comparisons if x.cid == cid)
    all_bugs = int(c.rows(s.tries).bug_id.nunique())
    dml = 5 + (3 if c.dataset == NVIDIA else 0) + (1 if c.dataset in ev.MANY_TRIES else 0)
    key = p.draws * (dml + 0.3) + (p.precision_draws - p.draws) * 1.3
    key += p.check_draws * 5.5 + p.forest_draws * 8
    return key_bugs * key + all_bugs * p.secondary_draws * 4.5


def plan(cfg: Config, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Split the comparisons across the machines and save the plan.

    Sealed comparisons go to the home machine. The rest are placed largest
    first on the machine that would finish soonest (load ÷ speed).
    """
    s = ev._study(cfg)
    p = ev.Params.from_config(cfg)
    keys = pd.read_parquet(cfg.path("onetry_results") / "answer_keys.parquet")
    size = dict(zip(keys.cid, keys.key_bugs, strict=True))
    cost = {c.cid: task_cost(s, c.cid, int(size[c.cid]), p) for c in s.comparisons}
    ms = machines(cfg)
    load = dict.fromkeys(ms, 0.0)
    out: dict[str, list[str]] = {m: [] for m in ms}
    sealed = set(ev.sealed_cids(cfg))
    h = home(cfg)
    for cid in sorted(sealed):
        out[h].append(cid)
        load[h] += cost[cid]
    for cid in sorted(set(cost) - sealed, key=lambda x: (-cost[x], x)):
        m = min(ms, key=lambda k: ((load[k] + cost[cid]) / float(ms[k]["speed"]), k))
        out[m].append(cid)
        load[m] += cost[cid]
    total = sum(cost.values())
    share = {m: round(load[m] / total, 3) for m in ms}
    result: dict[str, Any] = {
        "machines": ms,
        "comparisons": {m: sorted(v) for m, v in out.items()},
        "share_of_work": share,
        "relative_finish": {m: round(load[m] / float(ms[m]["speed"]) / total, 3) for m in ms},
    }
    path = cfg.root / PLAN
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for m in ms:
        log(f"{m}: {len(out[m])} comparisons, {share[m]:.0%} of the work")
    return result


def assigned(cfg: Config, machine: str) -> list[str]:
    """Return the comparisons the saved plan gives to ``machine``."""
    path = cfg.root / PLAN
    if not path.is_file():
        raise FileNotFoundError(f"no plan yet: run `agent-tips onetry plan` ({path})")
    got = json.loads(path.read_text(encoding="utf-8"))["comparisons"]
    if machine not in got:
        raise KeyError(f"unknown machine {machine!r}; the plan has {sorted(got)}")
    return list(got[machine])


# ---------------------------------------------------------------- run and record


def machine_info(name: str) -> dict[str, Any]:
    """Which machine and library versions produced a result."""
    vers = {}
    for lib in LIBS:
        try:
            vers[lib] = md.version(lib)
        except md.PackageNotFoundError:
            vers[lib] = "missing"
    return {
        "machine": name,
        "system": platform.system(),
        "release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "libraries": vers,
    }


def run(
    cfg: Config, machine: str, jobs: int | None = None, log: Callable[[str], None] = print
) -> Path:
    """Run this machine's share of the full run (resumable)."""
    p = ev.Params.from_config(cfg)
    jobs = jobs or int(machines(cfg)[machine]["jobs"])
    cids = assigned(cfg, machine)
    info = machine_info(machine)
    for cid in cids:
        folder = cfg.path("onetry_results") / "units" / ev._slug(cid)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "_machine.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"{machine}: {len(cids)} comparisons, {jobs} at a time")
    return ev.run_all(cfg, dataclasses.replace(p, jobs=jobs), only=cids, log=log)


def missing(cfg: Config, cids: list[str]) -> list[str]:
    """Tasks of ``cids`` not yet done, as ``cid part draw`` strings."""
    p = ev.Params.from_config(cfg)
    units = cfg.path("onetry_results") / "units"
    return [
        f"{cid} {part} {draw}"
        for cid in cids
        for part, draw in ev.tasks_of(cfg, cid, p)
        if not (units / ev._slug(cid) / ev._name(part, draw)).is_file()
    ]


# ---------------------------------------------------------------- zips


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_zip(dest: Path, files: dict[str, Path], extra: dict[str, str]) -> Path:
    """Zip ``files`` (archive name → source) plus text ``extra``, with a SHA-256 manifest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest = "".join(f"{_sha(src)}  {name}\n" for name, src in sorted(files.items()))
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, src in sorted(files.items()):
            z.write(src, name)
        for name, text in extra.items():
            z.writestr(name, text)
        z.writestr("MANIFEST.sha256", manifest)
    return dest


def _code_files(cfg: Config) -> dict[str, bytes]:
    """Return the committed code (``git archive HEAD``), so partners need no GitHub access."""
    raw = subprocess.run(
        ["git", "archive", "--format=zip", "HEAD"],
        cwd=cfg.root,
        check=True,
        capture_output=True,
    ).stdout
    out = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            if not name.endswith("/"):
                out[name] = z.read(name)
    return out


RUN_TEXT = """# One-try study: run your share ({machine})

You only need this folder. No download, no GitHub.

1. Install uv (once). In PowerShell:
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   (macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
2. Open a terminal **in this folder** and run:
   `uv sync --frozen`
   `uv run agent-tips onetry run --machine {machine}`
3. Leave it running (plug in the charger, turn off sleep). If it stops, run the
   same command again: finished work is kept.
4. When it says it is done, run:
   `uv run agent-tips onetry pack --machine {machine}`
   and send back the file it prints (`results-{machine}.zip`).

Expected: {n} comparisons. Memory: about 1 GB per parallel job ({jobs} jobs).
"""


def bundle(cfg: Config, machine: str, log: Callable[[str], None] = print) -> Path:
    """One zip for a partner machine: code, study data, answer keys and the plan."""
    ev.prepare_text(cfg, log)
    cids = assigned(cfg, machine)
    od, res = cfg.path("onetry_data"), cfg.path("onetry_results")
    rel = cfg.root.resolve()
    files = {
        str(f.resolve().relative_to(rel)).replace("\\", "/"): f
        for f in [
            od / "clean" / "tries.parquet",
            od / "clean" / "bugs.parquet",
            *sorted((od / "features").glob("*.parquet")),
            res / "inventory.parquet",
            res / "answer_keys.parquet",
        ]
        if f.is_file()
    }
    code = _code_files(cfg)
    code[str(PLAN).replace("\\", "/")] = (cfg.root / PLAN).read_bytes()
    dest = res / "bundles" / f"onetry-{machine}.zip"
    jobs = int(machines(cfg)[machine]["jobs"])
    _write_zip(
        dest,
        files,
        {"RUN_ME.md": RUN_TEXT.format(machine=machine, n=len(cids), jobs=jobs)},
    )
    with zipfile.ZipFile(dest, "a", compression=zipfile.ZIP_DEFLATED) as z:
        for name, blob in sorted(code.items()):
            z.writestr(name, blob)
    log(f"bundle for {machine}: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def pack(cfg: Config, machine: str, log: Callable[[str], None] = print) -> Path:
    """Zip this machine's finished results (refuses if any task is missing)."""
    cids = assigned(cfg, machine)
    todo = missing(cfg, cids)
    if todo:
        raise RuntimeError(f"{len(todo)} tasks not done yet, e.g. {todo[0]}; run again first")
    res = cfg.path("onetry_results")
    files = {}
    for folder in ("units", "leftovers"):
        for cid in cids:
            for f in sorted((res / folder / ev._slug(cid)).glob("*")):
                files[f"{folder}/{ev._slug(cid)}/{f.name}"] = f
    dest = res / f"results-{machine}.zip"
    _write_zip(dest, files, {"machine.json": json.dumps(machine_info(machine), indent=2)})
    log(f"results of {machine}: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def merge(cfg: Config, zip_path: Path, log: Callable[[str], None] = print) -> list[str]:
    """Check a results zip against its hashes and add it to this machine's results."""
    res = cfg.path("onetry_results")
    with zipfile.ZipFile(zip_path) as z:
        manifest = z.read("MANIFEST.sha256").decode("utf-8").splitlines()
        names = []
        for line in manifest:
            h, name = line.split("  ", 1)
            if not name.startswith(("units/", "leftovers/")) or ".." in name:
                raise ValueError(f"unexpected file in results zip: {name}")
            blob = z.read(name)
            if hashlib.sha256(blob).hexdigest() != h:
                raise ValueError(f"hash mismatch: {name}")
            target = res / name
            if target.is_file() and target.read_bytes() != blob:
                raise FileExistsError(f"different result already here: {target}")
            names.append(name)
        for name in names:
            target = res / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
    cids = sorted({n.split("/")[1] for n in names})
    log(f"merged {len(names)} files, {len(cids)} comparisons from {zip_path.name}")
    return cids


# ---------------------------------------------------------------- cross-machine check


def crosscheck(
    cfg: Config, per_machine: int = 2, seed: int = 0, log: Callable[[str], None] = print
) -> dict[str, Any]:
    """Re-run a few draws of every other machine here; all estimates must agree (A9).

    Rungs 0 and 1 must match exactly (1e-9); every other estimate within 0.1 point.
    """
    p = ev.Params.from_config(cfg)
    here = home(cfg)
    rng = np.random.default_rng(seed)
    rows = []
    for m in machines(cfg):
        if m == here:
            continue
        cids = assigned(cfg, m)
        if not cids:
            continue
        for cid in rng.choice(cids, size=min(per_machine, len(cids)), replace=False):
            draw = int(rng.integers(0, p.check_draws))  # a draw with every rung and check
            f = cfg.path("onetry_results") / "units" / ev._slug(str(cid)) / ev._name("key", draw)
            if not f.is_file():
                log(f"crosscheck: {cid} draw {draw} not merged yet")
                continue
            old = pd.read_parquet(f).set_index("method").estimate
            new = ev.run_task(cfg, str(cid), "key", draw, p)[0].set_index("method").estimate
            both = pd.concat({"old": old, "new": new}, axis=1)
            diff = (both.old - both.new).abs()
            exact = diff[diff.index.isin(["rung0", "rung1"])]
            rows.append(
                {
                    "machine": m,
                    "cid": str(cid),
                    "draw": draw,
                    "max_diff": float(diff.max()),
                    "exact_ok": bool((exact < 1e-9).all()),
                    "ok": bool((exact < 1e-9).all() and (diff < TOLERANCE).all()),
                }
            )
    out = {"rows": rows, "passed": bool(rows) and all(r["ok"] for r in rows)}
    (cfg.path("onetry_results") / "crosscheck.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    return out
