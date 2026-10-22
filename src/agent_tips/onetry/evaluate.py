"""Step 7 (one-try study): the full run, the scores and the hypothesis tests.

**Full run.** One unit of work is one comparison × one draw (amendment A9), run
in parallel with one thread each and saved to
``outputs/onetry/units/<comparison>/<part>-<draw>.parquet`` (so a stopped run
resumes). Per comparison (draw numbers per ``01_problem.md`` and amendments):

- draws 0-19: rungs 0, 1, 2, 4, 5, 6 on one-try copies of the key bugs, plus
  the realistic other-AI rung (``rung5_real``) and, for the many-tries
  datasets, the oracle rung (``rung7``) (A8);
- draws 20-39: rungs 0 and 5 (precision check, A4);
- draws 0-4: rung 3 and AIPW rungs 2, 4, 5 (checks, A1); draws 0-1: random-forest
  rung 5 (A9);
- NVIDIA, draws 0-19: rung 5 using only 1, 2 or 4 other setups (H4b);
- draws 0-4: rungs 0, 1, 2, 5 and AIPW rung 5 on copies drawn from all bugs,
  with and without trimming (secondary, A7 and A8);
- older AI and Nebius, draws 0-19: the within-bug answer from k = 2, 3, 5, 10
  tries per bug (RQ4).

For draws 0-19 (and the all-bugs copies) the per-bug leftovers of every rung are
saved to ``outputs/onetry/leftovers/`` for the joint bootstrap and the model
diagnostics (A8).

**Scores and tests** follow ``01_problem.md`` §6-7: error = mean over draws of
(estimate − answer key); averages over comparisons with CIs from bootstrap
resamples of AI setups; H1-H4 with Holm correction. Amendment A8 adds the
secondary scores in :mod:`agent_tips.onetry.extra`.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy import stats  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from agent_tips.config import Config  # noqa: E402
from agent_tips.eda.charts import COLORS  # noqa: E402
from agent_tips.onetry import corrections as cr  # noqa: E402
from agent_tips.onetry import data, extra  # noqa: E402
from agent_tips.onetry.comparisons import NEBIUS, NVIDIA, OLDER  # noqa: E402
from agent_tips.onetry.features import other_ai_rates, text_table, tokenize  # noqa: E402
from agent_tips.onetry.simulate import (  # noqa: E402
    k_try_answer,
    k_try_copy,
    one_try_copy,
    own_loo,
    realistic_other,
    seed_for,
)

MAIN = ["rung0", "rung1", "rung2", "rung4", "rung5", "rung6"]
LABELS = {
    "rung0": "Simple comparison",
    "rung1": "Linear regression on report facts",
    "rung2": "ML: report facts + text",
    "rung3": "ML: + language-model embeddings",
    "rung4": "ML: + human-fix size",
    "rung5": "ML: + other AIs' results",
    "rung6": "ML: + IRT difficulty",
    "rung5_real": "ML: + other AIs, one try each (realistic)",
    "rung7": "ML: + the AI's own other tries (oracle)",
}
SECONDARY_RUNGS = ["rung3", "rung5_real", "rung7"]  # reported, not in H1-H4 (A8)
TRIM = (0.1, 0.9)  # A8: all-bugs trimming on the out-of-fold chance of choice a
MANY_TRIES = (OLDER, NEBIUS)  # datasets where the oracle rung is defined (A8)


@dataclass(frozen=True)
class Params:
    """How much to run (from ``configs/default.yaml`` → ``onetry``)."""

    draws: int = 20
    precision_draws: int = 40
    check_draws: int = 5
    forest_draws: int = 2
    secondary_draws: int = 5
    folds: int = 5
    model_threads: int = 1
    cached_text: bool = False  # A9: True = one text model per dataset (failed the pass rule)
    jobs: int = 8
    h4_ks: tuple[int, ...] = (1, 2, 4)
    k_tries: tuple[int, ...] = (2, 3, 5, 10)
    min_ktry_bugs: int = 30

    @classmethod
    def from_config(cls, cfg: Config) -> Params:
        """Read the settings, falling back to the frozen defaults."""
        o = cfg.raw["onetry"]
        names = set(cls.__dataclass_fields__)
        kw: dict[str, Any] = {
            k: (tuple(v) if isinstance(v, list) else v) for k, v in o.items() if k in names
        }
        return cls(**kw)


# ---------------------------------------------------------------- one unit of work

_STUDY: dict[str, data.StudyData] = {}
_TOKENS: dict[tuple[str, bool], tuple[tuple[str, ...], list[list[str]]]] = {}


def _study(cfg: Config) -> data.StudyData:
    key = str(cfg.root)
    if key not in _STUDY:
        _STUDY[key] = data.load(cfg)
    return _STUDY[key]


def prepare_text(cfg: Config, log: Callable[[str], None] = print) -> Path:
    """Build the cached report-text numbers once (amendment A9)."""
    path = data.text_path(cfg)
    if not path.is_file():
        log("report text: one text model per dataset (cached)")
        reports = pd.read_parquet(
            cfg.path("onetry_data") / "clean" / "bugs.parquet",
            columns=["dataset", "bug_id", "report"],
        )
        text_table(reports.assign(report=reports.report.fillna(""))).to_parquet(path, index=False)
        _STUDY.clear()
    return path


def _text(
    c: Any, d: pd.DataFrame, key_only: bool, draw: int, folds: Any, p: Params
) -> list[dict[str, np.ndarray]] | None:
    """Report-text numbers of one draw, fitted inside each training fold (frozen in Step 1).

    The word-splitting is done once per comparison and worker and reused (A6);
    ``None`` when the cached text of amendment A9 is used instead.
    """
    if p.cached_text:
        return None
    ids = tuple(d.bug_id)
    hit = _TOKENS.get((c.cid, key_only))
    if hit is None or hit[0] != ids:
        _TOKENS.clear()  # keep one comparison's words only: they are large
        _TOKENS[(c.cid, key_only)] = hit = (ids, tokenize(d.report))
    return cr.text_folds(d, folds, 100, draw, hit[1])


class _Out:
    """Collects estimates and per-bug leftovers of one unit."""

    def __init__(self, cid: str, part: str, draw: int) -> None:
        self.cid, self.part, self.draw = cid, part, draw
        self.rows: list[dict[str, Any]] = []
        self.left: list[pd.DataFrame] = []

    def add(self, method: str, r: cr.Result, n: int, draw: int | None = None) -> None:
        self.rows.append(
            {
                "cid": self.cid,
                "draw": self.draw if draw is None else draw,
                "method": method,
                "estimate": r.estimate,
                "se": r.se,
                "n": n,
            }
        )

    def keep(self, method: str, bugs: pd.Series, ry: np.ndarray, rt: np.ndarray) -> None:
        self.left.append(
            pd.DataFrame(
                {
                    "draw": np.int16(self.draw),
                    "method": method,
                    "bug_id": bugs.to_numpy(),
                    "ry": ry.astype(np.float32),
                    "rt": rt.astype(np.float32),
                }
            )
        )

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        return pd.DataFrame(self.rows), (
            pd.concat(self.left, ignore_index=True) if self.left else None
        )


def _key_draw(s: data.StudyData, c: Any, draw: int, p: Params, out: _Out) -> None:
    """One draw of one-try copies of the key bugs: every rung due at this draw."""
    main, check, forest = draw < p.draws, draw < p.check_draws, draw < p.forest_draws
    has_emb = s.embeddings is not None
    copy = one_try_copy(c, s.tries, draw)
    d = data.with_features(copy, c, s, embeddings=check and has_emb, text=p.cached_text)
    n, bugs = len(d), d.bug_id
    folds = cr.make_folds(n, p.folds, draw)
    text = _text(c, d, True, draw, folds, p)

    def ladder_rung(name: str, frame: pd.DataFrame, rung: int, learner: str = "gbm") -> None:
        assert frame.bug_id.tolist() == bugs.tolist()  # same rows, so the text folds fit
        ry, rt = cr.dml_parts(frame, rung, draw, learner=learner, folds=folds, text=text)
        out.add(name, cr.partial_out(ry, rt), n)
        if main and learner == "gbm":
            out.keep(name, bugs, ry, rt)

    out.add("rung0", cr.simple(d), n)
    if main:
        out.keep("rung0", bugs, *cr.simple_parts(d))
        out.add("rung1", cr.linear(d), n)
        out.keep("rung1", bugs, *cr.linear_parts(d))
    for r in (2, 4, 5, 6) if main else (5,):
        ladder_rung(f"rung{r}", d, r)
    if not main:
        return
    idx = pd.Index(bugs, name="bug_id")
    real = d.drop(columns=cr.OTHER).merge(realistic_other(c, s.tries, idx, draw), on="bug_id")
    ladder_rung("rung5_real", real, 5)
    if c.dataset in MANY_TRIES:
        ladder_rung("rung7", d.merge(own_loo(c, s.tries, copy), on="bug_id"), 7)
    if check:
        if has_emb:
            ladder_rung("rung3", d, 3)
        for r in (2, 4, 5):
            psi, _ = cr.aipw_parts(d, r, draw, folds=folds, text=text)
            out.add(f"rung{r}_aipw", cr.mean_result(psi), n)
            out.keep(f"rung{r}_aipw", bugs, psi, np.ones(n))
    if forest:
        ladder_rung("rung5_forest", d, 5, learner="forest")
    if c.dataset == NVIDIA:
        others = sorted(
            s.tries.loc[(s.tries.setup != c.setup) & s.tries.bug_id.isin(idx), "setup"].unique()
        )
        for k in p.h4_ks:
            if k >= len(others):
                continue
            rng = np.random.default_rng(seed_for(c.cid, draw, 50 + k))
            chosen = set(rng.choice(others, size=k, replace=False))
            oa = other_ai_rates(s.tries[s.tries.setup.isin(chosen)], c.setup, idx).reset_index()
            ladder_rung(f"rung5_k{k}", d.drop(columns=cr.OTHER).merge(oa, on="bug_id"), 5)


def _all_draw(s: data.StudyData, c: Any, draw: int, p: Params, out: _Out) -> None:
    """One draw of one-try copies of all bugs (A7), with the trimmed versions (A8)."""
    copy = one_try_copy(c, s.tries, draw, key_only=False)
    d = data.with_features(copy, c, s, text=p.cached_text)
    n, bugs = len(d), d.bug_id
    folds = cr.make_folds(n, p.folds, draw)
    text = _text(c, d, False, draw, folds, p)
    out.add("rung0_allbugs", cr.simple(d), n)
    out.add("rung1_allbugs", cr.linear(d), n)
    for r in (2, 5):
        ry, rt = cr.dml_parts(d, r, draw, folds=folds, text=text)
        out.add(f"rung{r}_allbugs", cr.partial_out(ry, rt), n)
        out.keep(f"rung{r}_allbugs", bugs, ry, rt)
    psi, prop = cr.aipw_parts(d, 5, draw, folds=folds, text=text)
    out.add("rung5_aipw_allbugs", cr.mean_result(psi), n)
    if np.isfinite(prop).all():
        keep = (prop >= TRIM[0]) & (prop <= TRIM[1])
        dk = d[keep].reset_index(drop=True)
        ok = int(keep.sum()) > 10 and dk.t.nunique() == 2
        nan = cr.Result(float("nan"), float("nan"))
        out.add("rung0_allbugs_trim", cr.simple(dk) if ok else nan, int(keep.sum()))
        out.add("rung1_allbugs_trim", cr.linear(dk) if ok else nan, int(keep.sum()))
        out.add("rung5_aipw_allbugs_trim", cr.mean_result(psi[keep]), int(keep.sum()))


def _ktry(s: data.StudyData, c: Any, p: Params, out: _Out) -> None:
    """RQ4: the within-bug answer from k tries per bug (many-tries datasets)."""
    for k in p.k_tries:
        for draw in range(p.draws):
            kd = k_try_copy(c, s.tries, k, draw)
            if kd.bug_id.nunique() >= p.min_ktry_bugs:
                r = cr.Result(k_try_answer(c, kd), float("nan"))
                out.add(f"ktry{k}", r, int(kd.bug_id.nunique()), draw=draw)


def run_task(
    cfg: Config, cid: str, part: str, draw: int, p: Params
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """One unit of work: estimates and leftovers of one comparison × part × draw."""
    s = _study(cfg)
    c = next(x for x in s.comparisons if x.cid == cid)
    out = _Out(cid, part, draw)
    with threadpool_limits(p.model_threads):
        if part == "key":
            _key_draw(s, c, draw, p, out)
        elif part == "all":
            _all_draw(s, c, draw, p, out)
        else:
            _ktry(s, c, p, out)
    frames = out.frames()
    gc.collect()  # the per-fold text models are large; free them before the next task
    return frames


def run_unit(cfg: Config, cid: str, p: Params) -> pd.DataFrame:
    """All estimates of one comparison (every task in turn; used for the A3 re-runs)."""
    return pd.concat(
        [run_task(cfg, cid, part, draw, p)[0] for part, draw in tasks_of(cfg, cid, p)],
        ignore_index=True,
    )


def tasks_of(cfg: Config, cid: str, p: Params) -> list[tuple[str, int]]:
    """List the (part, draw) tasks of one comparison."""
    c = next(x for x in _study(cfg).comparisons if x.cid == cid)
    t = [("key", d) for d in range(p.precision_draws)]
    t += [("all", d) for d in range(p.secondary_draws)]
    if c.dataset in MANY_TRIES:
        t.append(("ktry", -1))
    return t


def _slug(cid: str) -> str:
    return cid.replace("/", "__").replace("|", "--")


def _name(part: str, draw: int) -> str:
    return f"{part}-{draw:02d}.parquet" if draw >= 0 else f"{part}.parquet"


def _task_to_file(cfg: Config, cid: str, part: str, draw: int, p: Params) -> str:
    t = time.time()
    est, left = run_task(cfg, cid, part, draw, p)
    res = cfg.path("onetry_results")
    for folder, frame in (("units", est), ("leftovers", left)):
        if frame is None:
            continue
        path = res / folder / _slug(cid) / _name(part, draw)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    with (res / "units" / "_progress.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%m-%d %H:%M:%S')} {cid} {part} {draw} {time.time() - t:.0f}s\n")
    return cid


def sealed_cids(cfg: Config) -> list[str]:
    """Comparisons in the sealed first run (amendment A9), from the sealed hash list."""
    lst = cfg.root / "docs" / "one_try" / "sealed_units_v1.sha256"
    if not lst.is_file():
        return []
    names = [
        ln.split()[1] for ln in lst.read_text(encoding="utf-8").splitlines() if ".parquet" in ln
    ]
    slugs = {n.removesuffix(".parquet") for n in names}
    return [c.cid for c in _study(cfg).comparisons if _slug(c.cid) in slugs]


def run_all(
    cfg: Config,
    p: Params,
    fresh: bool = False,
    only: list[str] | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Run every task not yet done, in parallel (sealed comparisons first, then largest first).

    ``only`` limits the run to some comparisons (one machine's share, A9).
    """
    if p.cached_text:
        prepare_text(cfg, log)
    res = cfg.path("onetry_results")
    (res / "units").mkdir(parents=True, exist_ok=True)
    s = _study(cfg)
    keys = pd.read_parquet(res / "answer_keys.parquet")
    size = dict(zip(keys.cid, keys.key_bugs, strict=True))
    first = set(sealed_cids(cfg))
    order = sorted(
        (c.cid for c in s.comparisons if only is None or c.cid in only),
        key=lambda cid: (cid not in first, -size.get(cid, 0)),
    )
    todo = [
        (cid, part, draw)
        for cid in order
        for part, draw in tasks_of(cfg, cid, p)
        if fresh or not (res / "units" / _slug(cid) / _name(part, draw)).is_file()
    ]
    log(f"full run: {len(todo)} tasks over {len(order)} comparisons, {p.jobs} at a time")
    Parallel(n_jobs=p.jobs, backend="loky", batch_size=1)(
        delayed(_task_to_file)(cfg, cid, part, draw, p) for cid, part, draw in todo
    )
    return res / "units"


def load_units(cfg: Config) -> pd.DataFrame:
    """All saved estimates."""
    files = sorted((cfg.path("onetry_results") / "units").glob("*/*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def a9_check(cfg: Config, tol: float = 0.003) -> dict[str, Any]:
    """Amendment A9 pass rule: the new code against the sealed first run.

    Checks the hashes, then compares estimates only (never the answer keys):
    rungs 0 and 1 must match draw by draw; for rungs 2, 4, 5 and 6 the mean
    over draws 0-19 per comparison must agree within ``tol`` (0.3 points).
    """
    sealed = cfg.path("onetry_results") / "sealed_v1"
    lst = cfg.root / "docs" / "one_try" / "sealed_units_v1.sha256"
    bad_hash = []
    for ln in lst.read_text(encoding="utf-8").splitlines():
        h, name = ln.split()
        if hashlib.sha256((sealed / name).read_bytes()).hexdigest() != h:
            bad_hash.append(name)
    rows = []
    for cid in sealed_cids(cfg):
        old = pd.read_parquet(sealed / f"{_slug(cid)}.parquet")
        new_files = sorted(
            (cfg.path("onetry_results") / "units" / _slug(cid)).glob("key-*.parquet")
        )
        if len(new_files) < 20:
            continue
        new = pd.concat([pd.read_parquet(f) for f in new_files])
        for m in MAIN:
            a = old[(old.method == m) & (old.draw < 20)].set_index("draw").estimate.sort_index()
            b = new[(new.method == m) & (new.draw < 20)].set_index("draw").estimate.sort_index()
            if m in ("rung0", "rung1"):
                diff = float(np.max(np.abs(a.to_numpy() - b.reindex(a.index).to_numpy())))
                ok = diff < 1e-9
            else:
                diff = abs(float(a.mean()) - float(b.mean()))
                ok = diff < tol
            rows.append({"cid": cid, "method": m, "diff": diff, "ok": ok})
    t = pd.DataFrame(rows)
    return {
        "hashes_ok": not bad_hash,
        "bad_hashes": bad_hash,
        "comparisons": int(t.cid.nunique()) if len(t) else 0,
        "passed": bool(not bad_hash and len(t) and t.ok.all()),
        "max_diff_dml": float(t[~t.method.isin(["rung0", "rung1"])]["diff"].max())
        if len(t)
        else float("nan"),
        "table": t.to_dict(orient="records"),
    }


# ---------------------------------------------------------------- scores


def scores(units: pd.DataFrame, keys: pd.DataFrame, draws: int | None = None) -> pd.DataFrame:
    """Per comparison × method: error, absolute error, spread, coverage, agreement."""
    k = keys.set_index("cid")
    d = units if draws is None else units[units.draw < draws]
    d = d.join(
        k[["estimate", "ci_lo", "ci_hi", "conclusion"]].rename(columns=lambda c: f"key_{c}"),
        on="cid",
    )
    d["err"] = d.estimate - d.key_estimate
    lo, hi = d.estimate - cr.Z95 * d.se, d.estimate + cr.Z95 * d.se
    d["covered"] = np.where(d.se.notna(), (lo <= d.key_estimate) & (d.key_estimate <= hi), np.nan)
    concl = np.where(lo > 0, 1, np.where(hi < 0, -1, 0))
    d["concl_agree"] = np.where(d.se.notna(), concl == d.key_conclusion, np.nan)
    d["sign_agree"] = np.where(
        d.key_conclusion != 0, np.sign(d.estimate) == np.sign(d.key_estimate), np.nan
    )
    g = d.groupby(["cid", "method"])
    out = pd.DataFrame(
        {
            "error": g.err.mean(),
            "spread": g.estimate.std(),
            "coverage": g.covered.mean(),
            "concl_agree": g.concl_agree.mean(),
            "sign_agree": g.sign_agree.mean(),
            "draws": g.size(),
        }
    ).reset_index()
    out["abs_error"] = out.error.abs()
    return out.merge(keys[["cid", "dataset", "setup", "kind", "key_bugs", "noise_floor"]], on="cid")


def cluster_boot(sc: pd.DataFrame, method: str, reps: int, seed: int) -> np.ndarray:
    """Bootstrap the mean absolute error over comparisons, resampling AI setups."""
    m = sc[sc.method == method]
    per = m.groupby("setup").abs_error.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(reps, len(per)))
    tot, cnt = per["sum"].to_numpy()[idx].sum(1), per["count"].to_numpy()[idx].sum(1)
    return np.asarray(tot / cnt)


def summary_table(sc: pd.DataFrame, methods: list[str], reps: int, seed: int) -> pd.DataFrame:
    """Mean absolute error per method with bootstrap bounds and the other scores."""
    rows = []
    for m in methods:
        part = sc[sc.method == m]
        if part.empty:
            continue
        b = cluster_boot(sc, m, reps, seed)
        rows.append(
            {
                "method": m,
                "comparisons": len(part),
                "mean_abs_error": float(part.abs_error.mean()),
                "lo95": float(np.percentile(b, 2.5)),
                "hi95": float(np.percentile(b, 97.5)),
                "lower_5pct": float(np.percentile(b, 5)),
                "upper_95pct": float(np.percentile(b, 95)),
                "median_abs_error": float(part.abs_error.median()),
                "coverage": float(part.coverage.mean()),
                "concl_agree": float(part.concl_agree.mean()),
                "sign_agree": float(part.sign_agree.mean()),
                "boot": b,
            }
        )
    return pd.DataFrame(rows)


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values."""
    items = sorted(p.items(), key=lambda kv: kv[1])
    adj, running = {}, 0.0
    for i, (k, v) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * v))
        adj[k] = running
    return adj


def hypotheses(
    sc: pd.DataFrame,
    summ: pd.DataFrame,
    inv: pd.DataFrame,
    units: pd.DataFrame,
    keys: pd.DataFrame,
    margin: float,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    """H1-H4 exactly as frozen in Step 1."""
    m: dict[str, dict[Any, Any]] = {str(r["method"]): r for r in summ.to_dict(orient="records")}

    def stat(method: str, key: str) -> float:
        return float(m[method][key])

    def share(method: str, below: bool) -> float:
        boot = np.asarray(m[method]["boot"], dtype=float)
        return float(np.mean(boot <= margin) if below else np.mean(boot >= margin))

    out: dict[str, Any] = {
        "H1": {
            "stat": stat("rung0", "mean_abs_error"),
            "bound": stat("rung0", "lower_5pct"),
            "p": share("rung0", below=True),
        },
        "H2": {
            "stat_rung1": stat("rung1", "mean_abs_error"),
            "stat_rung2": stat("rung2", "mean_abs_error"),
            "bound_rung1": stat("rung1", "lower_5pct"),
            "bound_rung2": stat("rung2", "lower_5pct"),
            "p": max(share("rung1", below=True), share("rung2", below=True)),
        },
        "H3": {
            "stat": stat("rung5", "mean_abs_error"),
            "bound": stat("rung5", "upper_95pct"),
            "p": share("rung5", below=False),
        },
    }
    out["H1"]["supported"] = out["H1"]["bound"] > margin
    out["H2"]["supported"] = min(out["H2"]["bound_rung1"], out["H2"]["bound_rung2"]) > margin
    out["H3"]["supported"] = out["H3"]["bound"] < margin
    out["H4"] = h4(sc, inv, units, keys, reps, seed)
    adj = holm({k: float(v["p"]) for k, v in out.items()})
    for k, v in out.items():
        v["p_holm"] = adj[k]
        v["decision"] = "supported" if v["supported"] and adj[k] < 0.05 else "not supported"
    return out


def h4(
    sc: pd.DataFrame,
    inv: pd.DataFrame,
    units: pd.DataFrame,
    keys: pd.DataFrame,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    """H4a (gain vs agreement) and H4b (error vs number of other setups); both required."""
    import statsmodels.api as sm

    wide = (
        sc[sc.method.isin(["rung2", "rung5"])]
        .pivot_table(index=["cid", "setup"], columns="method", values="abs_error")
        .reset_index()
    )
    wide["gain"] = wide.rung2 - wide.rung5
    wide = wide.merge(inv[["cid", "agreement"]], on="cid").dropna(subset=["agreement", "gain"])
    x = sm.add_constant(wide.agreement.to_numpy())
    fit = sm.OLS(wide.gain.to_numpy(), x).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(wide.setup)[0]}
    )
    slope, se = float(fit.params[1]), float(fit.bse[1])
    pa = float(stats.norm.sf(slope / se)) if se > 0 else 1.0
    # H4b: rung 5 with k other setups (k = 1, 2, 4, all) on NVIDIA comparisons
    kk = sc[sc.method.str.fullmatch(r"rung5(_k\d+)?") & (sc.dataset == NVIDIA)].copy()
    kk["k"] = kk.method.str.extract(r"_k(\d+)$")[0].astype(float).fillna(99.0)
    kk = kk[kk.cid.isin(kk[kk.k < 99].cid)]
    rho = float(stats.spearmanr(kk.k, kk.abs_error).statistic) if len(kk) > 2 else float("nan")
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(reps):
        shuffled = kk.groupby("cid").k.transform(lambda s: rng.permutation(s.to_numpy()))
        perm.append(stats.spearmanr(shuffled, kk.abs_error).statistic)
    pb = float(np.mean(np.asarray(perm) <= rho)) if len(kk) > 2 else 1.0
    by_k = kk.groupby("k").abs_error.mean().to_dict()
    return {
        "slope_gain_vs_agreement": slope,
        "slope_se": se,
        "p_a": pa,
        "rho_error_vs_k": rho,
        "p_b": pb,
        "error_by_k": {
            ("all" if float(str(k)) == 99 else int(float(str(k)))): v for k, v in by_k.items()
        },
        "comparisons_a": len(wide),
        "comparisons_b": int(kk.cid.nunique()),
        "p": max(pa, pb),
        "supported": slope > 0 and rho < 0,
    }


# ---------------------------------------------------------------- reproducibility


def home_cids(cfg: Config) -> list[str]:
    """Comparisons run on this (home) computer: all, unless the run was shared (A9)."""
    plan = cfg.root / "docs" / "one_try" / "share_plan.json"
    every = [c.cid for c in _study(cfg).comparisons]
    if not plan.is_file():
        return every
    got = json.loads(plan.read_text(encoding="utf-8"))
    home = next(k for k, v in got["machines"].items() if v.get("home"))
    return [c for c in every if c in set(got["comparisons"][home])]


def reproducibility(cfg: Config, p: Params, n: int = 3, seed: int = 0) -> dict[str, Any]:
    """Re-run ``n`` random comparisons in full; every row must match (amendment A3).

    If the run was shared across computers, the comparisons are drawn from those
    made on this one (identity across different chips is checked by A9 instead).
    """
    units = cfg.path("onetry_results") / "units"
    pool = home_cids(cfg)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(pool, size=min(n, len(pool)), replace=False))
    todo = [(str(cid), part, draw) for cid in chosen for part, draw in tasks_of(cfg, str(cid), p)]
    again = Parallel(n_jobs=p.jobs, backend="loky", batch_size=1)(
        delayed(run_task)(cfg, cid, part, draw, p) for cid, part, draw in todo
    )
    same: dict[str, bool] = dict.fromkeys(map(str, chosen), True)
    for (cid, part, draw), (b, _) in zip(todo, again, strict=True):
        a = pd.read_parquet(units / _slug(cid) / _name(part, draw))
        key = ["draw", "method"]
        a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
        same[cid] &= bool(
            a.shape == b.shape
            and np.allclose(a.estimate, b.estimate, equal_nan=True, atol=1e-12, rtol=0)
        )
    return {"comparisons": list(same), "identical": same, "all_identical": all(same.values())}


# ---------------------------------------------------------------- charts and report


def _save(fig: Figure, name: str, cfg: Config) -> None:
    png = cfg.root / "docs" / "figures" / "onetry"
    pdf = cfg.path("onetry_results") / "figures"
    png.mkdir(parents=True, exist_ok=True)
    pdf.mkdir(parents=True, exist_ok=True)
    fig.savefig(png / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(pdf / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def chart_ladder(summ: pd.DataFrame, margin: float, cfg: Config) -> None:
    """E1: mean absolute error per rung with 95% bounds."""
    d = summ[summ.method.isin(MAIN + ["rung3"])].copy()
    d["order"] = d.method.map(
        {
            m: i
            for i, m in enumerate(["rung0", "rung1", "rung2", "rung3", "rung4", "rung5", "rung6"])
        }
    )
    d = d.sort_values("order")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(d))[::-1]
    ax.barh(
        y,
        d.mean_abs_error * 100,
        color=[COLORS[3] if m == "rung0" else COLORS[0] for m in d.method],
    )
    ax.errorbar(
        d.mean_abs_error * 100,
        y,
        xerr=[(d.mean_abs_error - d.lo95) * 100, (d.hi95 - d.mean_abs_error) * 100],
        fmt="none",
        color="black",
        capsize=3,
        lw=0.8,
    )
    ax.axvline(margin * 100, color="black", ls="--", lw=0.8)
    ax.set_yticks(y, [LABELS[m] for m in d.method], fontsize=8)
    ax.set_xlabel("Average distance from the many-tries answer (points); dashed = 2-point margin")
    ax.set_title("E1. How far off is each method, with one try per bug?")
    _save(fig, "e1_ladder", cfg)


def chart_k(h4r: dict[str, Any], cfg: Config) -> None:
    """E2: rung-5 error by number of other AIs used."""
    eb = h4r["error_by_k"]
    ks = [k for k in (1, 2, 4, "all") if k in eb]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(range(len(ks)), [eb[k] * 100 for k in ks], marker="o", color=COLORS[0])
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_xlabel("Other AIs whose results the method may use")
    ax.set_ylabel("Average distance (points)")
    ax.set_title("E2. More other AIs, smaller error? (NVIDIA)")
    _save(fig, "e2_other_ais", cfg)


def chart_ktry(sc: pd.DataFrame, cfg: Config) -> None:
    """E3: error of the within-bug answer with k tries per bug (RQ4)."""
    d = sc[sc.method.str.startswith("ktry")].copy()
    if d.empty:
        return
    d["k"] = d.method.str[4:].astype(int)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, (ds, g) in enumerate(d.groupby("dataset")):
        m = g.groupby("k").abs_error.mean()
        ax.plot(m.index, m * 100, marker="o", color=COLORS[i], label=str(ds))
    ax.set_xlabel("Tries per bug")
    ax.set_ylabel("Average distance (points)")
    ax.set_title("E3. How many tries per bug are enough?")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, "e3_tries_needed", cfg)


def _p(x: float) -> str:
    return f"{x * 100:.1f}"


def write_report(
    cfg: Config, r: dict[str, Any], summ: pd.DataFrame, sc: pd.DataFrame, path: Path | None = None
) -> Path:
    """Write ``docs/one_try/07_evaluation.md``."""
    h, margin = r["hypotheses"], r["margin"]
    lines = [
        "# Step 7: Evaluation",
        "",
        "*Generated by `agent-tips onetry evaluate`. Paper 2, the one-try study. Rules frozen in "
        "`01_problem.md` (tag `one-try-plan-v1`, amendments A1–A9).*",
        "",
        "## In plain words",
        "",
        f"- **How wrong is a one-try study?** The simple comparison is off by {_p(h['H1']['stat'])} points "
        f"on average (H1: {h['H1']['decision']}).",
        f"- **Does machine learning on the bug report fix it?** Not enough: linear regression is off by "
        f"{_p(h['H2']['stat_rung1'])} and ML on report facts + text by {_p(h['H2']['stat_rung2'])} points "
        f"(H2: {h['H2']['decision']}).",
        f"- **Does adding other AIs' results fix it?** Rung 5 is off by {_p(h['H3']['stat'])} points on average; "
        f"the upper bound is {_p(h['H3']['bound'])} against the {margin * 100:.0f}-point margin "
        f"(H3: {h['H3']['decision']}).",
        f"- **When does it fail?** H4: {h['H4']['decision']} (gain vs agreement slope "
        f"{h['H4']['slope_gain_vs_agreement'] * 100:+.1f} points per unit; error vs number of other AIs "
        f"ρ = {h['H4']['rho_error_vs_k']:.2f}).",
        f"- **Realistic one-try studies (all bugs, amendment A7):** simple comparison off by "
        f"{_p(r['secondary']['rung0']['all'])}, report-text ML {_p(r['secondary']['rung2']['all'])}, "
        f"other-AI ML {_p(r['secondary']['rung5']['all'])} points.",
        "- **Reproducible:** "
        + (
            "re-running the sampled comparisons gave identical results."
            if r["repro"]["all_identical"]
            else "**re-running gave different results; see below.**"
        ),
        "",
        "![Ladder](../figures/onetry/e1_ladder.png)",
        "",
        "## Hypotheses",
        "",
        "| Hypothesis | Rule | Result | Holm p | Decision |",
        "| --- | --- | --- | --- | --- |",
        f"| H1: simple comparison is off by more than {margin * 100:.0f} points | lower 95% bound > {margin * 100:.0f} | {_p(h['H1']['stat'])} (bound {_p(h['H1']['bound'])}) | {h['H1']['p_holm']:.3f} | {h['H1']['decision']} |",
        f"| H2: report-only corrections stay off by more than {margin * 100:.0f} | both lower bounds > {margin * 100:.0f} | rung 1 {_p(h['H2']['stat_rung1'])} (bound {_p(h['H2']['bound_rung1'])}); rung 2 {_p(h['H2']['stat_rung2'])} (bound {_p(h['H2']['bound_rung2'])}) | {h['H2']['p_holm']:.3f} | {h['H2']['decision']} |",
        f"| H3: other-AI correction is within {margin * 100:.0f} | upper 95% bound < {margin * 100:.0f} | {_p(h['H3']['stat'])} (bound {_p(h['H3']['bound'])}) | {h['H3']['p_holm']:.3f} | {h['H3']['decision']} |",
        f"| H4: the gain shrinks when other-AI information is weaker | slope > 0 and ρ < 0, both one-sided | slope {h['H4']['slope_gain_vs_agreement'] * 100:+.1f} (p {h['H4']['p_a']:.3f}); ρ {h['H4']['rho_error_vs_k']:.2f} (p {h['H4']['p_b']:.3f}) | {h['H4']['p_holm']:.3f} | {h['H4']['decision']} |",
        "",
        "## All methods",
        "",
        "*Average absolute distance from the many-tries answer over comparisons (points), 95% bounds from "
        "bootstrap resamples of AI setups; coverage = share of draws whose 95% interval contains the answer; "
        "conclusion = share of draws that reach the same conclusion (helps / hurts / no clear difference).*",
        "",
        "| Method | Comparisons | Average distance | 95% bounds | Median | Coverage | Same conclusion | Same direction |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows_by_method = {str(r["method"]): r for r in summ.to_dict(orient="records")}
    for m in [x for x in MAIN + SECONDARY_RUNGS if x in rows_by_method]:
        q = rows_by_method[m]
        lines.append(
            f"| {LABELS[m]} | {int(q['comparisons'])} | {_p(q['mean_abs_error'])} | "
            f"{_p(q['lo95'])}–{_p(q['hi95'])} | {_p(q['median_abs_error'])} | {q['coverage']:.0%} | "
            f"{q['concl_agree']:.0%} | {q['sign_agree']:.0%} |"
        )
    lines += [
        "",
        "## By dataset and by kind of comparison",
        "",
        "| Group | " + " | ".join(LABELS[m] for m in MAIN) + " |",
        "| --- | " + " | ".join("---" for _ in MAIN) + " |",
    ]
    for col in ("dataset", "kind"):
        for gname, g in sc[sc.method.isin(MAIN)].groupby(col):
            vals = g.groupby("method").abs_error.mean()
            lines.append(
                f"| {gname} | " + " | ".join(_p(vals.get(m, float("nan"))) for m in MAIN) + " |"
            )
    ch = r["checks"]
    lines += [
        "",
        "## Checks (draws 0–4; not used for decisions)",
        "",
        "| Check | Average distance (points) | Same draws, main method |",
        "| --- | --- | --- |",
    ]
    for k, v in ch.items():
        lines.append(f"| {k} | {_p(v['check'])} | {_p(v['main'])} |")
    lines += [
        "",
        "## Secondary: copies drawn from all bugs (amendment A7, draws 0–4)",
        "",
        "*A real one-try study cannot leave out bugs where the AI always or never makes a choice; these "
        "copies keep them. The answer key covers key bugs only, so part of the difference can come from "
        "the two groups of bugs having different true effects.*",
        "",
        "| Method | Key bugs only | All bugs |",
        "| --- | --- | --- |",
    ]
    for m, v in r["secondary"].items():
        lines.append(f"| {LABELS.get(m, m)} | {_p(v['key'])} | {_p(v['all'])} |")
    lines += [
        "",
        "## H4b: rung 5 with fewer other AIs (NVIDIA)",
        "",
        "![Other AIs](../figures/onetry/e2_other_ais.png)",
        "",
        "| Other AIs used | Average distance (points) |",
        "| --- | --- |",
    ]
    for k, v in h["H4"]["error_by_k"].items():
        lines.append(f"| {k} | {_p(v)} |")
    lines += [
        "",
        "## RQ4: how many tries per bug are enough?",
        "",
        "![Tries needed](../figures/onetry/e3_tries_needed.png)",
        "",
        "| Dataset | Tries per bug | Average distance (points) | Comparisons |",
        "| --- | --- | --- | --- |",
    ]
    for row in r["rq4"]:
        lines.append(
            f"| `{row['dataset']}` | {row['k']} | {_p(row['abs_error'])} | {row['comparisons']} |"
        )
    pc = r["precision"]
    lines += [
        "",
        "## Precision check (amendment A4)",
        "",
        f"Average distance with 20 vs 40 draws: simple comparison {_p(pc['rung0'][0])} vs {_p(pc['rung0'][1])}; "
        f"rung 5 {_p(pc['rung5'][0])} vs {_p(pc['rung5'][1])}. The rule is a difference below 0.2 points: "
        + (
            "passed, 20 draws are enough."
            if pc["passed"]
            else "**not passed; the 40-draw values are reported as main.**"
        ),
        "",
        "## Reproducibility (amendment A3)",
        "",
        "| Comparison re-run in full | Identical |",
        "| --- | --- |",
    ]
    for cid, same in r["repro"]["identical"].items():
        lines.append(f"| `{cid}` | {'yes' if same else 'no'} |")
    lines += [
        "",
        "## Per comparison",
        "",
        "| Comparison | Kind | Key bugs | Noise floor | "
        + " | ".join(f"Rung {m[-1]}" for m in MAIN)
        + " |",
        "| --- | --- | --- | --- | " + " | ".join("---" for _ in MAIN) + " |",
    ]
    wide = (
        sc[sc.method.isin(MAIN)]
        .pivot_table(
            index=["cid", "kind", "key_bugs", "noise_floor"], columns="method", values="error"
        )
        .reset_index()
    )
    for row in wide.sort_values("key_bugs", ascending=False).to_dict(orient="records"):
        lines.append(
            f"| `{row['cid']}` | {row['kind']} | {int(row['key_bugs']):,} | {_p(row['noise_floor'])} | "
            + " | ".join(f"{row.get(m, float('nan')) * 100:+.1f}" for m in MAIN)
            + " |"
        )
    path = path or cfg.root / "docs" / "one_try" / "07_evaluation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _lab(m: str) -> str:
    return LABELS.get(m, m)


def a8_lines(a8: dict[str, Any], r: dict[str, Any]) -> list[str]:
    """Report sections for amendments A8 and A9 (secondary; H1-H4 unchanged)."""
    lines = [
        "## Secondary scores (amendment A8; none changes H1–H4)",
        "",
        "### Noise-corrected bias",
        "",
        "*Error floor = how far a perfect method would still land from the answer key by chance "
        "(joint bootstrap of bugs: the answer key and the one-try estimates are redrawn together). "
        "Bias = the error that is left once that chance part is taken out (root mean square over "
        "comparisons). Clearly off = share of comparisons whose error is more than 1.96 floors.*",
        "",
        "| Method | Comparisons | RMS error | Median floor | Bias (noise taken out) | Clearly off |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for q in a8["bias"].to_dict(orient="records"):
        lines.append(
            f"| {_lab(q['method'])} | {q['comparisons']} | {_p(q['rms_error'])} | "
            f"{_p(q['median_floor'])} | {_p(q['bias_rms'])} | {q['clearly_off']:.0%} |"
        )
    lines += [
        "",
        "### Against the answer key that weights bugs like the method",
        "",
        "*Partialling-out ML weights each bug by p(1 − p), AIPW weights bugs equally; the frozen "
        "answer key uses n_a·n_b/(n_a + n_b). Here each method is scored against the key with its "
        "own weights.*",
        "",
        "| Method | Average distance, frozen key | Average distance, aligned key | Bias, aligned key |",
        "| --- | --- | --- | --- |",
    ]
    ba = {q["method"]: q for q in a8["bias_aligned"].to_dict(orient="records")}
    frozen = {q["method"]: q for q in a8["bias"].to_dict(orient="records")}
    for m, v in a8["aligned_mae"].items():
        if m in ba:
            lines.append(
                f"| {_lab(m)} | {_p(frozen[m]['rms_error']) if m in frozen else '—'} (RMS) | "
                f"{_p(v)} | {_p(ba[m]['bias_rms'])} |"
            )
    lines += [
        "",
        "### Paired contrasts",
        "",
        "*|error A| − |error B| on the same comparisons (positive = B is closer). Intervals: per-setup "
        "t (setups − 1 degrees of freedom) and Webb wild cluster bootstrap; LOSO = range when one "
        "setup is left out. NVIDIA setups share bugs and models, so setups are not fully independent.*",
        "",
        "| A | B | Comparisons | Mean | t interval | Wild interval | LOSO range |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for q in a8["contrasts"].to_dict(orient="records"):
        lines.append(
            f"| {_lab(q['a'])} | {_lab(q['b'])} | {q['comparisons']} | {_p(q['mean'])} | "
            f"{_p(q['t_lo'])} to {_p(q['t_hi'])} | {_p(q['wild_lo'])} to {_p(q['wild_hi'])} | "
            f"{_p(q['loso_min'])} to {_p(q['loso_max'])} |"
        )
    lines += [
        "",
        "### Few-cluster intervals for the average distance",
        "",
        "| Method | Mean | Setups | t interval (per-setup means) | Wild interval | LOSO range |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for q in a8["few_clusters"].to_dict(orient="records"):
        lines.append(
            f"| {_lab(q['method'])} | {_p(q['mean'])} | {q['setups']} | {_p(q['t_lo'])} to "
            f"{_p(q['t_hi'])} | {_p(q['wild_lo'])} to {_p(q['wild_hi'])} | {_p(q['loso_min'])} "
            f"to {_p(q['loso_max'])} |"
        )
    lines += [
        "",
        "### Interval quality",
        "",
        "*Coverage rewards wide intervals, so we also report width, the interval score (width plus "
        "40 × the distance by which the interval misses the answer key; lower is better), detection "
        "(answer keys with a clear direction: share of draws reaching it) and false alarms (answer "
        "keys with no clear direction: share of draws claiming one).*",
        "",
        "| Method | Width | Interval score | Detection | False alarms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for q in a8["intervals"].to_dict(orient="records"):
        lines.append(
            f"| {_lab(q['method'])} | {_p(q['width'])} | {_p(q['score'])} | "
            f"{q['detect']:.0%} | {q['false_alarm']:.0%} |"
        )
    lines += [
        "",
        "### How well the guessers guess (model diagnostics)",
        "",
        "*R² of the fix model; AUC of the choice model (0.5 = no better than chance); choice left "
        "over = share of the choice's variation the model could not explain (the variation the "
        "estimate is built from).*",
        "",
        "| Method | R² (fix) | AUC (choice) | Choice left over |",
        "| --- | --- | --- | --- |",
    ]
    for q in a8["diagnostics"].to_dict(orient="records"):
        lines.append(
            f"| {_lab(q['method'])} | {q['r2_y']:.3f} | {q['auc_t']:.3f} | {q['t_left']:.2f} |"
        )
    lines += [
        "",
        f"### Sensitivity: precise answer keys only (noise floor ≤ 1.5 points, "
        f"{a8['sensitivity_comparisons']} comparisons)",
        "",
        "| Method | Comparisons | Average distance | 95% bounds |",
        "| --- | --- | --- | --- |",
    ]
    for q in a8["sensitivity"]:
        lines.append(
            f"| {_lab(q['method'])} | {q['comparisons']} | {_p(q['mean_abs_error'])} | "
            f"{_p(q['lo95'])}–{_p(q['hi95'])} |"
        )
    ab = a8["allbugs"]
    lines += [
        "",
        "### All-bugs copies: details",
        "",
        "- Share of the ML weight on bugs that are **not** key bugs: "
        + ", ".join(f"{_lab(k)} {v:.0%}" for k, v in ab["nonkey_weight"].items())
        + ".",
        f"- Gradient boosting's automatic early stopping (on above 10,000 rows) was on in "
        f"{ab['early_stopping_copies']} of {ab['ml_copies']} all-bugs ML fits (NVIDIA copies).",
        "",
        "## Amendment A9 checks",
        "",
    ]
    a9 = r.get("a9")
    if a9:
        lines.append(
            f"- **Sealed first run** ({a9['comparisons']} comparisons; hashes "
            f"{'intact' if a9['hashes_ok'] else '**changed**'}): rungs 0–1 identical and ML rungs "
            f"within 0.3 points (largest difference {_p(a9['max_diff_dml'])}): "
            + ("**passed**." if a9["passed"] else "**not passed**.")
        )
    cc = r.get("crosscheck")
    if cc:
        lines.append(
            f"- **Across computers** ({len(cc['rows'])} draws re-run here): "
            + ("**passed**." if cc["passed"] else "**not passed**.")
        )
    elif a9 is not None:
        lines.append("- **Across computers:** the whole run was on one computer.")
    return lines


def evaluate(cfg: Config, p: Params, log: Callable[[str], None] = print) -> Path:
    """Score all units, test H1-H4, run the checks and write the report."""
    res = cfg.path("onetry_results")
    units = load_units(cfg)
    keys = pd.read_parquet(res / "answer_keys.parquet")
    inv = pd.read_parquet(res / "inventory.parquet")
    margin = float(cfg.raw["onetry"]["margin"])
    reps = int(cfg.raw["onetry"]["bootstrap_reps"])
    main_units = units[units.draw < p.draws]
    sc = scores(main_units, keys)
    summ = summary_table(sc, MAIN + SECONDARY_RUNGS, reps, cfg.seed)
    log("hypotheses")
    hyp = hypotheses(sc, summ, inv, main_units, keys, margin, reps, cfg.seed)
    early = scores(units[units.draw < p.check_draws], keys)
    checks = {}
    for check, main_m in (
        ("rung5_forest", "rung5"),
        ("rung5_real", "rung5"),
        ("rung7", "rung5"),
        ("rung2_aipw", "rung2"),
        ("rung4_aipw", "rung4"),
        ("rung5_aipw", "rung5"),
        ("rung3", "rung2"),
    ):
        a, m = early[early.method == check], early[early.method == main_m]
        if len(a):
            checks[check] = {
                "check": float(a.abs_error.mean()),
                "main": float(m[m.cid.isin(a.cid)].abs_error.mean()),
            }
    sec_units = units[units.draw < p.secondary_draws]
    sec = scores(sec_units, keys)
    secondary = {
        m: {
            "key": float(sec[sec.method == m].abs_error.mean()),
            "all": float(sec[sec.method == f"{m}_allbugs"].abs_error.mean()),
        }
        for m in ("rung0", "rung1", "rung2", "rung5")
    }
    for name, base in (
        ("rung0_allbugs_trim", "rung0"),
        ("rung1_allbugs_trim", "rung1"),
        ("rung5_aipw_allbugs", "rung5_aipw"),
        ("rung5_aipw_allbugs_trim", "rung5_aipw"),
    ):
        secondary[name] = {
            "key": float(sec[sec.method == base].abs_error.mean()),
            "all": float(sec[sec.method == name].abs_error.mean()),
        }
    kt = sc[sc.method.str.startswith("ktry")].assign(k=lambda x: x.method.str[4:].astype(int))
    rq4 = [
        {
            "dataset": ds,
            "k": int(float(str(k))),
            "abs_error": float(g.abs_error.mean()),
            "comparisons": int(g.cid.nunique()),
        }
        for (ds, k), g in kt.groupby(["dataset", "k"])
    ]
    full = scores(units, keys)
    prec = {
        m: (
            float(sc[sc.method == m].abs_error.mean()),
            float(full[full.method == m].abs_error.mean()),
        )
        for m in ("rung0", "rung5")
    }
    prec_passed = all(abs(a - b) < 0.002 for a, b in prec.values())
    log("reproducibility re-runs")
    repro = reproducibility(cfg, p)
    log("amendment A8: secondary scores")
    a8 = secondary_scores(cfg, p, units, main_units, sc, keys, margin, reps)
    report = {
        "margin": margin,
        "params": asdict(p),
        "hypotheses": hyp,
        "checks": checks,
        "secondary": secondary,
        "rq4": rq4,
        "precision": {**prec, "passed": prec_passed},
        "repro": repro,
        "a8": {k: v for k, v in a8.items() if not isinstance(v, pd.DataFrame)},
        "a9": a9_check(cfg) if sealed_cids(cfg) else None,
        "crosscheck": (
            json.loads((res / "crosscheck.json").read_text(encoding="utf-8"))
            if (res / "crosscheck.json").is_file()
            else None
        ),
    }
    sc.to_parquet(res / "results.parquet", index=False)
    (res / "evaluation.json").write_text(
        json.dumps(
            report, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
        ),
        encoding="utf-8",
    )
    chart_ladder(summ, margin, cfg)
    chart_k(hyp["H4"], cfg)
    chart_ktry(sc, cfg)
    path = write_report(cfg, report, summ, sc)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + "\n".join(a8_lines(a8, report)) + "\n")
    return path


def secondary_scores(
    cfg: Config,
    p: Params,
    units: pd.DataFrame,
    main_units: pd.DataFrame,
    sc: pd.DataFrame,
    keys: pd.DataFrame,
    margin: float,
    reps: int,
) -> dict[str, Any]:
    """Amendment A8: every secondary score (none changes H1-H4)."""
    res = cfg.path("onetry_results")
    s = _study(cfg)
    methods = MAIN + SECONDARY_RUNGS
    ak = extra.aligned_keys(s, reps)
    al = extra.aligned_scores(main_units, keys, ak)
    al_mae = al.assign(a=al.aligned_error.abs()).groupby("method").a.mean().to_dict()
    fl, diag = extra.floors(cfg, s, _slug, p.draws)
    bias = extra.bias_table(fl, methods + extra.AIPW_METHODS, "cmh")
    fl_al = fl.assign(error_al=np.nan, floor_al=np.nan)
    for w in ("dml", "ate"):
        if f"error_{w}" in fl:
            hit = fl.method.map(extra.target_of) == w
            fl_al.loc[hit, "error_al"] = fl.loc[hit, f"error_{w}"]
            fl_al.loc[hit, "floor_al"] = fl.loc[hit, f"floor_{w}"]
    bias_al = extra.bias_table(fl_al, methods[2:] + extra.AIPW_METHODS, "al")
    pairs = [
        ("rung0", "rung2"),
        ("rung2", "rung5"),
        ("rung0", "rung5"),
        ("rung2", "rung4"),
        ("rung4", "rung5"),
        ("rung5_real", "rung5"),
        ("rung5", "rung7"),
    ]
    con = extra.contrasts(sc, pairs, 9999, cfg.seed)
    few = pd.DataFrame(
        [
            {"method": m, **extra.cluster_ci(g.abs_error, g.setup, 9999, cfg.seed)}
            for m, g in sc[sc.method.isin(methods)].groupby("method")
        ]
    )
    iq = extra.interval_quality(main_units, keys, methods + extra.AIPW_METHODS)
    precise = keys[keys.noise_floor <= extra.PRECISE].cid
    sens_sc = sc[sc.cid.isin(precise)]
    sens = summary_table(sens_sc, MAIN, reps, cfg.seed) if len(sens_sc) else pd.DataFrame()
    alld = extra.allbugs_details(cfg, s, units, _slug, p.secondary_draws)
    dg = diag.groupby("method")[["r2_y", "auc_t", "t_left"]].mean().reset_index()
    extra.write_parquets(
        res,
        aligned_keys=ak,
        aligned_errors=al,
        floors=fl,
        diagnostics=diag,
        contrasts=con,
        few_clusters=few,
        intervals=iq,
    )
    sens_rows = sens.drop(columns="boot").to_dict(orient="records") if len(sens) else []
    return {
        "aligned_mae": al_mae,
        "bias": bias,
        "bias_aligned": bias_al,
        "contrasts": con,
        "few_clusters": few,
        "intervals": iq,
        "diagnostics": dg,
        "sensitivity": sens_rows,
        "sensitivity_comparisons": int(len(precise)),
        "allbugs": alld,
        "margin": margin,
    }


def run(cfg: Config, fresh: bool = False, log: Callable[[str], None] = print) -> Path:
    """Full run (resumable), then scores, tests, checks and the report."""
    p = Params.from_config(cfg)
    run_all(cfg, p, fresh=fresh, log=log)
    return evaluate(cfg, p, log)
