"""Step 7 (one-try study): the full run, the scores and the hypothesis tests.

**Full run.** Every qualifying comparison is one unit of work, run in parallel
and saved to ``outputs/onetry/units/<comparison>.parquet`` (so a stopped run
resumes). Per comparison (draw numbers per ``01_problem.md`` and amendments):

- draws 0-19: rungs 0, 1, 2, 4, 5, 6 on one-try copies of the key bugs;
- draws 20-39: rungs 0 and 5 (precision check, A4);
- draws 0-4: rung 3, random-forest rung 5, AIPW rungs 2, 4, 5 (checks, A1);
- NVIDIA, draws 0-19: rung 5 using only 1, 2 or 4 other setups (H4b);
- draws 0-4: rungs 0, 1, 2, 5 on copies drawn from all bugs (secondary, A7);
- older AI and Nebius, draws 0-19: the within-bug answer from k = 2, 3, 5, 10
  tries per bug (RQ4).

**Scores and tests** follow ``01_problem.md`` §6-7: error = mean over draws of
(estimate − answer key); averages over comparisons with CIs from bootstrap
resamples of AI setups; H1-H4 with Holm correction.
"""

from __future__ import annotations

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
from agent_tips.onetry import data  # noqa: E402
from agent_tips.onetry.comparisons import NEBIUS, NVIDIA, OLDER  # noqa: E402
from agent_tips.onetry.features import other_ai_rates, tokenize  # noqa: E402
from agent_tips.onetry.simulate import k_try_answer, k_try_copy, one_try_copy, seed_for  # noqa: E402

MAIN = ["rung0", "rung1", "rung2", "rung4", "rung5", "rung6"]
LABELS = {
    "rung0": "Simple comparison",
    "rung1": "Linear regression on report facts",
    "rung2": "ML: report facts + text",
    "rung3": "ML: + language-model embeddings",
    "rung4": "ML: + human-fix size",
    "rung5": "ML: + other AIs' results",
    "rung6": "ML: + IRT difficulty",
}


@dataclass(frozen=True)
class Params:
    """How much to run (from ``configs/default.yaml`` → ``onetry``)."""

    draws: int = 20
    precision_draws: int = 40
    check_draws: int = 5
    secondary_draws: int = 5
    folds: int = 5
    threads: int = 2
    jobs: int = 5
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


def _study(cfg: Config) -> data.StudyData:
    key = str(cfg.root)
    if key not in _STUDY:
        _STUDY[key] = data.load(cfg)
    return _STUDY[key]


def run_unit(cfg: Config, cid: str, p: Params) -> pd.DataFrame:
    """All estimates for one comparison (one row per draw × method)."""
    s = _study(cfg)
    c = next(x for x in s.comparisons if x.cid == cid)
    rows: list[dict[str, Any]] = []

    def add(draw: int, method: str, r: cr.Result) -> None:
        rows.append(
            {"cid": cid, "draw": draw, "method": method, "estimate": r.estimate, "se": r.se}
        )

    has_emb = s.embeddings is not None
    with threadpool_limits(p.threads):
        first = data.with_features(one_try_copy(c, s.tries, 0), c, s)
        tokens = tokenize(first.report)
        bugs = pd.Index(first.bug_id)
        others = sorted(
            s.tries.loc[(s.tries.setup != c.setup) & s.tries.bug_id.isin(bugs), "setup"].unique()
        )
        for draw in range(p.precision_draws):
            main, check = draw < p.draws, draw < p.check_draws
            d = data.with_features(
                one_try_copy(c, s.tries, draw), c, s, embeddings=check and has_emb
            )
            assert d.bug_id.tolist() == first.bug_id.tolist()  # tokens stay aligned
            folds = cr.make_folds(len(d), p.folds, draw)
            text = cr.text_folds(d, folds, 100, draw, tokens)
            add(draw, "rung0", cr.simple(d))
            if main:
                add(draw, "rung1", cr.linear(d))
            for r in (2, 4, 5, 6) if main else (5,):
                add(draw, f"rung{r}", cr.dml(d, r, draw, folds=folds, text=text))
            if check:
                if has_emb:
                    add(draw, "rung3", cr.dml(d, 3, draw, folds=folds, text=text))
                add(
                    draw,
                    "rung5_forest",
                    cr.dml(d, 5, draw, learner="forest", folds=folds, text=text),
                )
                for r in (2, 4, 5):
                    add(draw, f"rung{r}_aipw", cr.aipw(d, r, draw, folds=folds, text=text))
            if main and c.dataset == NVIDIA:
                for k in p.h4_ks:
                    if k >= len(others):
                        continue
                    rng = np.random.default_rng(seed_for(cid, draw, 50 + k))
                    chosen = set(rng.choice(others, size=k, replace=False))
                    sub = s.tries[s.tries.setup.isin(chosen)]
                    oa = other_ai_rates(sub, c.setup, bugs).reset_index()
                    dk = d.drop(columns=cr.OTHER).merge(oa, on="bug_id", how="left")
                    add(draw, f"rung5_k{k}", cr.dml(dk, 5, draw, folds=folds, text=text))
        tokens_all: list[list[str]] | None = None
        for draw in range(p.secondary_draws):
            d = data.with_features(one_try_copy(c, s.tries, draw, key_only=False), c, s)
            tokens_all = tokens_all or tokenize(d.report)
            folds = cr.make_folds(len(d), p.folds, draw)
            text = cr.text_folds(d, folds, 100, draw, tokens_all)
            add(draw, "rung0_allbugs", cr.simple(d))
            add(draw, "rung1_allbugs", cr.linear(d))
            for r in (2, 5):
                add(draw, f"rung{r}_allbugs", cr.dml(d, r, draw, folds=folds, text=text))
        if c.dataset in (OLDER, NEBIUS):
            for k in p.k_tries:
                for draw in range(p.draws):
                    kd = k_try_copy(c, s.tries, k, draw)
                    if kd.bug_id.nunique() >= p.min_ktry_bugs:
                        add(draw, f"ktry{k}", cr.Result(k_try_answer(c, kd), float("nan")))
    return pd.DataFrame(rows)


def _slug(cid: str) -> str:
    return cid.replace("/", "__").replace("|", "--")


def _unit_to_file(cfg: Config, cid: str, p: Params, path: Path, log_path: Path) -> str:
    t = time.time()
    out = run_unit(cfg, cid, p)
    tmp = path.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"{time.strftime('%H:%M:%S')} done {cid} ({len(out)} rows, {time.time() - t:.0f}s)\n"
        )
    return cid


def run_all(
    cfg: Config, p: Params, fresh: bool = False, log: Callable[[str], None] = print
) -> Path:
    """Run every comparison not yet done (largest first), in parallel."""
    units = cfg.path("onetry_results") / "units"
    units.mkdir(parents=True, exist_ok=True)
    log_path = units / "_progress.log"
    s = _study(cfg)
    keys = pd.read_parquet(cfg.path("onetry_results") / "answer_keys.parquet")
    size = dict(zip(keys.cid, keys.key_bugs, strict=True))
    todo = sorted(
        (
            c.cid
            for c in s.comparisons
            if fresh or not (units / f"{_slug(c.cid)}.parquet").is_file()
        ),
        key=lambda cid: -size.get(cid, 0),
    )
    log(f"full run: {len(todo)} of {len(s.comparisons)} comparisons to do, {p.jobs} at a time")
    Parallel(n_jobs=p.jobs, backend="loky")(
        delayed(_unit_to_file)(cfg, cid, p, units / f"{_slug(cid)}.parquet", log_path)
        for cid in todo
    )
    return units


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


def reproducibility(cfg: Config, p: Params, n: int = 3, seed: int = 0) -> dict[str, Any]:
    """Re-run ``n`` random comparisons in full; every row must match (amendment A3)."""
    units = cfg.path("onetry_results") / "units"
    s = _study(cfg)
    rng = np.random.default_rng(seed)
    chosen = sorted(
        rng.choice([c.cid for c in s.comparisons], size=min(n, len(s.comparisons)), replace=False)
    )
    same = {}
    for cid in chosen:
        a = pd.read_parquet(units / f"{_slug(cid)}.parquet")
        b = run_unit(cfg, cid, p)
        key = ["draw", "method"]
        a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
        same[cid] = bool(
            a.shape == b.shape
            and np.allclose(a.estimate, b.estimate, equal_nan=True, atol=1e-12, rtol=0)
        )
    return {"comparisons": chosen, "identical": same, "all_identical": all(same.values())}


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
        "`01_problem.md` (tag `one-try-plan-v1`, amendments A1–A6).*",
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
    for m in [x for x in MAIN + ["rung3"] if x in rows_by_method]:
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
        lines.append(f"| {LABELS[m]} | {_p(v['key'])} | {_p(v['all'])} |")
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


def evaluate(cfg: Config, p: Params, log: Callable[[str], None] = print) -> Path:
    """Score all units, test H1-H4, run the checks and write the report."""
    res = cfg.path("onetry_results")
    units = pd.concat(
        [pd.read_parquet(f) for f in sorted((res / "units").glob("*.parquet"))], ignore_index=True
    )
    keys = pd.read_parquet(res / "answer_keys.parquet")
    inv = pd.read_parquet(res / "inventory.parquet")
    margin = float(cfg.raw["onetry"]["margin"])
    reps = int(cfg.raw["onetry"]["bootstrap_reps"])
    main_units = units[units.draw < p.draws]
    sc = scores(main_units, keys)
    summ = summary_table(sc, MAIN + ["rung3"], reps, cfg.seed)
    log("hypotheses")
    hyp = hypotheses(sc, summ, inv, main_units, keys, margin, reps, cfg.seed)
    early = scores(units[units.draw < p.check_draws], keys)
    checks = {}
    for check, main_m in (
        ("rung5_forest", "rung5"),
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
    report = {
        "margin": margin,
        "params": asdict(p),
        "hypotheses": hyp,
        "checks": checks,
        "secondary": secondary,
        "rq4": rq4,
        "precision": {**prec, "passed": prec_passed},
        "repro": repro,
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
    return write_report(cfg, report, summ, sc)


def run(cfg: Config, fresh: bool = False, log: Callable[[str], None] = print) -> Path:
    """Full run (resumable), then scores, tests, checks and the report."""
    p = Params.from_config(cfg)
    run_all(cfg, p, fresh=fresh, log=log)
    return evaluate(cfg, p, log)
