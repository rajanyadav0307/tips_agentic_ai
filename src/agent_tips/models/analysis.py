"""Step 6: run every hypothesis in ``docs/01_problem_and_hypotheses.md`` (tag ``analysis-plan-v1``).

Writes ``outputs/results/estimates.parquet``: one row per estimate, with its
RQ, hypothesis ID, AI, comparison, design (fair / simple / fe_lpm / fork /
...), estimate, 95% CI, 90% CI, p-value, q-value (secondary hypotheses only),
and sample sizes. Also writes ``outputs/results/analysis_summary.json`` with
decisions and design checks.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from agent_tips.config import Config
from agent_tips.models import forks as fk
from agent_tips.models import survival as sv
from agent_tips.models.estimators import (
    Estimate,
    benjamini_hochberg,
    build_strata,
    fair_estimate,
    simple_estimate,
)
from agent_tips.models.regression import fe_interactions, fe_lpm

OLDER, NEBIUS, NVIDIA = "older_ai", "newer_ai_nebius", "newer_ai_nvidia"
SEARCH = ("search_dir", "search_file", "find_file", "ls", "grep", "find")
DAY = ["batch_day"]


@dataclass
class Spec:
    """One comparison to estimate."""

    hid: str
    rq: str
    label: str
    data: pd.DataFrame
    arm_col: str
    a: str
    b: str
    y: str = "y"
    strata: Sequence[str] = field(default_factory=list)
    ai: str = "all"
    primary: bool = False
    perm: bool = False
    gap_pair: bool = False


# ---------------------------------------------------------------- data


def load(cfg: Config) -> pd.DataFrame:
    """Analysis-ready tries joined with their features and issue features."""
    proc = cfg.path("processed")
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    df = con.execute(f"""
      SELECT t.*, f.* EXCLUDE (try_id),
        i.has_code_block, i.has_stack_trace, i.names_file, i.text_length,
        s.command AS cmd0
      FROM read_parquet('{(proc / "tries.parquet").as_posix()}') t
      JOIN read_parquet('{(proc / "try_features.parquet").as_posix()}') f USING (try_id)
      LEFT JOIN read_parquet('{(proc / "issues.parquet").as_posix()}') i USING (issue_key)
      LEFT JOIN (SELECT try_id, command FROM read_parquet('{(proc / "steps").as_posix()}/*.parquet')
                 WHERE step = 0) s USING (try_id)
      WHERE t.in_scope AND t.same_input AND t.outcome_known
      ORDER BY try_id""").df()  # fixed row order: same random draws every run
    df["y"] = df["resolved"].astype(float)
    edited = df["first_edit_code"].notna()
    for name, col in [
        ("rerun_after_fix", "n_reruns_after_fix"),
        ("suite_after_fix", "n_suite_after_fix"),
        ("run_after_fix", "n_runs_after_fix"),
        ("tests_after_fix", "n_tests_after_fix"),
        ("suite_before_fix", "n_suite_before_fix"),
    ]:
        df[name] = pd.Series(np.where(df[col] > 0, "yes", "no"), index=df.index).where(edited)
    df["cleanup"] = np.where(df["n_cleanup"] > 0, "yes", "no")
    fc, fe = df["first_create"].astype(float), df["first_edit_code"].astype(float)
    df["order_arm_broad"] = np.select(
        [fe.isna(), fc.notna() & (fc < fe)], ["no edit", "test first"], "fix first"
    )
    df["exit_status"] = df["exit_status"].fillna("")
    df["ctx_submit"] = (df["exit_status"] == "submitted (exit_context)").astype(float)
    df["ctx_any"] = df["exit_status"].str.contains("exit_context").astype(float)
    df["chars_in_k"] = df["chars_in"] / 1000
    df["chars_out_k"] = df["chars_out"] / 1000
    med = df.groupby("dataset")["text_length"].transform("median")
    df["long_issue"] = df["text_length"] > med
    df["placebo"] = np.select(
        [
            df["cmd0"].str.strip() == "create reproduce.py",
            df["cmd0"].str.strip() == "create reproduce_bug.py",
        ],
        ["reproduce.py", "reproduce_bug.py"],
        "other",
    )
    df["first_tool"] = np.where(df["tool0"].isin(SEARCH + ("create", "open")), df["tool0"], "other")
    df["first_move_nebius"] = np.select(
        [df["tool0"] == "think", df["label0"] == "look"], ["think", "view"], "other"
    )
    last = df["n_steps"].clip(lower=1) - 1
    df["gold_event"] = df["first_gold_step"].notna()
    df["gold_time"] = np.where(df["gold_event"], df["first_gold_step"], last).astype(int)
    return df


# ---------------------------------------------------------------- helpers


def _row(spec: Spec, design: str, e: Estimate | None, **extra: Any) -> dict[str, Any]:
    base = {
        "hid": spec.hid,
        "rq": spec.rq,
        "label": spec.label,
        "ai": spec.ai,
        "comparison": f"{spec.a} minus {spec.b}",
        "outcome": spec.y,
        "design": design,
        "primary": spec.primary,
    }
    if e is None:
        return {**base, **extra}
    return {**base, **e.as_dict(), **extra}


def gap_draws(
    spec: Spec, rng: np.random.Generator, reps: int
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return simple and fair estimates with joint cluster-bootstrap draws (same issues)."""
    d = spec.data[spec.data[spec.arm_col].isin([spec.a, spec.b])]
    if d.empty or d[spec.arm_col].nunique() < 2:
        nan = np.full(reps, np.nan)
        return float("nan"), float("nan"), nan, nan
    g = d.groupby(["issue_key", spec.arm_col])[spec.y].agg(["size", "sum"]).unstack(spec.arm_col)
    g = g.fillna(0)
    issues = g.index.to_numpy()
    na, nb = g[("size", spec.a)].to_numpy(float), g[("size", spec.b)].to_numpy(float)
    ka, kb = g[("sum", spec.a)].to_numpy(float), g[("sum", spec.b)].to_numpy(float)
    st = build_strata(d, spec.strata, spec.arm_col, spec.a, spec.b, spec.y)
    pos = {k: i for i, k in enumerate(issues)}
    num, den = np.zeros(len(issues)), np.zeros(len(issues))
    if len(st):
        idx = np.array([pos[i] for i in st.issue])
        np.add.at(num, idx, st.weights * st.diffs)
        np.add.at(den, idx, st.weights)
    simple = ka.sum() / na.sum() - kb.sum() / nb.sum()
    fair = num.sum() / den.sum() if den.sum() > 0 else np.nan
    draws = rng.integers(0, len(issues), size=(reps, len(issues)))
    with np.errstate(invalid="ignore", divide="ignore"):
        s_d = ka[draws].sum(1) / na[draws].sum(1) - kb[draws].sum(1) / nb[draws].sum(1)
        f_d = num[draws].sum(1) / den[draws].sum(1)
    return float(simple), float(fair), s_d, f_d


def hour_balance(df: pd.DataFrame, rng: np.random.Generator, reps: int) -> dict[str, float]:
    """Test whether, within issue x day, the evaluation hour predicts the first move (R vs S)."""
    d = df[(df.dataset == OLDER) & df.dated & df.first_move_arm.isin(["R", "S"])].copy()
    d["sid"] = d.groupby(["issue_key", "batch_day"]).ngroup()
    d["hid"] = d.groupby(["issue_key", "batch_day", "batch_hour"]).ngroup()
    d = d[
        (d.groupby("sid").hid.transform("nunique") > 1)
        & (d.groupby("sid").first_move_arm.transform("nunique") > 1)
    ]
    sid, hid = pd.factorize(d.sid)[0], pd.factorize(d.hid)[0]
    is_r = (d.first_move_arm == "R").to_numpy(float)
    pr_s = np.bincount(sid, weights=is_r) / np.bincount(sid)
    n_h = np.bincount(hid)
    s_of_h = np.zeros(len(n_h), dtype=int)
    s_of_h[hid] = sid

    def stat(a: np.ndarray) -> float:
        nr = np.bincount(hid, weights=a, minlength=len(n_h))
        return float(np.sum((nr - n_h * pr_s[s_of_h]) ** 2 / n_h))

    obs = stat(is_r)
    labels = is_r[np.argsort(sid, kind="stable")]
    null = np.empty(reps)
    for r in range(reps):
        order = np.lexsort((rng.random(len(sid)), sid))
        a = np.empty_like(is_r)
        a[order] = labels
        null[r] = stat(a)
    return {
        "groups": int(len(np.unique(sid))),
        "tries": int(len(sid)),
        "ratio_to_chance": float(obs / null.mean()),
        "p_value": float(np.mean(null >= obs)),
    }


def load_forks(cfg: Config, tries: pd.DataFrame) -> pd.DataFrame:
    """Identical-history fork pairs for the NVIDIA tries (first 30 actions)."""
    proc = cfg.path("processed")
    tmp = cfg.path("interim") / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    nv = tries[tries.dataset == NVIDIA][["try_id", "issue_key", "y", "subset"]]
    nv = nv[nv.groupby("issue_key").try_id.transform("size") >= 2]  # only issues that can pair
    steps = (proc / "steps" / f"{NVIDIA}.parquet").as_posix()
    labels = (proc / "labels" / f"{NVIDIA}.parquet").as_posix()
    out = []
    for _sub, part in nv.groupby("subset"):  # one AI at a time keeps memory bounded
        con.register("eligible", part[["try_id"]])
        seq = con.execute(f"""
          SELECT s.try_id, list(s.args_hash ORDER BY s.step) AS hashes,
                 list(l.label ORDER BY s.step) AS kinds
          FROM (SELECT try_id, step, args_hash FROM read_parquet('{steps}')
                WHERE step < 30 AND try_id IN (SELECT try_id FROM eligible)) s
          JOIN (SELECT try_id, step, label FROM read_parquet('{labels}')
                WHERE step < 30 AND try_id IN (SELECT try_id FROM eligible)) l
            USING (try_id, step)
          GROUP BY s.try_id ORDER BY s.try_id""").df()
        con.unregister("eligible")
        out.append(fk.fork_pairs(part.merge(seq, on="try_id")))
    return (
        pd.concat(out, ignore_index=True) if out else fk.fork_pairs(nv.assign(hashes=[], kinds=[]))
    )


# ---------------------------------------------------------------- specs


def build_specs(df: pd.DataFrame) -> list[Spec]:
    """All comparisons from the frozen plan, in order."""
    older = df[df.dataset == OLDER]
    dated = older[older.dated]
    nv = df[df.dataset == NVIDIA]
    neb = df[df.dataset == NEBIUS]
    nv_edit = nv[nv.first_edit_code.notna()]
    specs: list[Spec] = [
        Spec(
            "P1",
            "RQ2",
            "Reproduce first (older AI, first step)",
            dated,
            "first_move_arm",
            "R",
            "S",
            strata=DAY,
            ai=OLDER,
            primary=True,
            perm=True,
            gap_pair=True,
        ),
        Spec(
            "P2",
            "RQ2",
            "Test first (newer AIs, pooled)",
            nv,
            "order_arm",
            "test first",
            "fix first",
            ai="NVIDIA pooled",
            primary=True,
            perm=True,
        ),
    ]
    for sub, g in nv.groupby("subset"):
        specs.append(
            Spec(
                "P2-sub",
                "RQ2",
                "Test first, per AI",
                g,
                "order_arm",
                "test first",
                "fix first",
                ai=str(sub),
                gap_pair=True,
            )
        )
    # S1-S6 tips after/around the fix
    tips = [
        ("S1", "Check your fix (re-ran own test script after fixing)", "rerun_after_fix"),
        ("S2", "Ran the project's test suite after fixing", "suite_after_fix"),
        ("S3", "Ran any code after fixing", "run_after_fix"),
        ("S4", "Wrote extra tests after fixing", "tests_after_fix"),
        ("S5", "Cleaned up created files", "cleanup"),
        ("S6", "Ran the project's test suite before fixing", "suite_before_fix"),
    ]
    older_edit = dated[dated.first_edit_code.notna()]
    for hid, label, col in tips:
        if hid in ("S1", "S2"):
            specs.append(
                Spec(
                    hid,
                    "RQ4",
                    label,
                    older_edit,
                    col,
                    "yes",
                    "no",
                    strata=DAY,
                    ai=OLDER,
                    gap_pair=True,
                )
            )
        for sub, g in nv_edit.groupby("subset"):
            specs.append(Spec(hid, "RQ4", label, g, col, "yes", "no", ai=str(sub), gap_pair=True))
    # S7 cost of reproducing / testing first
    specs += [
        Spec(
            "S7",
            "RQ4",
            "Cost: steps",
            dated,
            "first_move_arm",
            "R",
            "S",
            y="n_steps",
            strata=DAY,
            ai=OLDER,
        ),
        Spec(
            "S7",
            "RQ4",
            "Cost: steps",
            nv,
            "order_arm",
            "test first",
            "fix first",
            y="n_steps",
            ai="NVIDIA pooled",
        ),
        Spec(
            "S7",
            "RQ4",
            "Cost: characters read (thousands)",
            nv,
            "order_arm",
            "test first",
            "fix first",
            y="chars_in_k",
            ai="NVIDIA pooled",
        ),
        Spec(
            "S7",
            "RQ4",
            "Cost: characters written (thousands)",
            nv,
            "order_arm",
            "test first",
            "fix first",
            y="chars_out_k",
            ai="NVIDIA pooled",
        ),
    ]
    # S9 by language (per-language fair estimates)
    for lang, g in nv.groupby("language"):
        if g.issue_key.nunique() >= 300:
            specs.append(
                Spec(
                    "S9",
                    "RQ5",
                    f"Test first, language = {lang}",
                    g,
                    "order_arm",
                    "test first",
                    "fix first",
                    ai="NVIDIA pooled",
                )
            )
    # S10 mechanism: running out of context
    specs += [
        Spec(
            "S10",
            "RQ6",
            "Submitted because the context ran out",
            dated,
            "first_move_arm",
            "R",
            "S",
            y="ctx_submit",
            strata=DAY,
            ai=OLDER,
        ),
        Spec(
            "S10",
            "RQ6",
            "Any context-limit exit",
            dated,
            "first_move_arm",
            "R",
            "S",
            y="ctx_any",
            strata=DAY,
            ai=OLDER,
        ),
    ]
    # S12 Nebius first move (exact coin flip)
    specs.append(
        Spec(
            "S12",
            "RQ2",
            "First move think vs view (Qwen3-Coder)",
            neb,
            "first_move_nebius",
            "think",
            "view",
            ai=NEBIUS,
            gap_pair=True,
        )
    )
    # S13 older first moves vs search_dir
    for tool in ("find_file", "ls", "open", "grep"):
        specs.append(
            Spec(
                "S13",
                "RQ3",
                f"First move {tool} vs search_dir",
                dated,
                "first_tool",
                tool,
                "search_dir",
                strata=DAY,
                ai=OLDER,
                gap_pair=True,
            )
        )
    return specs


# ---------------------------------------------------------------- run


def run(cfg: Config, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run every analysis and write the results files."""
    reps = int(cfg.raw["analysis"]["bootstrap_reps"])
    perm_reps = int(cfg.raw["analysis"]["permutation_reps"])
    margin = float(cfg.raw["analysis"]["equivalence_margin"])
    rng = np.random.default_rng(cfg.seed)
    log("loading data")
    df = load(cfg)
    specs = build_specs(df)
    rows: list[dict[str, Any]] = []
    log(f"estimating {len(specs)} comparisons")
    for s in specs:
        e = fair_estimate(
            s.data, s.strata, s.arm_col, s.a, s.b, s.y, rng, reps, perm_reps if s.perm else 0
        )
        rows.append(_row(s, "fair", e))
        if s.primary or s.hid in ("P2-sub",):
            rows.append(
                _row(s, "simple", simple_estimate(s.data, s.arm_col, s.a, s.b, s.y, rng, reps))
            )
        if s.primary:
            c = fe_lpm(s.data, s.strata, s.arm_col, s.a, s.b, s.y)
            if c is not None:
                rows.append(
                    _row(
                        s,
                        "fe_lpm",
                        None,
                        estimate=c.estimate,
                        ci_lo=c.ci_lo,
                        ci_hi=c.ci_hi,
                        p_value=c.p_value,
                    )
                )

    older = df[(df.dataset == OLDER) & df.dated]
    nv = df[df.dataset == NVIDIA]
    # S8: P1 by issue type (interactions)
    log("interactions")
    feats = ["has_code_block", "has_stack_trace", "names_file", "long_issue"]
    s8 = Spec(
        "S8",
        "RQ5",
        "Reproduce first x issue type",
        older,
        "first_move_arm",
        "R",
        "S",
        strata=DAY,
        ai=OLDER,
    )
    for c in fe_interactions(
        older.dropna(subset=feats), DAY, "first_move_arm", "R", "S", "y", feats
    ):
        if c.name != "arm":
            rows.append(
                _row(
                    s8,
                    "fe_interaction",
                    None,
                    term=c.name,
                    estimate=c.estimate,
                    ci_lo=c.ci_lo,
                    ci_hi=c.ci_hi,
                    p_value=c.p_value,
                )
            )
    # S9: P2 by language (interactions, reference = python)
    langs = [
        str(x) for x, g in nv.groupby("language") if g.issue_key.nunique() >= 300 and x != "python"
    ]
    nvl = nv.copy()
    for x in langs:
        nvl[f"lang_{x}"] = nvl["language"] == x
    s9 = Spec(
        "S9",
        "RQ5",
        "Test first x language (vs python)",
        nvl,
        "order_arm",
        "test first",
        "fix first",
        ai="NVIDIA pooled",
    )
    for c in fe_interactions(
        nvl, [], "order_arm", "test first", "fix first", "y", [f"lang_{x}" for x in langs]
    ):
        if c.name != "arm":
            rows.append(
                _row(
                    s9,
                    "fe_interaction",
                    None,
                    term=c.name,
                    estimate=c.estimate,
                    ci_lo=c.ci_lo,
                    ci_hi=c.ci_hi,
                    p_value=c.p_value,
                )
            )
    # S11: time to the first gold file (older AI, R vs S)
    log("survival")
    gold = older[older.gold_known]
    s11 = Spec(
        "S11",
        "RQ6",
        "Finding the right file: per-step hazard",
        gold,
        "first_move_arm",
        "R",
        "S",
        ai=OLDER,
    )
    hz = sv.hazard_effect(gold, DAY, "first_move_arm", "R", "S", "gold_time", "gold_event")
    if hz is not None:
        rows.append(
            _row(
                s11,
                "hazard_lpm",
                None,
                estimate=hz.estimate,
                ci_lo=hz.ci_lo,
                ci_hi=hz.ci_hi,
                p_value=hz.p_value,
            )
        )
    km = {}
    for arm in ("R", "S"):
        g = gold[gold.first_move_arm == arm]
        curve = sv.kaplan_meier(g.gold_time.to_numpy(), g.gold_event.to_numpy(), 40)
        km[arm] = {
            "median_step": sv.median_time(curve),
            "curve": curve.round(4).tolist(),
            "tries": int(len(g)),
        }
    # P2 secondary: exact forks
    log("forks")
    pairs = load_forks(cfg, df)
    s_fork = Spec(
        "P2-fork",
        "RQ2",
        "Test file now vs anything else now (exact forks)",
        nv,
        "order_arm",
        "repro_create",
        "other",
        ai="NVIDIA pooled",
    )
    rows.append(_row(s_fork, "fork", fk.fork_effect(pairs, "repro_create", rng, reps=reps)))
    s_fork2 = Spec(
        "P2-fork",
        "RQ2",
        "Test file now vs edit code now (exact forks)",
        nv,
        "order_arm",
        "repro_create",
        "edit_code",
        ai="NVIDIA pooled",
    )
    rows.append(
        _row(s_fork2, "fork", fk.fork_effect(pairs, "repro_create", rng, ["edit_code"], reps))
    )

    res = pd.DataFrame(rows)
    # Benjamini-Hochberg over all secondary tests (one p-value per estimate)
    sec = (
        ~res["primary"]
        & ~res["hid"].isin(["P2-sub", "P2-fork"])
        & res["design"].isin(["fair", "fe_interaction", "hazard_lpm"])
    )
    res["q_value"] = np.nan
    res.loc[sec.to_numpy(), "q_value"] = benjamini_hochberg(
        res.loc[sec.to_numpy(), "p_value"].to_numpy(float)
    )

    # P3: size of the simple-analysis trap
    log("simple vs fair gaps (P3)")
    gaps = []
    gap_draws_all = []
    for s in [s for s in specs if s.gap_pair]:
        simple, fair, sd, fd = gap_draws(s, rng, reps)
        sig_s = not (np.nanpercentile(sd, 2.5) <= 0 <= np.nanpercentile(sd, 97.5))
        sig_f = not (np.nanpercentile(fd, 2.5) <= 0 <= np.nanpercentile(fd, 97.5))
        gaps.append(
            {
                "hid": s.hid,
                "label": s.label,
                "ai": s.ai,
                "simple": simple,
                "fair": fair,
                "gap": simple - fair,
                "sign_differs": bool(np.sign(simple) != np.sign(fair)),
                "significance_differs": bool(sig_s != sig_f),
            }
        )
        gap_draws_all.append(np.abs(sd - fd))
    gap_df = pd.DataFrame(gaps)
    mean_abs = np.nanmean(np.vstack(gap_draws_all), axis=0)
    p3 = {
        "pairs": int(len(gap_df)),
        "mean_abs_gap": float(gap_df.gap.abs().mean()),
        "ci": [float(x) for x in np.nanpercentile(mean_abs, [2.5, 97.5])],
        "share_sign_differs": float(gap_df.sign_differs.mean()),
        "share_significance_differs": float(gap_df.significance_differs.mean()),
        "supported": bool(np.nanpercentile(mean_abs, 2.5) > margin),
    }
    simpson = {
        "pooled_simple_nvidia": next(
            r["estimate"] for r in rows if r["hid"] == "P2" and r["design"] == "simple"
        ),
        "per_ai_simple": {
            r["ai"]: r["estimate"] for r in rows if r["hid"] == "P2-sub" and r["design"] == "simple"
        },
    }

    # Design checks (RQ1)
    log("design checks")
    pos = res[res.design == "fair"][["hid", "label", "ai", "n_strata"]].copy()
    pos["underpowered"] = pos["n_strata"] < 100
    balance = hour_balance(df, rng, perm_reps)
    plc = fair_estimate(older, DAY, "placebo", "reproduce.py", "reproduce_bug.py", "y", rng, reps)
    compliance = (
        df.assign(test_first=(df.order_arm == "test first").astype(float))[
            df.order_arm != "no edit"
        ]
        .groupby("subset")
        .test_first.mean()
        .round(4)
        .to_dict()
    )
    undecided = {k: bool(0.05 <= v <= 0.95) for k, v in compliance.items()}

    p1 = res[(res.hid == "P1") & (res.design == "fair")].iloc[0]
    p2 = res[(res.hid == "P2") & (res.design == "fair")].iloc[0]
    decisions = {
        "P1": {
            "estimate": p1.estimate,
            "ci": [p1.ci_lo, p1.ci_hi],
            "p_value": p1.p_value,
            "supported": bool(p1.ci_hi < 0 or p1.ci_lo > 0),
            "direction": "harm" if p1.ci_hi < 0 else ("benefit" if p1.ci_lo > 0 else "none"),
        },
        "P2": {
            "estimate": p2.estimate,
            "ci": [p2.ci_lo, p2.ci_hi],
            "ci90": [p2.ci90_lo, p2.ci90_hi],
            "p_value": p2.p_value,
            "verdict": (
                "difference"
                if (p2.ci_hi < 0 or p2.ci_lo > 0)
                else (
                    "equivalent within ±2 points"
                    if (p2.ci90_lo > -margin and p2.ci90_hi < margin)
                    else "inconclusive"
                )
            ),
        },
        "P3": p3,
    }
    out = cfg.path("results")
    out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out / "estimates.parquet", index=False)
    gap_df.to_parquet(out / "gaps.parquet", index=False)
    summary = {
        "decisions": decisions,
        "simpson": simpson,
        "design_checks": {
            "positivity": pos.to_dict("records"),
            "hour_balance": balance,
            "placebo": plc.as_dict(),
            "tip_share_test_first": compliance,
            "undecided": undecided,
        },
        "kaplan_meier": km,
        "n_analysis_tries": int(len(df)),
        "n_fork_pairs": int(len(pairs)),
    }
    (out / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    log(f"wrote {out / 'estimates.parquet'} ({len(res)} rows)")
    return summary


def results_path(cfg: Config) -> Path:
    """Where the estimates table lives."""
    return cfg.path("results") / "estimates.parquet"
