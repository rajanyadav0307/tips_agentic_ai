r"""Step 8: paper tables T1-T4 and the number macros used in the paper text.

Nothing here is typed by hand: every cell and every macro comes from the
results files. Outputs go to ``outputs/tables/``:

- ``t1_datasets``, ``t2_funnel``, ``t3_primary``, ``t4_secondary`` as ``.tex``
  (booktabs) and ``.md``;
- ``numbers.tex``: one ``\newcommand`` per number quoted in the paper, so the
  text can never drift from the results.

The Markdown versions are also collected in ``docs/08_tables.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from agent_tips.config import Config
from agent_tips.viz.figures import TIP_NAMES, short_ai

DATASET_NAMES = {
    "older_ai": "Older AI (SWE-agent 2024, Llama-70B)",
    "newer_ai_nebius": "Nebius OpenHands (Qwen3-Coder)",
    "newer_ai_nvidia": "NVIDIA Open-SWE-Traces (4 AIs, 7 configurations)",
}
COUNT_OUTCOMES = {"n_steps", "chars_in_k", "chars_out_k"}


def _tex_escape(x: str) -> str:
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("±", r"$\pm$"),
        ("×", r"$\times$"),
        ("−", "$-$"),
    ):
        x = x.replace(a, b)
    return x


def to_tex(df: pd.DataFrame, align: str | None = None) -> str:
    """Booktabs LaTeX tabular for a table of strings."""
    align = align or "l" + "r" * (len(df.columns) - 1)
    head = " & ".join(_tex_escape(str(c)) for c in df.columns) + r" \\"
    body = [" & ".join(_tex_escape(str(v)) for v in row) + r" \\" for row in df.to_numpy()]
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{align}}}",
            r"\toprule",
            head,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def to_md(df: pd.DataFrame) -> str:
    """GitHub-flavoured Markdown table."""
    rows = [list(map(str, df.columns)), ["---"] * len(df.columns)]
    rows += [[str(v).replace("|", "\\|") for v in r] for r in df.to_numpy()]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"


def _ci(r: Any, scale: float = 100.0, digits: int = 1) -> str:
    if pd.isna(r["estimate"]):
        return "—"
    f = f"{{:+.{digits}f}}"
    return (
        f"{f.format(r['estimate'] * scale)} "
        f"[{f.format(r['ci_lo'] * scale)}, {f.format(r['ci_hi'] * scale)}]"
    )


def _p(x: Any) -> str:
    if x is None or pd.isna(x):
        return "—"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


# ---------------------------------------------------------------- tables


def t1_datasets(cfg: Config, df: pd.DataFrame, steps: dict[str, int]) -> pd.DataFrame:
    """T1: the three trajectory datasets as analysed."""
    rows = []
    for key, name in DATASET_NAMES.items():
        d = df[df.dataset == key]
        per_issue = d.groupby("issue_key").size()
        rows.append(
            {
                "Dataset": name,
                "Source (revision)": f"{cfg.raw['datasets'][key]['hf_id']} "
                f"({cfg.raw['datasets'][key]['revision'][:7]})",
                "Tries": f"{len(d):,}",
                "Issues": f"{d.issue_key.nunique():,}",
                "Tries per issue (median)": f"{per_issue.median():.0f}",
                "Steps": f"{steps[key]:,}",
                "Bugs fixed": f"{d.y.mean():.1%}",
            }
        )
    return pd.DataFrame(rows)


def t2_funnel(funnel: dict[str, Any]) -> pd.DataFrame:
    """T2: the cleaning funnel."""
    rows = []
    for r in funnel["funnel"]:
        rows.append(
            {
                "Dataset": DATASET_NAMES[r["dataset"]].split(" (")[0],
                "Stage": r["stage"],
                "Tries": f"{r['tries']:,}",
                "Issues": f"{r['issues']:,}",
                "Removed": "—" if r["removed"] is None else f"{r['removed']:,}",
                "Bugs fixed": f"{r['resolved_rate']:.1%}",
            }
        )
    return pd.DataFrame(rows)


def t3_primary(est: pd.DataFrame, summary: dict[str, Any]) -> pd.DataFrame:
    """T3: primary estimates under every design."""
    rows = []
    for hid, name in (
        ("P1", "P1 Reproduce first (older AI)"),
        ("P2", "P2 Test first (newer AIs, pooled)"),
    ):
        e = est[est.hid == hid].set_index("design")
        d = summary["decisions"][hid]
        rows.append(
            {
                "Hypothesis": name,
                "Fair (95% CI)": _ci(e.loc["fair"]),
                "Permutation p": _p(e.loc["fair", "p_value"]),
                "FE regression": _ci(e.loc["fe_lpm"]),
                "Simple": _ci(e.loc["simple"]),
                "Groups / issues / tries": " / ".join(
                    f"{int(float(str(v))):,}"
                    for v in e.loc["fair", ["n_strata", "n_issues", "n_tries"]]
                ),
                "Decision": d.get("direction", d.get("verdict", "")),
            }
        )
    p3 = summary["decisions"]["P3"]
    rows.append(
        {
            "Hypothesis": "P3 Mean |simple − fair| over tip × AI pairs",
            "Fair (95% CI)": f"{p3['mean_abs_gap'] * 100:.1f} [{p3['ci'][0] * 100:.1f}, "
            f"{p3['ci'][1] * 100:.1f}]",
            "Permutation p": "—",
            "FE regression": "—",
            "Simple": "—",
            "Groups / issues / tries": f"{p3['pairs']} pairs",
            "Decision": "supported (> 2 pts)" if p3["supported"] else "not supported",
        }
    )
    return pd.DataFrame(rows)


def t4_secondary(est: pd.DataFrame, q: float) -> pd.DataFrame:
    """T4: every secondary estimate with its BH q-value."""
    d = est[est.q_value.notna()].copy()
    rows = []
    for _, r in d.iterrows():
        scale = 1.0 if r.outcome in COUNT_OUTCOMES else 100.0
        label = TIP_NAMES.get(r.hid, r.label) if r.hid in TIP_NAMES else r.label
        if isinstance(r.term, str):
            label = f"{r.label}: {r.term.replace('arm x ', '')}"
        rows.append(
            {
                "ID": r.hid,
                "Question": label,
                "AI": short_ai(r.ai),
                "Estimate (95% CI)": _ci(r, scale) + ("" if scale == 1 else " pts"),
                "Groups": "—" if pd.isna(r.n_strata) else f"{int(r.n_strata):,}",
                "q": _p(r.q_value),
                "Sig.": "yes"
                if r.q_value < q and not (r.n_strata < 100)
                else ("underpowered" if r.n_strata < 100 else ""),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- numbers


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{_tex_escape(value)}}}"


def numbers(
    est: pd.DataFrame,
    summary: dict[str, Any],
    rob: dict[str, Any],
    lab: dict[str, Any],
    df: pd.DataFrame,
    funnel: dict[str, Any],
    cfg: Config,
) -> list[str]:
    """Every number quoted in the paper, as LaTeX macros (points unless stated)."""

    def pts(x: Any) -> str:
        v = round(float(x) * 100, 1)
        return "0.0" if v == 0 else f"{v:+.1f}".replace("-", "−")

    def abs1(x: Any) -> str:
        return f"{abs(float(x)) * 100:.1f}"

    q = float(cfg.raw["analysis"]["fdr_q"])

    def row(
        hid: str, design: str = "fair", ai: str | None = None, outcome: str | None = None
    ) -> pd.Series:
        m = (est.hid == hid) & (est.design == design)
        if ai:
            m &= est.ai == ai
        if outcome:
            m &= est.outcome == outcome
        return est[m].iloc[0]

    out: dict[str, str] = {}
    p1, p1s, p1fe = row("P1"), row("P1", "simple"), row("P1", "fe_lpm")
    out |= {
        "POneEst": pts(p1.estimate),
        "POneAbs": abs1(p1.estimate),
        "POneLo": pts(p1.ci_lo),
        "POneHi": pts(p1.ci_hi),
        "POneP": _p(p1.p_value),
        "POneGroups": f"{int(p1.n_strata):,}",
        "POneIssues": f"{int(p1.n_issues):,}",
        "POneTries": f"{int(p1.n_tries):,}",
        "POneSimple": pts(p1s.estimate),
        "POneFE": pts(p1fe.estimate),
    }
    p2, p2s, p2fe = row("P2"), row("P2", "simple"), row("P2", "fe_lpm")
    out |= {
        "PTwoEst": pts(p2.estimate),
        "PTwoLo": pts(p2.ci_lo),
        "PTwoHi": pts(p2.ci_hi),
        "PTwoLoNinety": pts(p2.ci90_lo),
        "PTwoHiNinety": pts(p2.ci90_hi),
        "PTwoP": _p(p2.p_value),
        "PTwoIssues": f"{int(p2.n_issues):,}",
        "PTwoTries": f"{int(p2.n_tries):,}",
        "PTwoSimple": pts(p2s.estimate),
        "PTwoFE": pts(p2fe.estimate),
        "Margin": f"{float(cfg.raw['analysis']['equivalence_margin']) * 100:.0f}",
    }
    sub = est[est.hid == "P2-sub"]
    sf, ss = sub[sub.design == "fair"].estimate, sub[sub.design == "simple"].estimate
    out |= {
        "PerAIFairMin": pts(sf.min()),
        "PerAIFairMax": pts(sf.max()),
        "PerAISimpleMin": pts(ss.min()),
        "PerAISimpleMax": pts(ss.max()),
        "NumNewerGroups": str(len(sf)),
    }
    p3 = summary["decisions"]["P3"]
    ex = p3["excluding_underpowered"]
    out |= {
        "PThreePairs": str(p3["pairs"]),
        "PThreeGap": f"{p3['mean_abs_gap'] * 100:.1f}",
        "PThreeLo": f"{p3['ci'][0] * 100:.1f}",
        "PThreeHi": f"{p3['ci'][1] * 100:.1f}",
        "PThreeSign": f"{p3['share_sign_differs']:.0%}",
        "PThreeSig": f"{p3['share_significance_differs']:.0%}",
        "PThreeExPairs": str(ex["pairs"]),
        "PThreeExGap": f"{ex['mean_abs_gap'] * 100:.1f}",
        "PThreeExLo": f"{ex['ci'][0] * 100:.1f}",
        "PThreeExHi": f"{ex['ci'][1] * 100:.1f}",
        "PThreeExSign": f"{ex['share_sign_differs']:.0%}",
        "PThreeExSig": f"{ex['share_significance_differs']:.0%}",
    }
    s1 = row("S1", ai="older_ai")
    s1n = est[(est.hid == "S1") & (est.design == "fair") & (est.ai != "older_ai")]
    s1q = s1n.sort_values("estimate").iloc[-1]  # the newer AI with the largest association
    out |= {
        "SOneOlder": pts(s1.estimate),
        "SOneOlderLo": pts(s1.ci_lo),
        "SOneOlderHi": pts(s1.ci_hi),
        "SOneTopNewer": pts(s1q.estimate),
        "SOneTopNewerAI": short_ai(str(s1q.ai)).replace(" / ", " with ", 1).split(" / ")[0],
    }
    sec = est[est.q_value.notna()]
    out |= {
        "NumSecondary": str(len(sec)),
        "NumSecondarySig": str(int((sec.q_value < q).sum())),
        "NumUnderpowered": str(int((est[est.design == "fair"].n_strata < 100).sum())),
    }
    out |= {
        "CostStepsOlder": f"{row('S7', ai='older_ai', outcome='n_steps').estimate:.1f}",
        "CostStepsNewer": f"{row('S7', ai='NVIDIA pooled', outcome='n_steps').estimate:.1f}",
        "CostReadNewer": f"{row('S7', outcome='chars_in_k').estimate:.1f}",
        "CostWriteNewer": f"{row('S7', outcome='chars_out_k').estimate:.1f}",
    }
    s10 = row("S10", outcome="ctx_any")
    km = summary["kaplan_meier"]
    s11 = row("S11", "hazard_lpm")
    out |= {
        "CtxExit": pts(s10.estimate),
        "KMMedianR": f"{km['R']['median_step']:.0f}",
        "KMMedianS": f"{km['S']['median_step']:.0f}",
        "Hazard": pts(s11.estimate),
    }
    dc = summary["design_checks"]
    plc, hb = dc["placebo"], dc["hour_balance"]
    out |= {
        "Placebo": pts(plc["estimate"]),
        "PlaceboLo": pts(plc["ci_lo"]),
        "PlaceboHi": pts(plc["ci_hi"]),
        "HourBalanceP": f"{hb['p_value']:.2f}",
        "HourBalanceGroups": f"{hb['groups']:,}",
    }
    fk = row("P2-fork", "fork")
    out |= {
        "ForkPairs": f"{summary['n_fork_pairs']:,}",
        "ForkEst": pts(fk.estimate),
        "ForkLo": pts(fk.ci_lo),
        "ForkHi": pts(fk.ci_hi),
        "ForkN": f"{int(fk.n_tries):,}",
    }
    b = [e["estimate"] for e in rob["p1_batches"].values()]
    loo = rob["p2_leave_one_out"].values()
    und = rob["p1_undated"]["day"]
    tip, cal, cl = rob["p2_tipping_point"], rob["p2_removal_calibration"], rob["conditional_logit"]
    out |= {
        "BatchMin": pts(max(b)),
        "BatchMax": pts(min(b)),
        "LOOLo": pts(min(e["ci90_lo"] for e in loo)),
        "LOOHi": pts(max(e["ci90_hi"] for e in loo)),
        "Broad": pts(rob["p2_broad_definition"]["estimate"]),
        "BroadLo": pts(rob["p2_broad_definition"]["ci_lo"]),
        "BroadHi": pts(rob["p2_broad_definition"]["ci_hi"]),
        "UndatedImputed": pts(und["added back like other failures (median of draws)"]),
        "UndatedWorst": pts(und["added back, worst case for the finding"]),
        "TipPoint": f"{tip['extra_failure_share']:.1%}",
        "Calib": pts(cal["estimate"]),
        "CalibLo": pts(cal["ci_lo"]),
        "CalibHi": pts(cal["ci_hi"]),
        "CalibIssues": f"{cal['n_issues']:,}",
        "ORPOne": f"{cl['P1']['odds_ratio']:.2f}",
        "ORPTwo": f"{cl['P2']['odds_ratio']:.2f}",
        "Reproducible": "identical"
        if rob.get("reproducibility", {}).get("identical")
        else "not identical",
    }
    s12 = row("S12")
    s13 = est[(est.hid == "S13") & (est.design == "fair")]
    share = {k: v for k, v in dc["tip_share_test_first"].items() if "/" in k}
    out |= {
        "STwelve": pts(s12.estimate),
        "STwelveLo": pts(s12.ci_lo),
        "STwelveHi": pts(s12.ci_hi),
        "SThirteenMin": pts(s13.estimate.min()),
        "SThirteenMax": pts(s13.estimate.max()),
        "ComplianceMin": f"{min(share.values()):.0%}",
        "ComplianceMax": f"{max(share.values()):.0%}",
        "NebiusCompliance": f"{dc['tip_share_test_first']['newer_ai_nebius']:.0%}",
    }
    eda = json.loads((cfg.path("results") / "eda_stats.json").read_text(encoding="utf-8"))
    older = df[df.dataset == "older_ai"]
    nv_models = {k.split("/")[1] for k in share}
    out |= {
        "VersionRateMin": f"{eda['e3_older_ai_by_day']['rate_min']:.0%}",
        "VersionRateMax": f"{eda['e3_older_ai_by_day']['rate_max']:.0%}",
        "OlderTriesPerIssue": f"{older.groupby('issue_key').size().median():.0f}",
        "NumNewerModels": str(len(nv_models)),
        "NumNewerConfigs": str(len(share)),
    }
    sl = lab.get("second_labeling") or {}
    if "round_1" in sl:
        r2 = sl.get("round_2", sl["round_1"])
        rc = r2["by_label"]["repro_create"]
        out |= {
            "KappaOne": f"{sl['round_1']['kappa']:.2f}",
            "KappaTwo": f"{r2['kappa']:.2f}",
            "LabelNOne": str(sl["round_1"]["n"]),
            "LabelNTwo": str(r2["n"]),
            "ReproPrecision": f"{rc['precision']:.0%}",
            "ReproRecall": f"{rc['recall']:.0%}",
        }
    else:  # the second labeling has not been done yet
        keys = ("KappaOne", "KappaTwo", "LabelNOne", "LabelNTwo", "ReproPrecision", "ReproRecall")
        out |= dict.fromkeys(keys, "n/a")
    out |= {
        "TriesAnalysed": f"{len(df):,}",
        "IssuesAnalysed": f"{df.issue_key.nunique():,}",
        "TriesTotal": f"{sum(r['tries'] for r in funnel['funnel'] if r['condition'] == 'TRUE'):,}",
        "StepsTotal": f"{sum(funnel['steps'].values()) / 1e6:.1f}",
        "OlderShareR": f"{summary['design_checks']['tip_share_test_first']['older_ai']:.0%}",
    }
    return [_macro(k, v) for k, v in out.items()]


def run(cfg: Config, log: Callable[[str], None] = print) -> list[Path]:
    """Write every table (LaTeX + Markdown) and ``numbers.tex``."""
    from agent_tips.models import analysis

    res = cfg.path("results")
    est = pd.read_parquet(res / "estimates.parquet")

    def read(name: str) -> Any:
        return json.loads((res / name).read_text(encoding="utf-8"))

    summary, rob = read("analysis_summary.json"), read("robustness.json")
    funnel, lab = read("funnel.json"), read("label_validation.json")
    df = analysis.load(cfg)
    q = float(cfg.raw["analysis"]["fdr_q"])
    out_dir = cfg.root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: Sequence[tuple[str, str, pd.DataFrame, str | None]] = [
        ("t1_datasets", "T1. Datasets", t1_datasets(cfg, df, funnel["steps"]), "llrrrrr"),
        ("t2_funnel", "T2. Cleaning funnel", t2_funnel(funnel), "llrrrr"),
        (
            "t3_primary",
            "T3. Primary estimates (points per 100 tries)",
            t3_primary(est, summary),
            "lllllll",
        ),
        (
            "t4_secondary",
            "T4. Secondary estimates with Benjamini–Hochberg q-values",
            t4_secondary(est, q),
            "llllrrl",
        ),
    ]
    paths = []
    md = [
        "# Step 8: Paper tables",
        "",
        "*Generated by `agent-tips figures` from `outputs/results/`. Never edited by hand.*",
        "",
    ]
    for name, title, t, align in tables:
        (out_dir / f"{name}.tex").write_text(to_tex(t, align), encoding="utf-8")
        (out_dir / f"{name}.md").write_text(to_md(t), encoding="utf-8")
        md += [f"## {title}", "", to_md(t)]
        paths.append(out_dir / f"{name}.tex")
    nums = numbers(est, summary, rob, lab, df, funnel, cfg)
    (out_dir / "numbers.tex").write_text(
        "% Generated by agent-tips figures. Do not edit.\n" + "\n".join(nums) + "\n",
        encoding="utf-8",
    )
    paths.append(out_dir / "numbers.tex")
    (cfg.root / "docs").mkdir(parents=True, exist_ok=True)
    (cfg.root / "docs" / "08_tables.md").write_text("\n".join(md), encoding="utf-8")
    for p in paths:
        log(f"wrote {p}")
    return paths
