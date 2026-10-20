"""Step 6: write ``docs/06_results.md`` from the estimates table.

Every number comes from ``outputs/results/estimates.parquet``,
``gaps.parquet`` and ``analysis_summary.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agent_tips.config import Config  # noqa: E402
from agent_tips.eda.charts import COLORS, subset_label  # noqa: E402

# Exploratory numbers (2026-10-19) for task D4: (hid, ai, label part) -> estimate in points.
EXPLORATION: list[tuple[str, str, str, float]] = [
    ("P1", "older_ai", "Reproduce first", -3.49),
    ("P2", "NVIDIA pooled", "Test first (newer AIs, pooled)", 0.3),
    ("S1", "older_ai", "Check your fix", 3.0),
    ("S2", "older_ai", "test suite after fixing", 0.3),
    ("S7", "older_ai", "Cost: steps", 10.0),
    ("S7", "NVIDIA pooled", "Cost: steps", 1.8),
    ("S10", "older_ai", "Submitted because the context ran out", 11.0),
    ("S12", "newer_ai_nebius", "think vs view", -0.06),
    ("S13", "older_ai", "find_file", -0.5),
    ("S13", "older_ai", "ls", -1.2),
    ("S13", "older_ai", "open", 5.0),
    ("S13", "older_ai", "grep", -1.2),
]
COUNT_OUTCOMES = {"n_steps", "chars_in_k", "chars_out_k"}


def _ai(name: str) -> str:
    if name in ("NVIDIA pooled", "all"):
        return name
    return subset_label(name)


def _fmt(r: Any, scale: float = 100.0) -> str:
    if pd.isna(r["estimate"]):
        return "—"
    unit = "" if scale == 1 else " pts"
    return (
        f"{r['estimate'] * scale:+.1f}{unit} [{r['ci_lo'] * scale:+.1f}, {r['ci_hi'] * scale:+.1f}]"
    )


def _scale(outcome: str) -> float:
    return 1.0 if outcome in COUNT_OUTCOMES else 100.0


def _p(x: Any) -> str:
    if x is None or pd.isna(x):
        return "—"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def km_chart(km: dict[str, Any], root: Path) -> str:
    """Kaplan-Meier curves: share of tries that have not yet touched a gold file."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (arm, lab) in enumerate([("R", "Starts by reproducing"), ("S", "Starts by searching")]):
        if arm in km:
            curve = np.array(km[arm]["curve"])
            ax.step(np.arange(len(curve)), curve, where="post", color=COLORS[i], label=lab)
    ax.axhline(0.5, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Share of tries that have not yet\ntouched a file changed by the real fix")
    ax.set_title("S11. Older AI: how fast it finds the right file")
    ax.legend(fontsize=8, frameon=False)
    out = root / "docs" / "figures" / "results"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "s11_km.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "figures/results/s11_km.png"


def build(cfg: Config, out: Path | None = None) -> Path:
    """Write the results report."""
    res = cfg.path("results")
    est = pd.read_parquet(res / "estimates.parquet")
    gaps = pd.read_parquet(res / "gaps.parquet")
    s = json.loads((res / "analysis_summary.json").read_text(encoding="utf-8"))
    d, dc = s["decisions"], s["design_checks"]
    km_png = km_chart(s["kaplan_meier"], cfg.root)
    fair = est[est.design == "fair"]

    def one(hid: str, design: str = "fair", **kw: str) -> pd.Series:
        q = est[(est.hid == hid) & (est.design == design)]
        for k, v in kw.items():
            q = q[q[k].astype(str).str.contains(v, regex=False)]
        return q.iloc[0]

    p1, p1s, p1l = one("P1"), one("P1", "simple"), one("P1", "fe_lpm")
    p2, p2s, p2l = one("P2"), one("P2", "simple"), one("P2", "fe_lpm")
    p3 = d["P3"]
    sec = est[est.q_value.notna()]
    sig = sec[sec.q_value < 0.05]

    # ---- plain-words summary, written from the frozen decision rules ----
    p1_txt = {
        "harm": f"**Reproducing first hurt the older AI:** {p1.estimate * 100:+.1f} bugs fixed per 100 tries "
        f"(95% CI {p1.ci_lo * 100:+.1f} to {p1.ci_hi * 100:+.1f}). This is a true coin-flip comparison.",
        "benefit": f"**Reproducing first helped the older AI:** {p1.estimate * 100:+.1f} per 100 tries.",
        "none": "**No clear effect of reproducing first for the older AI** (the CI includes zero).",
    }[d["P1"]["direction"]]
    p2_txt = {
        "equivalent within ±2 points": f"**For the newer AIs, testing first made no meaningful difference:** "
        f"{p2.estimate * 100:+.1f} per 100 tries, and the 90% CI ({p2.ci90_lo * 100:+.1f} to "
        f"{p2.ci90_hi * 100:+.1f}) lies within the ±2-point margin set in advance.",
        "difference": f"**For the newer AIs, testing first changed success:** {p2.estimate * 100:+.1f} per 100 tries "
        f"(95% CI {p2.ci_lo * 100:+.1f} to {p2.ci_hi * 100:+.1f}).",
        "inconclusive": f"**For the newer AIs the result is inconclusive:** {p2.estimate * 100:+.1f} per 100 tries; "
        "the CI is too wide to call it either way.",
    }[d["P2"]["verdict"]]
    p3_txt = (
        f"**Simple analyses are badly misleading:** across {p3['pairs']} tip × AI comparisons, "
        f"simple and fair estimates differ by {p3['mean_abs_gap'] * 100:.1f} points on average "
        f"(95% CI {p3['ci'][0] * 100:.1f} to {p3['ci'][1] * 100:.1f}). In "
        f"{p3['share_sign_differs']:.0%} of comparisons they point in opposite directions."
        if p3["supported"]
        else f"Simple and fair estimates differ by {p3['mean_abs_gap'] * 100:.1f} points on average, "
        "which is not clearly above the 2-point threshold set in advance."
    )

    lines: list[str] = []
    add = lines.append
    add("# Step 6: Results\n")
    add(
        "*Generated by `agent-tips model` from `outputs/results/`. Analyses follow the frozen plan "
        "`docs/01_problem_and_hypotheses.md` (tag `analysis-plan-v1`).*\n"
    )
    add("## In plain words\n")
    add(f"- {p1_txt}")
    add(f"- {p2_txt}")
    add(f"- {p3_txt}")
    add(
        f"- **Secondary questions:** {len(sig)} of {len(sec)} secondary estimates stay significant after "
        "correcting for testing many things at once (Benjamini–Hochberg, q < 0.05). See the tables below."
    )
    km = s["kaplan_meier"]
    add(
        f"- **Why reproducing first hurt:** it uses up the older AI's memory and slows it down. These tries ran "
        f"out of context more often, and reached the file the real fix changed later (median step "
        f"{km['R']['median_step']:.0f} vs {km['S']['median_step']:.0f})."
    )
    add(
        f"- **Design checks:**\n  - placebo {_fmt(dc['placebo'])} (should be about 0);\n"
        f"  - the hour does not predict the first move within a day (p = {dc['hour_balance']['p_value']:.2f}).\n"
    )

    add("## Main results\n")
    add(
        "| Hypothesis | Fair comparison (main) | Simple comparison | Fixed-effects regression | Decision |"
    )
    add("| --- | --- | --- | --- | --- |")
    add(
        f"| **P1** reproduce first, older AI (issue × day; permutation p {_p(p1.p_value)}) | {_fmt(p1)} | {_fmt(p1s)} | "
        f"{_fmt(p1l)} | {d['P1']['direction']} |"
    )
    add(
        f"| **P2** test first, newer AIs pooled (issue; permutation p {_p(p2.p_value)}) | {_fmt(p2)} (90% CI "
        f"{p2.ci90_lo * 100:+.1f}, {p2.ci90_hi * 100:+.1f}) | {_fmt(p2s)} | {_fmt(p2l)} | {d['P2']['verdict']} |"
    )
    add(
        f"| **P3** mean absolute gap between simple and fair | {p3['mean_abs_gap'] * 100:.1f} pts "
        f"[{p3['ci'][0] * 100:.1f}, {p3['ci'][1] * 100:.1f}] over {p3['pairs']} pairs | — | — | "
        f"{'supported (> 2 pts)' if p3['supported'] else 'not supported'} |\n"
    )
    add(
        f"Fair estimates use {int(p1.n_strata):,} issue × day groups ({int(p1.n_issues):,} issues, "
        f"{int(p1.n_tries):,} tries) for P1 and {int(p2.n_strata):,} issues ({int(p2.n_tries):,} tries) for P2.\n"
    )

    add("### P2 per AI\n")
    add("| AI | Fair (same issue) | Simple (pooled) | Issues used |")
    add("| --- | --- | --- | ---: |")
    for _, r in fair[fair.hid == "P2-sub"].iterrows():
        sm = est[(est.hid == "P2-sub") & (est.design == "simple") & (est.ai == r.ai)].iloc[0]
        add(f"| {_ai(r.ai)} | {_fmt(r)} | {_fmt(sm)} | {int(r.n_issues):,} |")
    add("")
    add("### Near coin flips (exact forks, newer AIs)\n")
    add("| Comparison | Estimate | Fork pairs | Issues |")
    add("| --- | --- | ---: | ---: |")
    for _, r in est[est.design == "fork"].iterrows():
        add(f"| {r.label} | {_fmt(r)} | {int(r.n_strata):,} | {int(r.n_issues):,} |")
    add(
        f"\nTotal fork pairs (tries of the same issue with identical histories up to their first "
        f"difference): {s['n_fork_pairs']:,}.\n"
    )

    add("## P3: the simple-analysis trap\n")
    add(
        f"- **Mean absolute gap:** {p3['mean_abs_gap'] * 100:.1f} points [{p3['ci'][0] * 100:.1f}, {p3['ci'][1] * 100:.1f}]."
    )
    add(f"- **Sign differs:** {p3['share_sign_differs']:.0%} of pairs.")
    add(f"- **Significance differs:** {p3['share_significance_differs']:.0%} of pairs.")
    ex = p3.get("excluding_underpowered")
    if ex:
        add(
            f"- **Without the {p3['pairs'] - ex['pairs']} underpowered pairs** (fewer than 100 "
            f"groups, which the plan says not to interpret): mean gap "
            f"{ex['mean_abs_gap'] * 100:.1f} points [{ex['ci'][0] * 100:.1f}, "
            f"{ex['ci'][1] * 100:.1f}] over {ex['pairs']} pairs; sign differs in "
            f"{ex['share_sign_differs']:.0%}, significance in {ex['share_significance_differs']:.0%}."
        )
    add("- An estimate of exactly 0 has no direction, so it never counts as an opposite sign.")
    simp = s["simpson"]
    add(
        f"- **Simpson's paradox:** pooled across the newer AIs, the simple estimate for test first is "
        f"{simp['pooled_simple_nvidia'] * 100:+.1f} points. Within each AI it ranges from "
        f"{min(simp['per_ai_simple'].values()) * 100:+.1f} to {max(simp['per_ai_simple'].values()) * 100:+.1f}.\n"
    )
    add("| Comparison | AI | Simple | Fair | Gap |")
    add("| --- | --- | ---: | ---: | ---: |")
    for _, g in gaps.sort_values("gap", key=np.abs, ascending=False).iterrows():
        add(
            f"| {g.hid} {g.label} | {_ai(g.ai)} | {g.simple * 100:+.1f} | {g.fair * 100:+.1f} | "
            f"{g.gap * 100:+.1f} |"
        )
    add("")

    add("## Secondary hypotheses (Benjamini–Hochberg q-values)\n")
    add("| ID | Comparison | AI | Fair estimate | p | q | Issues |")
    add("| --- | --- | --- | --- | ---: | ---: | ---: |")
    for _, r in sec.sort_values(["hid", "ai"]).iterrows():
        label = r.label + (f" — {r.term}" if isinstance(r.get("term"), str) else "")
        add(
            f"| {r.hid} | {label} | {_ai(r.ai)} | {_fmt(r, _scale(r.outcome))} | {_p(r.p_value)} | "
            f"{_p(r.q_value)} | {'' if pd.isna(r.n_issues) else f'{int(r.n_issues):,}'} |"
        )
    add("\n- Steps and characters are counts or thousands, not points.")
    add(
        "- Interaction rows (S8, S9) show how the effect changes for issues with the feature (S8) or "
        "for each language relative to Python (S9)."
    )
    add("- Mid-run comparisons (S1–S6) are same-issue associations, not coin flips.\n")

    add("## Why reproducing first hurt the older AI (RQ6)\n")
    add(f"![S11]({km_png})\n")
    add(
        f"- **Median step at which a file changed by the real fix is first touched:** "
        f"{km['R']['median_step']:.0f} when starting by reproducing ({km['R']['tries']:,} tries), "
        f"vs {km['S']['median_step']:.0f} when starting by searching ({km['S']['tries']:,} tries)."
    )
    hz = est[est.design == "hazard_lpm"]
    if len(hz):
        h = hz.iloc[0]
        add(
            f"- **Per-step chance of finding it** (fixed effects for issue × day × step): {_fmt(h)}.\n"
        )

    add("## Design checks (RQ1)\n")
    hb = dc["hour_balance"]
    add(
        f"- **Hour balance (older AI):** within issue × day, the evaluation hour does not predict the first move. "
        f"Variation is {hb['ratio_to_chance']:.2f}× the chance level (p = {hb['p_value']:.2f}; {hb['groups']:,} groups)."
    )
    add(
        f"- **Placebo** (`reproduce.py` vs `reproduce_bug.py`, which should show nothing): {_fmt(dc['placebo'])}."
    )
    add(
        "- **Undecided agents** (tip followed by 5–95% of tries, so a fair comparison is possible): "
        + ", ".join(
            f"{_ai(k)} {'yes' if v else 'no'} ({dc['tip_share_test_first'][k]:.0%})"
            for k, v in dc["undecided"].items()
        )
        + "."
    )
    under = [p for p in dc["positivity"] if p["underpowered"]]
    add(
        f"- **Positivity:** {len(under)} of {len(dc['positivity'])} fair comparisons have fewer than 100 "
        "informative groups and are labelled underpowered:"
    )
    for p in under:
        add(f"  - {p['hid']} {p['label']} ({_ai(p['ai'])}, {int(p['n_strata'])} groups)")
    add("")

    add("## Final vs. exploratory numbers (task D4)\n")
    add("| Comparison | AI | Exploration | Final | Difference |")
    add("| --- | --- | ---: | ---: | ---: |")
    for hid, ai, part, v in EXPLORATION:
        q = fair[(fair.hid == hid) & (fair.ai == ai) & fair.label.str.contains(part, regex=False)]
        if q.empty:
            continue
        r = q.iloc[0]
        sc = _scale(r.outcome)
        add(
            f"| {hid} {r.label} | {_ai(ai)} | {v:+.1f} | {r.estimate * sc:+.1f} | {r.estimate * sc - v:+.1f} |"
        )
    add("\nDifferences larger than 0.5 come from three changes since the exploration:")
    add(
        "1. the fixed labeler: the `/testbed` and test-tool fixes, and broader scratch-script words;"
    )
    add("2. analysis-ready tries only: known outcome and same issue text;")
    add(
        '3. "check your fix", which is now defined from the labeler (re-running the try\'s own test or '
        "reproduction script after the first code edit), not from file names in the editor."
    )
    path = out or cfg.root / "docs" / "06_results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
