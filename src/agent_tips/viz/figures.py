"""Step 8: paper figures F1-F8.

Every figure is drawn from the results files written by Steps 6-7
(``estimates.parquet``, ``gaps.parquet``, ``analysis_summary.json``,
``robustness.json``). F2 and F8 also read the analysis-ready tries for
per-day and per-arm rates. Figures are saved as vector PDF under
``outputs/figures/paper/`` and as PNG under ``docs/figures/paper/``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from agent_tips.config import Config  # noqa: E402
from agent_tips.eda.charts import COLORS, subset_label  # noqa: E402

BLUE, ORANGE, GREEN, RED, PINK, SKY = COLORS[:6]
GREY = "#888888"
STYLE: dict[str, Any] = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
}
ONE_COL, TWO_COL = 3.4, 7.0
TIP_NAMES = {
    "S1": "Check your fix",
    "S2": "Test suite after fix",
    "S3": "Run any code after fix",
    "S4": "Extra tests after fix",
    "S5": "Clean up created files",
    "S6": "Test suite before fix",
}


def short_ai(name: str) -> str:
    """Compact AI label for axes, e.g. ``Qwen3.6-27B / SWE-agent / scale-swe``."""
    if name == "older_ai":
        return "Llama-70B (older AI)"
    if name in ("NVIDIA pooled", "newer_ai_nebius"):
        return {"NVIDIA pooled": "Newer AIs, pooled", "newer_ai_nebius": "Qwen3-Coder"}[name]
    label = subset_label(name)
    model, rest = label.split(" (", 1)
    scaffold, source = rest.rstrip(")").split(", ")
    return f"{model} / {scaffold} / {source}"


def _save(fig: Figure, name: str, cfg: Config) -> Path:
    pdf = cfg.path("figures") / "paper"
    png = cfg.root / "docs" / "figures" / "paper"
    pdf.mkdir(parents=True, exist_ok=True)
    png.mkdir(parents=True, exist_ok=True)
    # no creation date, so re-running gives byte-identical files
    fig.savefig(pdf / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(png / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf / f"{name}.pdf"


def _forest(
    ax: Axes,
    rows: pd.DataFrame,
    labels: list[str],
    color: str = BLUE,
    offset: float = 0.0,
    marker: str = "o",
    label: str | None = None,
    scale: float = 100.0,
) -> None:
    y = np.arange(len(rows))[::-1] + offset
    est = rows.estimate.to_numpy(float) * scale
    lo = rows.ci_lo.to_numpy(float) * scale
    hi = rows.ci_hi.to_numpy(float) * scale
    ax.errorbar(
        est,
        y,
        xerr=[est - lo, hi - est],
        fmt=marker,
        color=color,
        ms=4,
        capsize=2,
        lw=1,
        label=label,
    )
    ax.set_yticks(np.arange(len(rows))[::-1], labels)
    ax.axvline(0, color=GREY, lw=0.6, ls="--")


# ---------------------------------------------------------------- figures


def f1_method(cfg: Config) -> Path:
    """F1: toy example of why the simple comparison misleads and the fair one does not."""
    # Within each issue the tip changes nothing (same fix rate with and without),
    # but it is used more on the easy issue, so pooling makes it look helpful.
    issues = [("Easy issue", 8, 4, 0.75), ("Hard issue", 4, 8, 0.25)]
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.1), width_ratios=[1, 1, 1.25])
    for ax, (name, n_tip, n_no, rate) in zip(axes[:2], issues, strict=True):
        arms = ((n_tip, "with tip", ORANGE), (n_no, "without tip", BLUE))
        for row, (n, arm, color) in enumerate(arms):
            k = round(n * rate)
            for j in range(n):
                ax.add_patch(
                    Circle(
                        (j * 0.12 + 0.06, 0.72 - row * 0.45),
                        0.048,
                        color=color,
                        alpha=1.0 if j < k else 0.25,
                    )
                )
            ax.text(0.0, 0.86 - row * 0.45, f"{arm}: {k} of {n} fixed ({rate:.0%})", fontsize=7)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0.1, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name)
    tip_k = sum(round(n * r) for _, n, _, r in issues)
    tip_n = sum(n for _, n, _, _ in issues)
    no_k = sum(round(n * r) for _, _, n, r in issues)
    no_n = sum(n for _, _, n, _ in issues)
    simple = (tip_k / tip_n - no_k / no_n) * 100
    ax = axes[2]
    ax.barh([1, 0], [simple, 0.0], color=[GREY, GREEN], height=0.5)
    ax.set_yticks([1, 0], ["Simple:\npool all tries", "Fair: compare\nwithin each issue"])
    ax.axvline(0, color="black", lw=0.6)
    ax.text(simple + 1, 1, f"{simple:+.0f} pts (wrong)", va="center", fontsize=7)
    ax.text(1, 0, "+0 pts (right)", va="center", fontsize=7)
    ax.set_xlim(-2, 40)
    ax.set_xlabel("Apparent effect of the tip (points)")
    ax.set_title("Same tries, two answers")
    fig.suptitle("F1. A tip used more on easy issues looks helpful when tries are pooled", y=1.02)
    fig.tight_layout()
    return _save(fig, "f1_method", cfg)


def f2_versions(df: pd.DataFrame, cfg: Config) -> Path:
    """F2: older AI by evaluation day - success and reproduce-first share swing together."""
    o = df[(df.dataset == "older_ai") & df.dated]
    d = (
        o.assign(r=o.first_move_arm.eq("R"))
        .groupby("batch_day")
        .agg(tries=("y", "size"), rate=("y", "mean"), repro=("r", "mean"))
        .reset_index()
    )
    d = d[d.tries >= 200]
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.3))
    ax.scatter(d.repro * 100, d.rate * 100, s=np.sqrt(d.tries) * 1.5, color=BLUE, alpha=0.7)
    r = float(np.corrcoef(d.repro, d.rate)[0, 1])
    ax.set_xlabel("Tries that start by writing a script (%)")
    ax.set_ylabel("Bugs fixed (%)")
    ax.set_title(f"F2. Older AI, one dot per evaluation day (r = {r:+.2f})")
    ax.text(0.02, 0.02, "Dot size: tries that day", transform=ax.transAxes, fontsize=6, color=GREY)
    return _save(fig, "f2_versions", cfg)


def f3_older(est: pd.DataFrame, rob: dict[str, Any], cfg: Config) -> Path:
    """F3: P1 under the main design, the alternatives and the batch definitions."""
    p1 = est[est.hid == "P1"].set_index("design")
    rows: list[tuple[str, Mapping[Any, Any]]] = [
        ("Fair (issue x day) - main", p1.loc["fair"].to_dict()),
        ("Fixed-effects regression", p1.loc["fe_lpm"].to_dict()),
        ("Simple (all tries pooled)", p1.loc["simple"].to_dict()),
    ]
    for name, e in rob["p1_batches"].items():
        if name != "day":
            rows.append((f"Fair, batch = {name}", e))
    rows.append(("Fair, issue only", rob["p1_within_issue_only"]))
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.5))
    colors = [BLUE, BLUE, GREY] + [SKY] * (len(rows) - 3)
    y = np.arange(len(rows))[::-1]
    for yi, (_, r), c in zip(y, rows, colors, strict=True):
        e, lo, hi = (float(r[k]) * 100 for k in ("estimate", "ci_lo", "ci_hi"))
        ax.errorbar(e, yi, xerr=[[e - lo], [hi - e]], fmt="o", color=c, ms=4, capsize=2, lw=1)
    ax.set_yticks(y, [n for n, _ in rows])
    ax.axvline(0, color=GREY, lw=0.6, ls="--")
    ax.set_xlabel("Change in bugs fixed per 100 tries (95% CI)")
    ax.set_title("F3. Older AI: reproducing first hurts")
    return _save(fig, "f3_older", cfg)


def f4_newer(est: pd.DataFrame, cfg: Config) -> Path:
    """F4: P2 per AI and pooled, fair vs simple, with the equivalence margin."""
    margin = float(cfg.raw["analysis"]["equivalence_margin"]) * 100
    sub = est[est.hid == "P2-sub"]
    pooled = est[est.hid == "P2"]
    ais = list(sub[sub.design == "fair"].ai) + ["NVIDIA pooled"]
    both = pd.concat([sub, pooled])
    fair = both[both.design == "fair"].set_index("ai").loc[ais].reset_index()
    simple = both[both.design == "simple"].set_index("ai").loc[ais].reset_index()
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.8))
    ax.axvspan(-margin, margin, color=GREEN, alpha=0.12, lw=0)
    labels = [short_ai(a) for a in ais]
    _forest(ax, simple, labels, color=GREY, offset=-0.18, marker="s", label="Simple")
    _forest(ax, fair, labels, color=BLUE, offset=0.18, label="Fair (same issue)")
    ax.set_xlabel("Test first vs fix first, bugs fixed per 100 tries")
    ax.set_title(f"F4. Newer AIs: no meaningful effect (shaded: ±{margin:.0f} pts)")
    ax.set_ylim(-1.4, len(ais) - 0.4)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    return _save(fig, "f4_newer", cfg)


def f5_gaps(gaps: pd.DataFrame, summary: dict[str, Any], cfg: Config) -> Path:
    """F5: simple vs fair estimate for every tip x AI pair (the headline chart)."""
    g = gaps.dropna(subset=["simple", "fair"])
    fig, ax = plt.subplots(figsize=(ONE_COL, ONE_COL))
    lim = float(max(g.simple.abs().max(), g.fair.abs().max()) * 100 * 1.08)
    ax.fill_between(
        [-lim, lim], [-lim - 2, lim - 2], [-lim + 2, lim + 2], color=GREEN, alpha=0.12, lw=0
    )
    ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=0.6, ls="--")
    ax.axhline(0, color=GREY, lw=0.4)
    ax.axvline(0, color=GREY, lw=0.4)
    c = np.where(g.sign_differs, RED, BLUE)
    ax.scatter(g.simple * 100, g.fair * 100, s=12, c=c, alpha=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Simple estimate (points)")
    ax.set_ylabel("Fair estimate (points)")
    p3 = summary["decisions"]["P3"]
    ax.set_title(
        f"F5. {p3['pairs']} tip x AI pairs: mean gap {p3['mean_abs_gap'] * 100:.1f} pts",
    )
    ax.scatter([], [], c=RED, s=12, label="opposite sign")
    ax.scatter([], [], c=BLUE, s=12, label="same sign")
    ax.legend(loc="upper left", frameon=False)
    return _save(fig, "f5_gaps", cfg)


def f6_tips(est: pd.DataFrame, cfg: Config) -> Path:
    """F6: forest plot of the other tips (S1-S6), one panel per tip."""
    q = float(cfg.raw["analysis"]["fdr_q"])
    tips = [h for h in TIP_NAMES if h in set(est.hid)]
    order = list(est[est.hid == "S2"].ai)  # older AI first, then the newer AIs
    fig, axes = plt.subplots(2, 3, figsize=(TWO_COL, 4.4), sharex=True)
    fair = est[est.design == "fair"]
    for n, (ax, h) in enumerate(zip(axes.ravel(), tips, strict=False)):
        # every panel in a row uses the same slots, so labels on the left line up
        row_tips = tips[(n // 3) * 3 : (n // 3) * 3 + 3]
        slots = [a for a in order if a in set(fair[fair.hid.isin(row_tips)].ai)]
        pos = {a: len(slots) - 1 - i for i, a in enumerate(slots)}
        d = fair[fair.hid == h]
        for _, r in d.iterrows():
            e, lo, hi = r.estimate * 100, r.ci_lo * 100, r.ci_hi * 100
            weak = r.n_strata < 100
            color = GREY if weak else (RED if r.q_value < q else BLUE)
            ax.errorbar(
                e,
                pos[r.ai],
                xerr=[[e - lo], [hi - e]],
                fmt="o",
                ms=4,
                capsize=2,
                lw=1,
                color=color,
                mfc="white" if weak else color,
            )
        ax.axvline(0, color=GREY, lw=0.6, ls="--")
        ax.set_ylim(-0.6, len(slots) - 0.4)
        ticks = [pos[a] for a in slots]
        ax.set_yticks(ticks, [short_ai(a) for a in slots] if n % 3 == 0 else [""] * len(slots))
        ax.set_title(f"{h}. {TIP_NAMES[h]}")
    for ax in axes[-1]:
        ax.set_xlabel("Points (95% CI)")
    for ax in axes.ravel()[len(tips) :]:
        ax.axis("off")
    fig.suptitle(
        "F6. Other tips. Red: significant after BH correction; hollow grey: underpowered "
        "(< 100 groups)",
        y=1.0,
    )
    fig.tight_layout()
    return _save(fig, "f6_tips", cfg)


def f7_cost(est: pd.DataFrame, cfg: Config) -> Path:
    """F7: what testing first costs - extra steps, text read and text written."""
    s7 = est[est.hid == "S7"]
    rows = [
        ("Steps (older AI)", s7[(s7.ai == "older_ai") & (s7.outcome == "n_steps")]),
        ("Steps (newer AIs)", s7[(s7.ai == "NVIDIA pooled") & (s7.outcome == "n_steps")]),
        ("Characters read, k (newer)", s7[s7.outcome == "chars_in_k"]),
        ("Characters written, k (newer)", s7[s7.outcome == "chars_out_k"]),
    ]
    t = pd.concat([r for _, r in rows])
    fig, ax = plt.subplots(figsize=(ONE_COL, 1.8))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t.estimate, color=[ORANGE, ORANGE, SKY, SKY], height=0.55)
    ax.errorbar(
        t.estimate,
        y,
        xerr=[t.estimate - t.ci_lo, t.ci_hi - t.estimate],
        fmt="none",
        color="black",
        capsize=2,
        lw=0.8,
    )
    for yi, e, hi in zip(y, t.estimate, t.ci_hi, strict=True):
        ax.text(hi + 0.3, yi, f"+{e:.1f}", va="center", fontsize=7)
    ax.set_xlim(0, float(t.ci_hi.max()) * 1.15)
    ax.set_yticks(y, [n for n, _ in rows])
    ax.set_xlabel("Extra per try when testing/reproducing first (same issue)")
    ax.set_title("F7. The cost of testing first")
    return _save(fig, "f7_cost", cfg)


def f8_mechanism(df: pd.DataFrame, summary: dict[str, Any], est: pd.DataFrame, cfg: Config) -> Path:
    """F8: older AI - running out of context, and time to reach the gold file."""
    o = df[(df.dataset == "older_ai") & df.dated & df.first_move_arm.isin(["R", "S"])]
    rate = o.groupby("first_move_arm").ctx_any.mean() * 100
    s10 = est[(est.hid == "S10") & (est.outcome == "ctx_any")].iloc[0]
    km = summary["kaplan_meier"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TWO_COL, 2.2), width_ratios=[0.8, 1.4])
    a1.bar([0, 1], [rate["R"], rate["S"]], color=[ORANGE, BLUE], width=0.6)
    a1.set_xticks([0, 1], ["Reproduce first", "Search first"])
    for x, v in enumerate([rate["R"], rate["S"]]):
        a1.text(x, v + 0.5, f"{v:.0f}%", ha="center", fontsize=7)
    a1.set_ylabel("Ran out of context (%)")
    a1.set_title(f"(a) Out of context: {s10.estimate * 100:+.0f} pts (same issue)")
    for arm, color, name in (("R", ORANGE, "Reproduce first"), ("S", BLUE, "Search first")):
        curve = km[arm]["curve"]
        a2.step(
            range(len(curve)),
            curve,
            where="post",
            color=color,
            label=f"{name} (median step {km[arm]['median_step']:.0f})",
        )
    a2.set_xlabel("Step")
    a2.set_ylabel("Not yet at the gold file")
    a2.set_title("(b) Time to open the file the real fix changed")
    a2.legend(frameon=False)
    fig.suptitle("F8. Why reproducing first hurt the older AI", y=1.0)
    fig.tight_layout()
    return _save(fig, "f8_mechanism", cfg)


def run(cfg: Config, log: Callable[[str], None] = print) -> list[Path]:
    """Draw every paper figure."""
    from agent_tips.models import analysis

    res = cfg.path("results")
    est = pd.read_parquet(res / "estimates.parquet")
    gaps = pd.read_parquet(res / "gaps.parquet")
    summary = json.loads((res / "analysis_summary.json").read_text(encoding="utf-8"))
    rob = json.loads((res / "robustness.json").read_text(encoding="utf-8"))
    df = analysis.load(cfg)
    with plt.rc_context(STYLE):  # type: ignore[arg-type]
        out = [
            f1_method(cfg),
            f2_versions(df, cfg),
            f3_older(est, rob, cfg),
            f4_newer(est, cfg),
            f5_gaps(gaps, summary, cfg),
            f6_tips(est, cfg),
            f7_cost(est, cfg),
            f8_mechanism(df, summary, est, cfg),
        ]
    for p in out:
        log(f"wrote {p}")
    return out
