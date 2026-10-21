"""Step 4 (one-try study): exploratory data analysis.

Describes the clean data and lists the comparisons the study can make. Counts
and descriptions only: no correction method is run here.

Outputs:
- charts as PNG (``docs/figures/onetry/``) and PDF (``outputs/onetry/figures/``);
- ``outputs/onetry/inventory.parquet``: every candidate comparison with its counts
  and how much other-AI information its key bugs have;
- ``outputs/onetry/eda.json``: the numbers behind every chart;
- ``docs/one_try/04_eda.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from agent_tips.config import Config  # noqa: E402
from agent_tips.eda.charts import COLORS, between_issue_share, subset_label  # noqa: E402
from agent_tips.onetry.comparisons import candidates, key_counts  # noqa: E402

MIN_SHARED = 200  # shared bugs needed to compare two setups' difficulty


def short(setup: str) -> str:
    """Readable label for an AI setup."""
    if setup.startswith("swe-agent-llama"):
        return f"Older AI {setup.rsplit('-', 1)[-1].upper()}"
    if setup == "newer_ai_nebius":
        return "Qwen3-Coder (Nebius)"
    return subset_label(setup).replace(" (", " / ").replace(", ", " / ").rstrip(")")


def _save(fig: Figure, name: str, cfg: Config) -> None:
    png = cfg.root / "docs" / "figures" / "onetry"
    pdf = cfg.path("onetry_results") / "figures"
    png.mkdir(parents=True, exist_ok=True)
    pdf.mkdir(parents=True, exist_ok=True)
    fig.savefig(png / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(pdf / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


# ---------------------------------------------------------------- descriptions


def bug_vs_luck(tries: pd.DataFrame) -> pd.DataFrame:
    """Per studied setup: share of outcome variation explained by the bug, and mixed bugs."""
    rows = []
    for setup, g in tries[tries.studied].groupby("setup"):
        per = g.groupby("bug_id").y.agg(["mean", "count"])
        per = per[per["count"] >= 2]
        share = between_issue_share(g.rename(columns={"bug_id": "issue_key", "y": "resolved"}))
        mixed = float(((per["mean"] > 0) & (per["mean"] < 1)).mean()) if len(per) else float("nan")
        rows.append(
            {"setup": setup, "bug_share": share, "mixed_bugs": mixed, "bugs_2plus": len(per)}
        )
    return pd.DataFrame(rows)


def agreement(tries: pd.DataFrame) -> pd.DataFrame:
    """Correlation of per-bug fix rates between every pair of setups with enough shared bugs."""
    rates = tries.groupby(["setup", "bug_id"]).y.mean().unstack("setup")
    rows = []
    cols = list(rates.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            x = rates[[a, b]].dropna()
            if len(x) >= MIN_SHARED:
                rows.append({"a": a, "b": b, "shared_bugs": len(x), "corr": float(x[a].corr(x[b]))})
    return pd.DataFrame(rows, columns=["a", "b", "shared_bugs", "corr"])


def report_hints(tries: pd.DataFrame, bugs: pd.DataFrame) -> pd.DataFrame:
    """Per-bug fix rate by report facts (studied setups only)."""
    per = (
        tries[tries.studied]
        .groupby(["dataset", "bug_id"])
        .y.mean()
        .rename("fix_rate")
        .reset_index()
    )
    d = per.merge(bugs, on=["dataset", "bug_id"])
    d["long_report"] = d.groupby("dataset").report_chars.transform(lambda s: s > s.median())
    rows = []
    for ds, g in d.groupby("dataset"):
        for fact in ("has_code_block", "has_stack_trace", "names_file", "long_report"):
            yes, no = g[g[fact] == True], g[g[fact] == False]  # noqa: E712
            rows.append(
                {
                    "dataset": ds,
                    "fact": fact,
                    "share_yes": float((g[fact] == True).mean()),  # noqa: E712
                    "fix_rate_yes": float(yes.fix_rate.mean()) if len(yes) else float("nan"),
                    "fix_rate_no": float(no.fix_rate.mean()) if len(no) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _largest_hint(hints: pd.DataFrame) -> dict[str, Any]:
    """Return the report fact with the largest fix-rate difference (for the summary)."""
    h = hints.assign(gap=(hints.fix_rate_yes - hints.fix_rate_no).abs()).dropna(subset=["gap"])
    if h.empty:
        return {
            "hint_gap": 0.0,
            "hint_fact": "-",
            "hint_dataset": "-",
            "hint_yes": 0.0,
            "hint_no": 0.0,
        }
    r = h.sort_values("gap").iloc[-1]
    return {
        "hint_gap": float(r["gap"]),
        "hint_fact": str(r["fact"]),
        "hint_dataset": str(r["dataset"]),
        "hint_yes": float(r["fix_rate_yes"]),
        "hint_no": float(r["fix_rate_no"]),
    }


def other_ai_info(tries: pd.DataFrame, setup: str, key_bugs: set[str]) -> dict[str, float]:
    """How much other-AI information the key bugs of one setup have (descriptive)."""
    others = tries[(tries.setup != setup) & tries.bug_id.isin(key_bugs)]
    if not key_bugs:
        return {"share_with_other": 0.0, "mean_other_setups": 0.0, "agreement": float("nan")}
    n_other = others.groupby("bug_id").setup.nunique().reindex(sorted(key_bugs)).fillna(0)
    own = tries[(tries.setup == setup) & tries.bug_id.isin(key_bugs)].groupby("bug_id").y.mean()
    oth = others.groupby(["bug_id", "setup"]).y.mean().groupby("bug_id").mean()
    j = pd.concat([own.rename("own"), oth.rename("other")], axis=1).dropna()
    return {
        "share_with_other": float((n_other > 0).mean()),
        "mean_other_setups": float(n_other.mean()),
        "agreement": float(j.own.corr(j.other)) if len(j) >= 30 else float("nan"),
    }


def inventory(tries: pd.DataFrame, studied_older: str, min_bugs: int) -> pd.DataFrame:
    """Every candidate comparison with its counts and other-AI information."""
    rows = []
    for c in candidates(tries, studied_older):
        k = key_counts(c, tries)
        d = c.rows(tries)
        grp = c.groups(d)
        both = d.assign(g=grp).groupby("g")[c.column].nunique()
        key_bugs = set(d[grp.isin(both[both == 2].index)].bug_id)
        rows.append(
            {
                "cid": c.cid,
                "dataset": c.dataset,
                "setup": c.setup,
                "behaviour": c.behaviour,
                "column": c.column,
                "a": c.a,
                "b": c.b,
                "kind": "true experiment" if c.experiment else "mid-run",
                **k,
                **other_ai_info(tries, c.setup, key_bugs),
                "qualifies": k["key_bugs"] >= min_bugs,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- charts


def chart_fix_rates(setups: pd.DataFrame, cfg: Config) -> None:
    """O1: fix rate per AI setup (grey: other AIs only)."""
    d = setups.sort_values("fix_rate")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = [COLORS[0] if s else COLORS[7] for s in d.studied]
    ax.barh([short(s) for s in d.setup], d.fix_rate * 100, color=colors)
    ax.set_xlabel("Tries that fixed the bug (%)")
    ax.set_title("O1. How often each AI setup fixes a bug (grey: used only as other AIs)")
    ax.tick_params(axis="y", labelsize=7)
    _save(fig, "o1_fix_rates", cfg)


def chart_bug_vs_luck(bl: pd.DataFrame, cfg: Config) -> None:
    """O2: share of outcome variation explained by the bug vs luck."""
    d = bl.sort_values("bug_share")
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(d))
    ax.barh(y, d.bug_share, color=COLORS[0], label="Explained by which bug it is")
    ax.barh(y, 1 - d.bug_share, left=d.bug_share, color=COLORS[1], label="Luck between tries")
    ax.set_yticks(y, [short(s) for s in d.setup], fontsize=7)
    ax.set_xlabel("Share of variation in fixed / not fixed")
    ax.set_title("O2. Is a fix decided by the bug or by luck?")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _save(fig, "o2_bug_vs_luck", cfg)


def chart_agreement(ag: pd.DataFrame, cfg: Config) -> None:
    """O3: heat map of per-bug fix-rate correlations between setups."""
    names = sorted(set(ag.a) | set(ag.b))
    m = pd.DataFrame(np.nan, index=names, columns=names)
    for r in ag.itertuples():
        m.loc[r.a, r.b] = m.loc[r.b, r.a] = r.corr
    fig, ax = plt.subplots(figsize=(7.5, 6))
    arr = m.to_numpy(float)
    im = ax.imshow(arr, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(names)), [short(n) for n in names], rotation=90, fontsize=6)
    ax.set_yticks(range(len(names)), [short(n) for n in names], fontsize=6)
    for i in range(len(names)):
        for j in range(len(names)):
            v = float(arr[i, j])
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if v > 0.6 else "black",
                )
    fig.colorbar(im, ax=ax, shrink=0.7, label="Correlation of per-bug fix rates")
    ax.set_title(
        f"O3. Do different AIs find the same bugs hard? (pairs with ≥{MIN_SHARED} shared bugs)"
    )
    _save(fig, "o3_agreement", cfg)


def chart_steps(tries: pd.DataFrame, cfg: Config) -> None:
    """O4: distribution of actions per try."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, (ds, g) in enumerate(tries[tries.studied].groupby("dataset")):
        ax.hist(g.n_steps.clip(upper=300), bins=60, alpha=0.55, color=COLORS[i], label=str(ds))
    ax.set_yscale("log")
    ax.set_xlabel("Actions per try (capped at 300 for display)")
    ax.set_ylabel("Tries (log scale)")
    ax.set_title("O4. Length of tries; outliers are flagged, not removed")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, "o4_steps", cfg)


def chart_inventory(inv: pd.DataFrame, min_bugs: int, cfg: Config) -> None:
    """O5: key bugs per candidate comparison."""
    d = inv.sort_values("key_bugs")
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.16 * len(d))))
    colors = [COLORS[2] if k == "true experiment" else COLORS[0] for k in d.kind]
    ax.barh(range(len(d)), d.key_bugs, color=colors)
    ax.axvline(min_bugs, color="black", lw=0.8, ls="--")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yticks(
        range(len(d)),
        [f"{short(s)}: {b}" for s, b in zip(d.setup, d.behaviour, strict=True)],
        fontsize=5,
    )
    ax.set_xlabel("Key bugs (both choices seen); dashed line = minimum to qualify")
    ax.set_title("O5. Which comparisons are possible? (green: true experiments)")
    _save(fig, "o5_inventory", cfg)


# ---------------------------------------------------------------- report


def _pct(x: float) -> str:
    return "—" if x != x else f"{x:.0%}"


def write_report(
    cfg: Config, s: dict[str, Any], inv: pd.DataFrame, path: Path | None = None
) -> Path:
    """Write ``docs/one_try/04_eda.md``."""
    q = inv[inv.qualifies]
    lines = [
        "# Step 4: Exploratory Data Analysis",
        "",
        "*Generated by `agent-tips onetry eda`. Paper 2, the one-try study. Counts and descriptions "
        "only: no correction method is run in this step.*",
        "",
        "## In plain words",
        "",
        f"- **Bug or luck?** Which bug it is explains {s['bug_share_min']:.0%}–{s['bug_share_max']:.0%} "
        "of whether a try fixes it; the rest is luck between tries. This is why one-try studies can be fooled by easy bugs.",
        f"- **Do AIs agree on which bugs are hard?** Among NVIDIA setups, strongly (median correlation "
        f"{s['agree_nvidia_median']:.2f}). Between the 2024 older AI and the 2026 Nebius AI, weakly "
        f"({s['agree_cross_gen']:.2f}). So other AIs' results should help most for the NVIDIA setups.",
        f"- **Does the bug report hint at difficulty?** Somewhat: the largest gap is "
        f"{s['hint_gap'] * 100:.0f} points ({s['hint_fact']} in `{s['hint_dataset']}`: "
        f"{s['hint_yes']:.0%} fixed with it vs {s['hint_no']:.0%} without). So a model that reads "
        "the report has something to work with; how much it actually removes the easy-bug effect is "
        "tested in Step 6.",
        f"- **What can we test?** {len(q)} comparisons qualify (at least {s['min_bugs']} bugs with both "
        f"choices): {int((q.kind == 'true experiment').sum())} true experiments (first moves) and "
        f"{int((q.kind == 'mid-run').sum())} mid-run choices. This list is now fixed for modelling.",
        "",
        "## O1. Fix rates by AI setup",
        "",
        "![Fix rates](../figures/onetry/o1_fix_rates.png)",
        "",
        "## O2. Bug or luck?",
        "",
        "![Bug vs luck](../figures/onetry/o2_bug_vs_luck.png)",
        "",
        "| AI setup | Explained by the bug | Bugs with mixed outcomes | Bugs with ≥2 tries |",
        "| --- | --- | --- | --- |",
    ]
    for r in s["bug_vs_luck"]:
        lines.append(
            f"| {short(r['setup'])} | {_pct(r['bug_share'])} | {_pct(r['mixed_bugs'])} | {r['bugs_2plus']:,} |"
        )
    lines += [
        "",
        "## O3. Do different AIs find the same bugs hard?",
        "",
        "![Agreement](../figures/onetry/o3_agreement.png)",
        "",
        "| Setup A | Setup B | Shared bugs | Correlation |",
        "| --- | --- | --- | --- |",
    ]
    for r in sorted(s["agreement"], key=lambda r: -r["shared_bugs"]):
        lines.append(
            f"| {short(r['a'])} | {short(r['b'])} | {r['shared_bugs']:,} | {r['corr']:.2f} |"
        )
    lines += [
        "",
        "## Does the bug report hint at difficulty?",
        "",
        "| Dataset | Report fact | Bugs with it | Fix rate with | Fix rate without |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in s["report_hints"]:
        lines.append(
            f"| `{r['dataset']}` | {r['fact']} | {_pct(r['share_yes'])} | {_pct(r['fix_rate_yes'])} | {_pct(r['fix_rate_no'])} |"
        )
    lines += [
        "",
        "## O4. Length of tries",
        "",
        "![Steps](../figures/onetry/o4_steps.png)",
        "",
        f"Flagged as outliers (zero actions or above the 99.9th percentile): {s['steps_outliers']:,} tries. "
        "They are kept: an unusually long try is still a real try.",
        "",
        "## O5. Comparison inventory",
        "",
        "![Inventory](../figures/onetry/o5_inventory.png)",
        "",
        "*Key bugs: bugs where both choices occur. Other-AI columns describe those key bugs: the share "
        "with at least one other setup's result, the mean number of other setups, and how well this setup's "
        "per-bug fix rate agrees with the other setups' (descriptive only).*",
        "",
        "| Qualifies | Kind | AI setup | Behaviour (a vs b) | Key bugs | Tries a / b | Share a | With other AIs | Other setups (mean) | Agreement |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    ordered = inv.sort_values(["qualifies", "kind", "key_bugs"], ascending=[False, False, False])
    for r in ordered.to_dict(orient="records"):
        agr = float(r["agreement"])
        ag = "—" if agr != agr else f"{agr:.2f}"
        lines.append(
            f"| {'yes' if r['qualifies'] else 'no'} | {r['kind']} | {short(str(r['setup']))} | "
            f"{r['behaviour']} | {int(r['key_bugs']):,} | {int(r['tries_a']):,} / {int(r['tries_b']):,} | "
            f"{float(r['share_a']):.0%} | {float(r['share_with_other']):.0%} | "
            f"{float(r['mean_other_setups']):.1f} | {ag} |"
        )
    lines += [
        "",
        "## Notes for the next steps",
        "",
        "- The qualifying comparisons above are the fixed list for Step 6. Comparisons that do not qualify are not used.",
        '- "Check your fix" and the other after-fix choices only count tries that edited code (as in Paper 1).',
        "- The older AI's comparisons use dated tries in bug × day groups for the answer key; its one-try copies ignore the day, as a one-try researcher would.",
        "",
    ]
    path = path or cfg.root / "docs" / "one_try" / "04_eda.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def run(cfg: Config, log: Callable[[str], None] = print) -> Path:
    """Describe the clean data, build the comparison inventory and write the report."""
    clean = cfg.path("onetry_data") / "clean"
    tries = pd.read_parquet(clean / "tries.parquet")
    bugs = pd.read_parquet(clean / "bugs.parquet")
    setups = pd.read_parquet(clean / "setups.parquet")
    min_bugs = int(cfg.raw["onetry"]["min_bugs"])
    studied_older = cfg.raw["datasets"]["older_ai"]["model_name"]
    log("descriptions")
    bl = bug_vs_luck(tries)
    ag = agreement(tries)
    hints = report_hints(tries, bugs)
    log("comparison inventory")
    inv = inventory(tries, studied_older, min_bugs)
    res = cfg.path("onetry_results")
    res.mkdir(parents=True, exist_ok=True)
    inv.to_parquet(res / "inventory.parquet", index=False)
    log("charts")
    chart_fix_rates(setups, cfg)
    chart_bug_vs_luck(bl, cfg)
    if not ag.empty:
        chart_agreement(ag, cfg)
    chart_steps(tries, cfg)
    chart_inventory(inv, min_bugs, cfg)
    nv = ag[ag.a.str.contains("/") & ag.b.str.contains("/")]
    cross = ag[
        (ag.a == studied_older) & (ag.b == "newer_ai_nebius")
        | (ag.b == studied_older) & (ag.a == "newer_ai_nebius")
    ]
    summary: dict[str, Any] = {
        "min_bugs": min_bugs,
        "bug_vs_luck": bl.to_dict(orient="records"),
        "bug_share_min": float(bl.bug_share.min()),
        "bug_share_max": float(bl.bug_share.max()),
        "agreement": ag.to_dict(orient="records"),
        "agree_nvidia_median": float(nv["corr"].median()) if len(nv) else float("nan"),
        "agree_cross_gen": float(cross["corr"].iloc[0]) if len(cross) else float("nan"),
        "report_hints": hints.to_dict(orient="records"),
        **_largest_hint(hints),
        "steps_outliers": int(tries.steps_outlier.sum()),
        "qualifying": inv[inv.qualifies].cid.tolist(),
    }
    (res / "eda.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return write_report(cfg, summary, inv)
