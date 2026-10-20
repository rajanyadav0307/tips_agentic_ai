"""Step 4: exploration charts and the numbers behind them.

Charts are saved twice: PNG under ``docs/figures/eda/`` (shown in the docs and
kept in git) and PDF under ``outputs/figures/eda/``. The numbers behind every
chart are written to ``outputs/results/eda_stats.json``.

All charts use the analysis-ready tries: in scope, same issue text, and a
known outcome.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import duckdb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from agent_tips.config import Config  # noqa: E402

# Colorblind-safe palette (Okabe-Ito).
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

LABELS = {
    "older_ai": "Older AI (Llama-70B, SWE-agent 2024)",
    "newer_ai_nebius": "Qwen3-Coder (OpenHands, Nebius)",
}


def subset_label(subset: str) -> str:
    """Readable name for a subset."""
    if subset in LABELS:
        return LABELS[subset]
    scaffold, ai, source = subset.split("/")
    names = {
        "minimax_m25": "MiniMax-M2.5",
        "qwen35_122b": "Qwen3.5-122B",
        "qwen36_27b": "Qwen3.6-27B",
        "deepseek_v4_flash_0731": "DeepSeek-V4-Flash",
    }
    tool = {"sweagent": "SWE-agent", "openhands": "OpenHands"}.get(scaffold, scaffold)
    return f"{names.get(ai, ai)} ({tool}, {source})"


def _save(fig: Figure, name: str, cfg: Config) -> None:
    png = cfg.root / "docs" / "figures" / "eda"
    pdf = cfg.path("figures") / "eda"
    png.mkdir(parents=True, exist_ok=True)
    pdf.mkdir(parents=True, exist_ok=True)
    fig.savefig(png / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(pdf / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _records(
    df: pd.DataFrame, key: str, fields: dict[str, tuple[str, type[int] | type[float]]]
) -> dict[str, dict[str, Any]]:
    """Turn a small table into ``{key: {name: value}}``, rounding floats to 4 decimals."""
    out: dict[str, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        vals: dict[str, Any] = {}
        for name, (col, kind) in fields.items():
            v = kind(row[col])
            vals[name] = round(v, 4) if kind is float else v
        out[str(row[key])] = vals
    return out


def _wilson(k: float, n: float) -> tuple[float, float]:
    """95% Wilson interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_tries(cfg: Config) -> pd.DataFrame:
    """Analysis-ready tries."""
    path = (cfg.path("processed") / "tries.parquet").as_posix()
    return duckdb.sql(
        f"SELECT * FROM read_parquet('{path}') WHERE in_scope AND same_input AND outcome_known"
    ).df()


def first_actions(cfg: Config) -> pd.DataFrame:
    """Return the first action of every in-scope try (raw tool/action names, no labels)."""
    steps = (cfg.path("processed") / "steps").as_posix()
    return duckdb.sql(
        f"""SELECT try_id, CASE WHEN tool = 'str_replace_editor' THEN 'editor ' || action
                                 WHEN tool IN ('bash', 'execute_bash') THEN 'shell ' || action
                                 ELSE tool END AS first_action
            FROM read_parquet('{steps}/*.parquet') WHERE step = 0"""
    ).df()


def chart_tries_per_issue(t: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """E1: how many tries each issue has (left: many-try datasets; right: NVIDIA, 1-3 tries)."""
    k = t.groupby(["subset", "issue_key"]).size().rename("k").reset_index()
    stats: dict[str, Any] = {
        str(s): {
            "issues": int(len(g)),
            "median": float(g.k.median()),
            "p90": float(g.k.quantile(0.9)),
            "max": int(g.k.max()),
        }
        for s, g in k.groupby("subset")
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.2, 1]})
    many = [s for s in ("older_ai", "newer_ai_nebius") if s in stats]
    for i, s in enumerate(many):
        g = k[k.subset == s].k
        a1.hist(
            g,
            bins=np.arange(1, g.max() + 2) - 0.5,
            alpha=0.6,
            color=COLORS[i],
            label=subset_label(s),
            density=True,
        )
    a1.set_xlabel("Tries per issue")
    a1.set_ylabel("Share of issues")
    a1.set_title("Older AI and Qwen3-Coder")
    a1.legend(fontsize=7, frameon=False)
    few = sorted((s for s in stats if s not in many), key=subset_label)
    width = 0.8 / 3
    for j, n in enumerate((1, 2, 3)):
        vals = [float((k[k.subset == s].k == n).mean()) for s in few]
        a2.barh(
            np.arange(len(few)) + (j - 1) * width,
            vals,
            height=width,
            color=COLORS[j + 2],
            label=f"{n} tr{'y' if n == 1 else 'ies'}",
        )
    a2.set_yticks(range(len(few)), [subset_label(s) for s in few], fontsize=7)
    a2.set_xlabel("Share of issues")
    a2.set_title("Newer AIs (NVIDIA)")
    a2.legend(fontsize=7, frameon=False)
    fig.suptitle("E1. How many times each issue was tried")
    fig.tight_layout()
    _save(fig, "e1_tries_per_issue", cfg)
    return stats


def chart_success_by_ai(t: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """E2: resolution rate per subset with 95% Wilson intervals."""
    g = t.groupby("subset").resolved.agg(["sum", "count"]).reset_index()
    g["rate"] = g["sum"] / g["count"]
    g = g.sort_values("rate")
    ci = np.array([_wilson(k, n) for k, n in zip(g["sum"], g["count"], strict=True)])
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(g))
    ax.errorbar(g.rate, y, xerr=[g.rate - ci[:, 0], ci[:, 1] - g.rate], fmt="o", color=COLORS[0])
    ax.set_yticks(y, [subset_label(s) for s in g.subset], fontsize=7)
    ax.set_xlabel("Share of tries that fixed the issue")
    ax.set_title("E2. Success rate by AI")
    _save(fig, "e2_success_by_ai", cfg)
    return _records(g, "subset", {"tries": ("count", int), "rate": ("rate", float)})


def chart_versions_by_day(t: pd.DataFrame, fa: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """E3: older AI - success rate and reproduce-first share by evaluation day."""
    o = t[(t.dataset == "older_ai") & t.dated].merge(fa, on="try_id", how="left")
    o["reproduce_first"] = o.first_action.eq("create")
    d = (
        o.groupby("batch_day")
        .agg(tries=("try_id", "size"), rate=("resolved", "mean"), repro=("reproduce_first", "mean"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    x = pd.to_datetime(d.batch_day)
    ax.plot(x, d.rate, marker="o", color=COLORS[0], label="Success rate")
    ax.plot(
        x, d.repro, marker="s", color=COLORS[1], label="Share starting with `create` (reproduce)"
    )
    ax.set_ylabel("Share of tries")
    ax.set_title("E3. Older AI by evaluation day: hidden mixture of model versions")
    ax.legend(fontsize=7, frameon=False)
    fig.autofmt_xdate()
    _save(fig, "e3_older_ai_by_day", cfg)
    return {
        "days": int(len(d)),
        "rate_min": round(float(d.rate.min()), 4),
        "rate_max": round(float(d.rate.max()), 4),
        "repro_min": round(float(d.repro.min()), 4),
        "repro_max": round(float(d.repro.max()), 4),
    }


def chart_first_actions(t: pd.DataFrame, fa: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """E4: distribution of the first action per subset (top 5 per subset)."""
    m = t[["try_id", "subset"]].merge(fa, on="try_id")
    share: pd.DataFrame = (
        m.groupby("subset").first_action.value_counts(normalize=True).rename("share").reset_index()
    )
    top = share.groupby("subset").head(5)
    subsets = sorted(map(str, top["subset"].unique()), key=subset_label)
    actions = list(dict.fromkeys(top.sort_values("share", ascending=False)["first_action"]))[:8]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    left = np.zeros(len(subsets))
    for i, a in enumerate(actions):
        vals = np.array(
            [share[(share.subset == s) & (share.first_action == a)].share.sum() for s in subsets]
        )
        ax.barh(range(len(subsets)), vals, left=left, color=COLORS[i % len(COLORS)], label=a)
        left += vals
    ax.barh(range(len(subsets)), 1 - left, left=left, color="#DDDDDD", label="other")
    ax.set_yticks(range(len(subsets)), [subset_label(s) for s in subsets], fontsize=7)
    ax.set_xlabel("Share of tries")
    ax.set_title("E4. What each AI does first")
    ax.legend(fontsize=6, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, "e4_first_actions", cfg)
    return {
        str(s): {
            str(a): round(float(v), 4) for a, v in zip(g["first_action"], g["share"], strict=True)
        }
        for s, g in top.groupby("subset")
    }


def chart_steps(t: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """E5: steps per try, and how often tries end by hitting a limit."""
    limit = t.exit_status.fillna("").str.contains("exit_context|maximum iteration|exit_cost")
    t = t.assign(hit_limit=limit)
    subsets = sorted(t.subset.unique(), key=subset_label)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [t[t.subset == s].n_steps.clip(upper=300) for s in subsets]
    ax.boxplot(data, orientation="horizontal", showfliers=False)
    ax.set_yticks(range(1, len(subsets) + 1), [subset_label(s) for s in subsets], fontsize=7)
    ax.set_xlabel("AI actions per try")
    ax.set_title("E5. Length of a try")
    _save(fig, "e5_steps_per_try", cfg)
    return {
        s: {
            "median_steps": float(t[t.subset == s].n_steps.median()),
            "hit_limit_share": round(float(t[t.subset == s].hit_limit.mean()), 4),
        }
        for s in subsets
    }


def chart_languages(t: pd.DataFrame, cfg: Config, min_tries: int = 500) -> dict[str, Any]:
    """E6: NVIDIA tries by language and success rate by language."""
    n = t[t.dataset == "newer_ai_nvidia"]
    g = n.groupby("language").resolved.agg(["sum", "count"]).reset_index()
    g = g[g["count"] >= min_tries].sort_values("count")
    if g.empty:
        return {}
    g["rate"] = g["sum"] / g["count"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    y = np.arange(len(g))
    a1.barh(y, g["count"], color=COLORS[0])
    a1.set_yticks(y, g.language)
    a1.set_xlabel("Tries")
    ci = np.array([_wilson(k, c) for k, c in zip(g["sum"], g["count"], strict=True)])
    a2.errorbar(g.rate, y, xerr=[g.rate - ci[:, 0], ci[:, 1] - g.rate], fmt="o", color=COLORS[2])
    a2.set_xlabel("Success rate")
    fig.suptitle("E6. Newer AIs (NVIDIA): languages")
    _save(fig, "e6_languages", cfg)
    return _records(g, "language", {"tries": ("count", int), "rate": ("rate", float)})


def between_issue_share(g: pd.DataFrame) -> float:
    """Share of variance in success explained by the issue (one-way ANOVA ICC)."""
    k = g.groupby("issue_key").resolved.agg(["mean", "count", "var"])
    k = k[k["count"] >= 2]
    if len(k) < 2:
        return float("nan")
    n0 = k["count"].mean()
    msw = float((k["var"] * (k["count"] - 1)).sum() / (k["count"] - 1).sum())
    grand = float((k["mean"] * k["count"]).sum() / k["count"].sum())
    msb = float((k["count"] * (k["mean"] - grand) ** 2).sum() / (len(k) - 1))
    s2b = max((msb - msw) / n0, 0.0)
    return s2b / (s2b + msw) if s2b + msw > 0 else float("nan")


def chart_variance(t: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """E7: how much of success depends on the issue vs. chance between tries."""
    rows = []
    for s, g in t.groupby("subset"):
        per = g.groupby("issue_key").resolved.agg(["mean", "count"])
        per = per[per["count"] >= 2]
        mixed = float(((per["mean"] > 0) & (per["mean"] < 1)).mean())
        rows.append((s, between_issue_share(g), mixed))
    d = pd.DataFrame(rows, columns=["subset", "issue_share", "mixed"]).sort_values("issue_share")
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(d))
    ax.barh(y, d.issue_share, color=COLORS[0], label="Explained by the issue")
    ax.barh(y, 1 - d.issue_share, left=d.issue_share, color=COLORS[1], label="Chance between tries")
    ax.set_yticks(y, [subset_label(s) for s in d.subset], fontsize=7)
    ax.set_xlabel("Share of variation in success")
    ax.set_title("E7. Is success decided by the issue or by chance?")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _save(fig, "e7_variance_split", cfg)
    return _records(
        d, "subset", {"issue_share": ("issue_share", float), "share_issues_mixed": ("mixed", float)}
    )


def run(cfg: Config, log: Callable[[str], None] = print) -> Path:
    """Make all exploration charts and write their numbers to ``eda_stats.json``."""
    t = load_tries(cfg)
    fa = first_actions(cfg)
    stats = {
        "e1_tries_per_issue": chart_tries_per_issue(t, cfg),
        "e2_success_by_ai": chart_success_by_ai(t, cfg),
        "e3_older_ai_by_day": chart_versions_by_day(t, fa, cfg),
        "e4_first_actions": chart_first_actions(t, fa, cfg),
        "e5_steps_per_try": chart_steps(t, cfg),
        "e6_languages": chart_languages(t, cfg),
        "e7_variance_split": chart_variance(t, cfg),
    }
    out = cfg.path("results") / "eda_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    log(f"wrote 7 charts and {out}")
    return out
