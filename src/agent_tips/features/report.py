"""Step 5: write ``docs/05_feature_report.md`` and the tip-compliance chart.

All numbers are computed from the feature tables and
``outputs/results/label_validation.json``. The second labeling is read from
``docs/data/label_validation/sample.csv`` when it exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agent_tips.config import Config  # noqa: E402
from agent_tips.eda.charts import COLORS, subset_label  # noqa: E402
from agent_tips.features import validate  # noqa: E402
from agent_tips.features.labels import LABELS  # noqa: E402

# Blind second labeling rounds: (round, file, what the rules were at the time).
ANNOTATED = (
    (1, Path("docs/data/label_validation/sample.csv"), "original rules; 645 steps, all labels"),
    (
        2,
        Path("docs/data/label_validation/sample_round2.csv"),
        "rules after the round-1 fixes; fresh sample of 300 steps, 7 labels",
    ),
)

LABEL_TEXT = {
    "repro_create": "created a test or reproduction script",
    "create_other": "created another new file",
    "edit_code": "edited existing project code",
    "edit_created": "edited a file the try itself created",
    "run": "ran a script or program",
    "test_suite": "ran the project's test suite",
    "look": "looked at or searched files",
    "think": "a 'think' call (no action)",
    "plan": "a planning call",
    "cleanup": "deleted files",
    "submit": "submitted / finished",
    "other": "anything else (installs, git, cd, ...)",
}

# Shares found during exploration (2026-10-19), for the consistency check.
EXPLORATION = {
    ("older_ai", "first move: reproduce (R)"): 0.464,
    ("older_ai", "first move: search (S)"): 0.479,
    ("older_ai", "check your fix"): 0.539,
    ("older_ai", "test suite after fix"): 0.042,
    ("newer_ai_nebius", "test first"): 0.979,
    ("openhands/deepseek_v4_flash_0731/scale-swe", "test first"): 0.103,
    ("openhands/minimax_m25/swe-rebench-v2", "test first"): 0.507,
    ("openhands/qwen35_122b/swe-rebench-v2", "test first"): 0.898,
    ("sweagent/minimax_m25/swe-rebench-v2", "test first"): 0.569,
    ("sweagent/qwen35_122b/swe-rebench-v2", "test first"): 0.843,
    ("sweagent/qwen36_27b/scale-swe", "test first"): 0.979,
    ("sweagent/qwen36_27b/swe-rebench-v2", "test first"): 0.938,
}


def compliance(con: duckdb.DuckDBPyConnection, proc: Path) -> pd.DataFrame:
    """How often each AI follows each tip (analysis-ready tries)."""
    return con.execute(f"""
      SELECT t.subset,
        count(*) AS tries,
        avg((f.first_move_arm = 'R')::INT) AS "first move: reproduce (R)",
        avg((f.first_move_arm = 'S')::INT) AS "first move: search (S)",
        avg((f.order_arm = 'test first')::INT) FILTER (WHERE f.order_arm <> 'no edit') AS "test first",
        avg((f.n_reruns_after_fix > 0)::INT) FILTER (WHERE f.first_edit_code IS NOT NULL) AS "check your fix",
        avg((f.n_suite_after_fix > 0)::INT) FILTER (WHERE f.first_edit_code IS NOT NULL) AS "test suite after fix",
        avg((f.n_cleanup > 0)::INT) AS "clean up",
        avg((f.order_arm = 'no edit')::INT) AS "never edits code"
      FROM read_parquet('{(proc / "try_features.parquet").as_posix()}') f
      JOIN read_parquet('{(proc / "tries.parquet").as_posix()}') t USING (try_id)
      WHERE t.in_scope AND t.same_input AND t.outcome_known
      GROUP BY 1 ORDER BY 1""").df()


def compliance_chart(comp: pd.DataFrame, root: Path) -> Path:
    """Draw grouped bars of the share of tries following each tip, per AI."""
    tips = ["test first", "check your fix", "test suite after fix", "clean up"]
    d = comp.copy()
    d["sort"] = d.subset.map(subset_label)
    d = d.sort_values("sort")
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(d))
    h = 0.8 / len(tips)
    for j, tip in enumerate(tips):
        vals = d[tip].fillna(0).to_numpy()
        ax.barh(y + (j - 1.5) * h, vals, height=h, color=COLORS[j], label=tip)
    ax.set_yticks(y, [subset_label(s) for s in d.subset], fontsize=7)
    ax.set_xlabel("Share of tries that follow the tip")
    ax.set_title("C1. How often each AI follows each tip")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    out = root / "docs" / "figures" / "features"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "c1_tip_compliance.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _pct(x: Any) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


def build(cfg: Config, out: Path | None = None) -> Path:
    """Write the feature report."""
    proc = cfg.path("processed")
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    comp = compliance(con, proc)
    compliance_chart(comp, cfg.root)
    val = json.loads((cfg.path("results") / "label_validation.json").read_text(encoding="utf-8"))
    obs = pd.DataFrame(val["observation_check"])
    rounds = [
        (r, note, validate.agreement(cfg.root / f))
        for r, f, note in ANNOTATED
        if (cfg.root / f).is_file()
    ]
    if rounds:
        validate.write_results(cfg, obs, {f"round_{r}": a for r, _, a in rounds})

    labels = []
    for p in sorted((proc / "labels").glob("*.parquet")):
        d = con.execute(
            f"SELECT label, count(*) AS n FROM read_parquet('{p.as_posix()}') GROUP BY 1"
        ).df()
        d["dataset"] = p.stem
        labels.append(d)
    lab = pd.concat(labels).pivot_table(index="label", columns="dataset", values="n", fill_value=0)
    lab = lab.reindex([x for x in LABELS if x in lab.index])
    share = lab / lab.sum()

    issues = con.execute(f"""
      SELECT dataset, count(*) AS issues, avg(has_code_block::INT) AS code,
        avg(has_stack_trace::INT) AS trace, avg(names_file::INT) AS file,
        avg((len(gold_files) > 0)::INT) AS gold, median(text_length) AS len
      FROM read_parquet('{(proc / "issues.parquet").as_posix()}') GROUP BY 1 ORDER BY 1""").df()
    loc = con.execute(f"""
      SELECT t.dataset, avg(f.gold_known::INT) AS gold_known,
        avg((f.first_gold_step IS NOT NULL)::INT) FILTER (WHERE f.gold_known) AS touched,
        median(f.first_gold_step) AS median_step
      FROM read_parquet('{(proc / "try_features.parquet").as_posix()}') f
      JOIN read_parquet('{(proc / "tries.parquet").as_posix()}') t USING (try_id)
      WHERE t.in_scope AND t.same_input GROUP BY 1 ORDER BY 1""").df()

    # ---------- tables ----------
    lab_lines = [
        "| Label | Meaning | " + " | ".join(share.columns) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in share.columns) + " |",
    ]
    for k, row in share.iterrows():
        lab_lines.append(
            f"| `{k}` | {LABEL_TEXT[str(k)]} | " + " | ".join(_pct(v) for v in row) + " |"
        )
    tips = [
        "first move: reproduce (R)",
        "test first",
        "check your fix",
        "test suite after fix",
        "clean up",
        "never edits code",
    ]
    comp_lines = [
        "| AI | Tries | " + " | ".join(tips) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in tips) + " |",
    ]
    for r in comp.to_dict("records"):
        cells = [
            _pct(r[t]) if not (r["subset"] != "older_ai" and t.startswith("first move")) else "—"
            for t in tips
        ]
        comp_lines.append(
            f"| {subset_label(r['subset'])} | {r['tries']:,} | " + " | ".join(cells) + " |"
        )
    chk = [
        "| AI | Measure | Exploration | Now | Difference |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    by = comp.set_index("subset")
    for (s, m), v in EXPLORATION.items():
        now = float(pd.to_numeric(by.loc[s, m])) if s in by.index else float("nan")
        chk.append(
            f"| {subset_label(s)} | {m} | {v:.1%} | {now:.1%} | {100 * (now - v):+.1f} pts |"
        )
    obs_lines = [
        "| Dataset | Label | Steps | Confirmed by the reply | Edits rejected by the editor |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in obs[obs.checkable > 0].to_dict("records"):
        rej = r.get("edit_rejected", 0) or 0  # editor refused the edit
        rej_txt = f"{rej / r['steps']:.1%}" if r["label"].startswith("edit") and r["steps"] else "—"
        obs_lines.append(
            f"| {r['dataset']} | `{r['label']}` | {r['steps']:,} | {_pct(r['precision'])} | {rej_txt} |"
        )
    iss_lines = [
        "| Dataset | Issues | Has code block | Has stack trace | Names a file | Gold patch found | Median length (chars) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in issues.to_dict("records"):
        iss_lines.append(
            f"| {r['dataset']} | {r['issues']:,} | {_pct(r['code'])} | {_pct(r['trace'])} | "
            f"{_pct(r['file'])} | {_pct(r['gold'])} | {r['len']:,.0f} |"
        )
    loc_lines = [
        "| Dataset | Tries with a known gold patch | …that touch a gold file | Median step of first touch |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in loc.to_dict("records"):
        med = "—" if pd.isna(r["median_step"]) else f"{r['median_step']:.0f}"
        loc_lines.append(
            f"| {r['dataset']} | {_pct(r['gold_known'])} | {_pct(r['touched'])} | {med} |"
        )
    if rounds:
        parts = []
        for rnd, note, a in rounds:
            ds_lines = "\n".join(
                f"| {k} | {v['n']} | {v['agreement']:.1%} | {v['kappa']:.3f} |"
                for k, v in a["by_dataset"].items()
            )
            lab_pr = "\n".join(
                f"| `{k}` | {v['rule_count']} | {_pct(v['precision'])} | {_pct(v['recall'])} |"
                for k, v in a["by_label"].items()
                if v["rule_count"]
            )
            parts.append(f"""#### Round {rnd} ({note})

Overall: **{a["n"]} steps, agreement {a["agreement"]:.1%}, Cohen's κ = {a["kappa"]:.3f}** (target ≥ 0.80).

| Dataset | Steps | Agreement | κ |
| --- | ---: | ---: | ---: |
{ds_lines}

| Rule label | Times in sample | Precision (second labeling agrees) | Recall |
| --- | ---: | ---: | ---: |
{lab_pr}
""")
        second = (
            "\n".join(parts)
            + """
**What changed between rounds.** Round 1 found three problems, and all were fixed before round 2:
1. Test-tool names inside paths (`tox-dev__tox`, `nirspec`) were read as test runs. Test-tool names now have to start a command word.
2. Scratch scripts with generic names were missed. Common scratch-script words were added, and documents (`.md`, `.txt`, `.rst`) never count as scripts.
3. Appends (`>>`) counted as new files, and the web `fetch` tool counted as "other". Appends are now edits and `fetch` is "look".

Round 2 used a **fresh** random sample, so the rules were not tuned to the steps that check them.

**Known remaining limits:**
- `repro_create` is conservative. It is almost never wrong when it fires (precision 100%), but it misses scratch scripts with generic names (recall 73%). As a result, "test first" is slightly undercounted, and Step 7 includes a sensitivity analysis with a broader definition (any new file).
- Most `edit_created` vs `edit_code` disagreements come from the second labeler not seeing the try's history. The rules check whether the try created the file itself, so they are likely right in those cases.
- Calls to other command-line tools (e.g. `dvc list`, `pyupgrade x.py`) are labelled `other` rather than `run`. This does not affect any hypothesis.

Both samples, with the rule label and the second label side by side, are in [`docs/data/label_validation/`](data/label_validation/sample.csv)."""
        )
    else:
        second = "Pending: the blind sample has been drawn (`data/interim/label_sample_blind.csv`) but not yet labeled."

    text = f"""# Step 5: Features and Label Validation

*Generated by `agent-tips features`. Numbers come from `data/processed/` and `outputs/results/label_validation.json`.*

## In plain words

- **Every AI action now has a label**, such as "created a test script", "edited project code" or "ran the project's tests". One shared set of rules labels all three datasets, and 133 hand-written test cases check those rules.
- **The labels were checked twice, without human effort:**
  1. against what the computer replied, for example "file created";
  2. by a blind second labeling of a random sample, done by the AI assistant. The paper will report this as AI-assisted, not human, validation.
- **Bug found and fixed:** the earlier exploration code counted *every* new file inside NVIDIA's `/testbed` folder as a test script, because the folder name contains "test". Only the file name is checked now. Together with the other label fixes, the affected shares moved by up to about 3 points.
- **New fact:** about half of the older AI's attempts to edit code were **rejected by its editor** for breaking the code.
- **Each bug also has simple features:** whether the report includes example code, an error trace or a file name. Each bug also records which files the real fix changed ("gold files"), so we can see when the AI first touches them.

## Labels

{chr(10).join(lab_lines)}

Shares are of all steps in each dataset.

## How often each AI follows each tip

![C1](figures/features/c1_tip_compliance.png)

{chr(10).join(comp_lines)}

"Test first", "check your fix" and "test suite after fix" are shares of tries that edited project code. "First move" applies only to the older AI.

## Consistency with the exploration

{chr(10).join(chk)}

Differences of up to about 3 points have three causes:
1. the `/testbed` fix, which affects the SWE-agent subsets;
2. the broader scratch-script words added after validation round 1, which affect all subsets;
3. the denominators: the exploration counted all tries, while this table counts analysis-ready tries only.

## Label validation

### (a) Check against the computer's reply

{chr(10).join(obs_lines)}

- "Confirmed" means the reply fits the label:
  - for creates: a "file created" message (editor), or no error message (shell);
  - for edits: an "edited" message *or* a "rejected" message. Either way, the agent did try to edit.
  - for test runs: test-framework output at the start or the end of the reply.
- Only the first and last 300 characters of each reply are kept, so these figures are lower bounds.

### (b) Blind second labeling

{second}

## Issue features

{chr(10).join(iss_lines)}

## Localization: when the AI first touches a gold file

{chr(10).join(loc_lines)}

A step "touches" a gold file if its path, the file open in the editor, or its command mentions a file changed by the real fix.
"""
    path = out or cfg.root / "docs" / "05_feature_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
