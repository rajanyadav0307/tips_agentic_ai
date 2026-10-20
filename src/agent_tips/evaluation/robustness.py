"""Step 7: robustness and sensitivity checks for the primary results.

- **P1 (older AI):** model-version batches defined five ways; bounds and
  imputation for the undated tries (all unresolved); a within-issue-only
  version.
- **P2 (newer AIs):** leave one AI out; the broader "any new file" definition
  of test first; moving tries the labeler may have misread; and a
  tipping point for NVIDIA's removed tries (how many hidden failures among
  test-first tries would turn the result into meaningful harm), checked
  against how often the unfiltered Nebius data would have lost such tries.
- **Reproducibility:** run the whole analysis twice and compare the results.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_tips.config import Config
from agent_tips.models import analysis
from agent_tips.models.estimators import build_strata, fair_estimate, weighted_diff
from agent_tips.models.regression import Coef, conditional_logit

OLDER, NEBIUS, NVIDIA = analysis.OLDER, analysis.NEBIUS, analysis.NVIDIA
BATCHES = {
    "day": "batch_day",
    "6 hours": "batch_6h",
    "hour": "batch_hour",
    "gap > 60 min": "batch_gap60",
    "gap > 10 min": "batch_gap10",
}


def undated_scenarios(
    older: pd.DataFrame, key: str, rng: np.random.Generator, reps: int = 200
) -> dict[str, float]:
    """Add back the undated (always unresolved) R/S tries under several assumptions."""
    base = older[older.first_move_arm.isin(["R", "S"])]
    dated, undated = base[base.dated], base[~base.dated]
    g = (
        dated.groupby(["issue_key", key, "first_move_arm"])
        .y.agg(["size", "sum"])
        .unstack("first_move_arm", fill_value=0)
        .reset_index()
    )
    inst = g["issue_key"].to_numpy()
    n_r0, n_s0 = g[("size", "R")].to_numpy(float), g[("size", "S")].to_numpy(float)
    k_r0, k_s0 = g[("sum", "R")].to_numpy(float), g[("sum", "S")].to_numpy(float)
    extra = np.unique(undated.issue_key)
    inst = np.concatenate([inst, extra])
    pad = np.zeros(len(extra))
    n_r0, n_s0, k_r0, k_s0 = (np.concatenate([x, pad]) for x in (n_r0, n_s0, k_r0, k_s0))
    is_new = np.concatenate([np.zeros(len(g), bool), np.ones(len(extra), bool)])
    by_inst = pd.Series(np.arange(len(inst))).groupby(inst).apply(np.array).to_dict()
    counts = undated.groupby(["issue_key", "first_move_arm"]).size().reset_index(name="m")
    todo = [(str(i), str(a), int(m)) for i, a, m in counts.itertuples(index=False)]

    def est(n_r: np.ndarray, n_s: np.ndarray) -> float:
        ok = (n_r > 0) & (n_s > 0)
        w = n_r[ok] * n_s[ok] / (n_r[ok] + n_s[ok])
        return float(np.sum(w * (k_r0[ok] / n_r[ok] - k_s0[ok] / n_s[ok])) / np.sum(w))

    out = {"left out (main analysis)": est(n_r0, n_s0)}
    fail_r0, fail_s0 = n_r0 - k_r0, n_s0 - k_s0
    draws = []
    for _ in range(reps):
        n_r, n_s = n_r0.copy(), n_s0.copy()
        for iid, arm, m in todo:
            idx = by_inst[iid]
            old = idx[~is_new[idx]]
            if len(old) == 0:
                cand, p = idx[is_new[idx]], np.array([1.0])
            else:
                fails = (fail_r0 if arm == "R" else fail_s0)[old]
                if fails.sum() == 0:
                    fails = fail_r0[old] + fail_s0[old]
                if fails.sum() == 0:
                    fails = np.ones(len(old))
                cand, p = old, fails / fails.sum()
            add = rng.multinomial(m, p)
            (n_r if arm == "R" else n_s)[cand] += add
        draws.append(est(n_r, n_s))
    out["added back like other failures (median of draws)"] = float(np.median(draws))
    runs = [(iid, arm) for iid, arm, m in todo for _ in range(m)]
    order = rng.permutation(len(runs))
    for label, pick in (
        ("added back, worst case for the finding", np.argmax),
        ("added back, best case for the finding", np.argmin),
    ):
        n_r, n_s = n_r0.copy(), n_s0.copy()

        def contrib(j: int, n_r: np.ndarray = n_r, n_s: np.ndarray = n_s) -> tuple[float, float]:
            if n_r[j] > 0 and n_s[j] > 0:
                w = n_r[j] * n_s[j] / (n_r[j] + n_s[j])
                return w * (k_r0[j] / n_r[j] - k_s0[j] / n_s[j]), w
            return 0.0, 0.0

        num = den = 0.0
        for j in range(len(n_r)):
            c, w = contrib(j)
            num, den = num + c, den + w
        for o in order:
            iid, arm = runs[o]
            arr = n_r if arm == "R" else n_s
            vals = []
            for j in by_inst[iid]:
                c0, w0 = contrib(j)
                arr[j] += 1
                c1, w1 = contrib(j)
                arr[j] -= 1
                vals.append(((num - c0 + c1) / (den - w0 + w1), j, c0, w0, c1, w1))
            best = vals[int(pick([v[0] for v in vals]))]
            arr[best[1]] += 1
            num, den = num - best[2] + best[4], den - best[3] + best[5]
        out[label] = num / den
    return out


def tipping_point(
    nv: pd.DataFrame, harm: float = -0.02, max_extra: float = 2.0
) -> dict[str, float]:
    """Hidden failures among test-first tries needed to reach a meaningful harm.

    NVIDIA removed tries that hit the step limit or produced no patch; these
    are failures we cannot see. Suppose test-first tries lost extra failures
    in proportion ``e`` to their observed count in every issue. Find the
    smallest ``e`` at which the fair estimate reaches ``harm`` (default −2 points).
    """
    st = build_strata(nv, [], "order_arm", "test first", "fix first", "y")

    def est(e: float) -> float:
        n_a = st.n_a * (1 + e)
        w = n_a * st.n_b / (n_a + st.n_b)
        return float(np.sum(w * (st.k_a / n_a - st.k_b / st.n_b)) / np.sum(w))

    lo, hi = 0.0, max_extra
    if est(hi) > harm:
        return {"extra_failure_share": float("nan"), "estimate_at_zero": est(0.0)}
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if est(mid) > harm else (lo, mid)
    return {"extra_failure_share": hi, "estimate_at_zero": est(0.0), "harm_threshold": harm}


def misclassification(
    nv: pd.DataFrame,
    rng: np.random.Generator,
    shares: tuple[float, ...] = (0.25, 0.5, 1.0),
    draws: int = 50,
) -> dict[str, dict[str, float]]:
    """Move "fix first" tries the labeler may have got wrong into "test first".

    The labeler never wrongly calls a file a test script (precision 100%) but
    misses some (recall about 73%). A miss can only change a try's group if
    the try made a new file before its fix and no detected test script, i.e.
    the tries that only the broad definition calls "test first". A random
    ``share`` of those is moved and the fair estimate recomputed.
    """
    cand = np.flatnonzero(
        ((nv.order_arm == "fix first") & (nv.order_arm_broad == "test first")).to_numpy()
    )
    out = {}
    for share in shares:
        ests = []
        for _ in range(draws if share < 1 else 1):
            arm = nv.order_arm.to_numpy().copy()
            arm[rng.choice(cand, int(round(share * len(cand))), replace=False)] = "test first"
            st = build_strata(
                nv.assign(order_arm=arm), [], "order_arm", "test first", "fix first", "y"
            )
            ests.append(weighted_diff(st))
        out[f"{share:.0%} of {len(cand)} moved"] = {
            "median": float(np.median(ests)),
            "min": float(np.min(ests)),
            "max": float(np.max(ests)),
        }
    return out


def _odds(c: Coef | None) -> dict[str, float]:
    if c is None:
        return {}
    return {
        "odds_ratio": float(np.exp(c.estimate)),
        "ci_lo": float(np.exp(c.ci_lo)),
        "ci_hi": float(np.exp(c.ci_hi)),
        "p_value": c.p_value,
    }


def frame_hash(df: pd.DataFrame) -> str:
    """Stable hash of a results table (values rounded to 12 digits)."""
    d = df.copy()
    for c in d.select_dtypes("number").columns:
        d[c] = d[c].round(12)
    return hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()


def reproducibility(cfg: Config, log: Callable[[str], None]) -> dict[str, Any]:
    """Run the whole analysis twice into separate folders and compare the results."""
    hashes = []
    for i in (1, 2):
        raw = copy.deepcopy(cfg.raw)
        raw["paths"]["results"] = f"outputs/repro_check/run{i}"
        c = Config(root=cfg.root, raw=raw)
        analysis.run(c, log=lambda _: None)
        hashes.append(frame_hash(pd.read_parquet(c.path("results") / "estimates.parquet")))
        log(f"reproducibility run {i}: {hashes[-1][:16]}")
    return {"hash_run1": hashes[0], "hash_run2": hashes[1], "identical": hashes[0] == hashes[1]}


def run(cfg: Config, log: Callable[[str], None] = print, repro: bool = True) -> Path:
    """Run every robustness check and write ``outputs/results/robustness.json``."""
    reps = int(cfg.raw["analysis"]["bootstrap_reps"])
    rng = np.random.default_rng(cfg.seed + 7)
    df = analysis.load(cfg)
    older = df[df.dataset == OLDER]
    dated = older[older.dated]
    nv = df[df.dataset == NVIDIA]
    out: dict[str, Any] = {}

    log("P1: batch definitions")
    out["p1_batches"] = {
        name: fair_estimate(dated, [col], "first_move_arm", "R", "S", "y", rng, reps).as_dict()
        for name, col in BATCHES.items()
    }
    out["p1_within_issue_only"] = fair_estimate(
        older, [], "first_move_arm", "R", "S", "y", rng, reps
    ).as_dict()
    log("conditional logistic regression (odds-ratio cross-check)")
    out["conditional_logit"] = {
        name: _odds(conditional_logit(d, st, arm, a, b, "y"))
        for name, d, st, arm, a, b in (
            ("P1", dated, ["batch_day"], "first_move_arm", "R", "S"),
            ("P2", nv, [], "order_arm", "test first", "fix first"),
        )
    }
    log("P1: undated tries")
    out["p1_undated"] = {
        name: undated_scenarios(older, col, rng)
        for name, col in (("day", "batch_day"), ("hour", "batch_hour"))
    }

    log("P2: leave one AI out, broad definition, misclassification, tipping point")
    out["p2_leave_one_out"] = {
        sub: fair_estimate(
            nv[nv.subset != sub], [], "order_arm", "test first", "fix first", "y", rng, reps
        ).as_dict()
        for sub in sorted(nv.subset.unique())
    }
    out["p2_broad_definition"] = fair_estimate(
        nv, [], "order_arm_broad", "test first", "fix first", "y", rng, reps
    ).as_dict()
    out["p2_misclassification"] = misclassification(nv, rng)
    out["p2_tipping_point"] = tipping_point(nv)
    # Calibration from the unfiltered Nebius newer-AI data: how much more often
    # would test-first tries have been removed by NVIDIA-style filtering?
    neb = df[df.dataset == NEBIUS].assign(
        removed=lambda d: (
            d.exit_status.str.contains("maximum iteration", na=False) | ~d.has_patch
        ).astype(float)
    )
    out["p2_removal_calibration"] = fair_estimate(
        neb, [], "order_arm", "test first", "fix first", "removed", rng, reps
    ).as_dict()
    if repro:
        log("reproducibility (two full runs)")
        out["reproducibility"] = reproducibility(cfg, log)
    path = cfg.path("results") / "robustness.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    log(f"wrote {path}")
    return path
