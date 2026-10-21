"""Step 7 (one-try study): the secondary scores of amendment A8.

None of these change H1-H4, the margin, the main score, the comparisons or the
answer keys. They help read the results:

- **aligned answer keys**: each method is also scored against the answer key
  that weights bugs the way the method does (:func:`aligned_keys`);
- **error floor**: how far a perfect method would still land from the answer
  key by chance, from a *joint* bootstrap of bugs that redraws the answer key
  and the one-try estimates together (they share tries, so their noise is
  shared) (:func:`floors`); with it, the **noise-corrected bias**;
- **paired contrasts** between methods on the same comparisons;
- **few-cluster inference**: per-setup t intervals (8 degrees of freedom with 9
  setups), a Webb six-point wild cluster bootstrap and leave-one-setup-out;
- **interval quality**: width, interval score, detection and false alarms
  (coverage alone rewards wide intervals);
- **model diagnostics**: how well the two guessers guess (R² of the fix
  model, AUC of the choice model, share of the choice left over);
- **sensitivity**: only comparisons whose answer key is precise (noise floor at
  most 1.5 points);
- **all-bugs details**: the share of the ML weight on non-key bugs, trimmed
  versions, and how often gradient boosting's automatic early stopping was on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from agent_tips.config import Config
from agent_tips.onetry import data
from agent_tips.onetry.simulate import aligned_key, key_parts, key_rows, seed_for

DML_METHODS = ["rung2", "rung3", "rung4", "rung5", "rung6", "rung5_real", "rung7"]
AIPW_METHODS = ["rung2_aipw", "rung4_aipw", "rung5_aipw"]
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
PRECISE = 0.015  # sensitivity: answer keys with a noise floor of at most 1.5 points
EARLY_STOP_ROWS = 10_000  # scikit-learn turns early stopping on above this many rows


def target_of(method: str) -> str:
    """Which aligned answer key a method is scored against (``dml``, ``ate`` or ``cmh``)."""
    if method in AIPW_METHODS or method.endswith("_aipw"):
        return "ate"
    if method in DML_METHODS or method.startswith("rung5_k") or method == "rung5_forest":
        return "dml"
    return "cmh"


# ---------------------------------------------------------------- aligned keys


def aligned_keys(s: data.StudyData, reps: int, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Compute the answer key of every comparison under each bug weighting, with 95% bounds."""
    rows = []
    for c in s.comparisons:
        row: dict[str, Any] = {"cid": c.cid}
        for w in ("dml", "ate"):
            est, lo, hi = aligned_key(c, s.tries, w, reps)
            row |= {f"{w}_estimate": est, f"{w}_ci_lo": lo, f"{w}_ci_hi": hi}
        rows.append(row)
    log(f"aligned answer keys: {len(rows)} comparisons")
    return pd.DataFrame(rows)


def aligned_scores(sc_units: pd.DataFrame, keys: pd.DataFrame, ak: pd.DataFrame) -> pd.DataFrame:
    """Error of every (comparison, method) against its aligned answer key."""
    k = keys.set_index("cid").estimate.rename("cmh_estimate")
    a = ak.set_index("cid")[["dml_estimate", "ate_estimate"]].join(k)
    g = sc_units.groupby(["cid", "method"]).estimate.mean().reset_index()
    g["target"] = g.method.map(target_of)
    g = g.join(a, on="cid")
    g["key"] = np.select(
        [g.target == "dml", g.target == "ate"], [g.dml_estimate, g.ate_estimate], g.cmh_estimate
    )
    g["aligned_error"] = g.estimate - g.key
    return g[["cid", "method", "target", "aligned_error"]]


# ---------------------------------------------------------------- joint bootstrap


def _left(cfg: Config, cid: str, slug: str, draws: int) -> pd.DataFrame:
    folder = cfg.path("onetry_results") / "leftovers" / slug
    files = [folder / f"key-{d:02d}.parquet" for d in range(draws)]
    return pd.concat([pd.read_parquet(f) for f in files if f.is_file()], ignore_index=True)


def floors(
    cfg: Config,
    s: data.StudyData,
    slug: Callable[[str], str],
    draws: int,
    reps: int = 1000,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joint-bootstrap error floors and model diagnostics, per comparison × method.

    Every rung is a slope Σ r_y r_t / Σ r_t² of per-bug leftovers (AIPW: a mean of
    per-bug scores, saved with r_t = 1), so resampling bugs redraws each estimate
    without refitting, together with the answer key on the same resampled bugs.
    """
    out, diag = [], []
    for c in s.comparisons:
        left = _left(cfg, c.cid, slug(c.cid), draws)
        parts = {w: key_parts(c, s.tries, w).set_index("bug_id") for w in ("cmh", "dml", "ate")}
        bugs = parts["cmh"].index
        b = len(bugs)
        rng = np.random.default_rng(seed_for(c.cid, 0, 70))
        counts = np.stack([np.bincount(rng.integers(0, b, b), minlength=b) for _ in range(reps)])
        counts = counts.astype(float)
        key_star = {
            w: (counts @ p.num.to_numpy()) / (counts @ p.den.to_numpy()) for w, p in parts.items()
        }
        key_hat = {w: float(p.num.sum() / p.den.sum()) for w, p in parts.items()}
        base = left[left.method == "rung0"].set_index(["draw", "bug_id"])
        for m, g in left.groupby("method"):
            wide_ry = g.pivot(index="bug_id", columns="draw", values="ry").reindex(bugs)
            wide_rt = g.pivot(index="bug_id", columns="draw", values="rt").reindex(bugs)
            num = (wide_ry * wide_rt).fillna(0).to_numpy(float)
            den = (wide_rt * wide_rt).fillna(0).to_numpy(float)
            theta = num.sum(0) / den.sum(0)
            theta_star = (counts @ num) / (counts @ den)
            row = {"cid": c.cid, "method": str(m), "draws": num.shape[1]}
            for w in ("cmh", target_of(str(m))):
                e_star = theta_star.mean(1) - key_star[w]
                row[f"floor_{w}"] = float(np.std(e_star, ddof=1))
                row[f"error_{w}"] = float(theta.mean() - key_hat[w])
            out.append(row)
            if m in ("rung0",) or str(m).endswith("_aipw"):
                continue
            j = g.set_index(["draw", "bug_id"]).join(base, rsuffix="0")
            t = (j.rt0 > 0).astype(int)
            per = []
            for _d, jd in j.assign(t=t).groupby(level="draw"):
                auc = roc_auc_score(jd.t, jd.t - jd.rt) if jd.t.nunique() == 2 else np.nan
                per.append(
                    (
                        1 - float((jd.ry**2).sum() / (jd.ry0**2).sum()),
                        float(auc),
                        float((jd.rt**2).sum() / (jd.rt0**2).sum()),
                    )
                )
            r2, auc, share = np.nanmean(np.asarray(per), axis=0)
            diag.append({"cid": c.cid, "method": str(m), "r2_y": r2, "auc_t": auc, "t_left": share})
    log(f"error floors: {len(out)} comparison × method pairs")
    return pd.DataFrame(out), pd.DataFrame(diag)


def bias_table(fl: pd.DataFrame, methods: Sequence[str], key: str = "cmh") -> pd.DataFrame:
    """Noise-corrected bias per method: RMS error with the error floor taken out."""
    rows = []
    for m in methods:
        g = fl[fl.method == m]
        if g.empty:
            continue
        e2, v = g[f"error_{key}"] ** 2, g[f"floor_{key}"] ** 2
        rows.append(
            {
                "method": m,
                "comparisons": len(g),
                "rms_error": float(np.sqrt(e2.mean())),
                "median_floor": float(g[f"floor_{key}"].median()),
                "bias_rms": float(np.sqrt(max(float((e2 - v).mean()), 0.0))),
                "clearly_off": float((g[f"error_{key}"].abs() > 1.96 * g[f"floor_{key}"]).mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- few clusters


def cluster_ci(x: pd.Series, setup: pd.Series, reps: int = 9999, seed: int = 0) -> dict[str, float]:
    """Mean of ``x`` over comparisons with three few-cluster intervals (95%).

    - ``t``: per-setup means, t interval with (setups − 1) degrees of freedom;
    - ``wild``: Webb six-point wild cluster bootstrap of the overall mean;
    - ``loso``: smallest and largest mean when one setup is left out.
    """
    x, setup = x.reset_index(drop=True), setup.reset_index(drop=True)
    mean = float(x.mean())
    per = x.groupby(setup).mean()
    g = len(per)
    half = float(stats.t.ppf(0.975, g - 1) * per.std(ddof=1) / np.sqrt(g)) if g > 1 else np.nan
    u = (x - mean).groupby(setup).sum().to_numpy()
    rng = np.random.default_rng(seed)
    v = WEBB[rng.integers(0, 6, size=(reps, g))]
    dev = (v @ u) / len(x)
    lo_w, hi_w = mean - np.percentile(dev, 97.5), mean - np.percentile(dev, 2.5)
    loso = [float(x[setup != k].mean()) for k in per.index] if g > 1 else [mean]
    return {
        "mean": mean,
        "setups": g,
        "setup_mean": float(per.mean()),
        "t_lo": float(per.mean() - half),
        "t_hi": float(per.mean() + half),
        "wild_lo": float(lo_w),
        "wild_hi": float(hi_w),
        "loso_min": min(loso),
        "loso_max": max(loso),
    }


def contrasts(
    sc: pd.DataFrame, pairs: Sequence[tuple[str, str]], reps: int, seed: int
) -> pd.DataFrame:
    """Paired contrasts |error A| − |error B| on the comparisons both methods have."""
    rows = []
    for a, b in pairs:
        w = sc[sc.method.isin([a, b])].pivot_table(
            index=["cid", "setup"], columns="method", values="abs_error"
        )
        if a not in w or b not in w:
            continue
        w = w.dropna().reset_index()
        if len(w) < 2:
            continue
        d = w[a] - w[b]
        rows.append({"a": a, "b": b, "comparisons": len(w), **cluster_ci(d, w.setup, reps, seed)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- intervals


def interval_quality(
    units: pd.DataFrame, keys: pd.DataFrame, methods: Sequence[str]
) -> pd.DataFrame:
    """Width, interval score, detection and false alarms per method (draws with an error bar)."""
    k = keys.set_index("cid")[["estimate", "conclusion"]].rename(columns=lambda c: f"key_{c}")
    d = units[units.method.isin(methods) & units.se.notna()].join(k, on="cid")
    lo, hi = d.estimate - 1.96 * d.se, d.estimate + 1.96 * d.se
    concl = np.where(lo > 0, 1, np.where(hi < 0, -1, 0))
    alpha = 0.05
    miss = np.maximum(lo - d.key_estimate, 0) + np.maximum(d.key_estimate - hi, 0)
    d = d.assign(
        width=hi - lo,
        score=(hi - lo) + 2 / alpha * miss,
        detect=np.where(d.key_conclusion != 0, concl == d.key_conclusion, np.nan),
        false_alarm=np.where(d.key_conclusion == 0, concl != 0, np.nan),
    )
    per = d.groupby(["method", "cid"])[["width", "score", "detect", "false_alarm"]].mean()
    return per.groupby("method").mean().reset_index()


# ---------------------------------------------------------------- all bugs


def allbugs_details(
    cfg: Config,
    s: data.StudyData,
    units: pd.DataFrame,
    slug: Callable[[str], str],
    draws: int,
) -> dict[str, Any]:
    """Weight share on non-key bugs (ML rungs) and early-stopping counts (all-bugs copies)."""
    rows = []
    for c in s.comparisons:
        folder = cfg.path("onetry_results") / "leftovers" / slug(c.cid)
        files = [folder / f"all-{d:02d}.parquet" for d in range(draws)]
        left = [pd.read_parquet(f) for f in files if f.is_file()]
        if not left:
            continue
        lf = pd.concat(left)
        key_bugs = set(key_rows(c, s.tries).bug_id)
        lf["key"] = lf.bug_id.isin(key_bugs)
        for m, g in lf.groupby("method"):
            w = g.rt.astype(float) ** 2
            rows.append(
                {"cid": c.cid, "method": m, "nonkey_weight": float(w[~g.key].sum() / w.sum())}
            )
    share = pd.DataFrame(rows)
    allb = units[units.method.str.contains("_allbugs")]
    ml = allb[allb.method.isin(["rung2_allbugs", "rung5_allbugs", "rung5_aipw_allbugs"])]
    return {
        "nonkey_weight": share.groupby("method").nonkey_weight.mean().to_dict()
        if len(share)
        else {},
        "early_stopping_copies": int((ml.n > EARLY_STOP_ROWS).sum()),
        "ml_copies": len(ml),
    }


def write_parquets(res: Path, **frames: pd.DataFrame) -> None:
    """Save the A8 tables next to the other results."""
    for name, f in frames.items():
        f.to_parquet(res / f"a8_{name}.parquet", index=False)
