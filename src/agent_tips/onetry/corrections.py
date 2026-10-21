"""Step 6 (one-try study): the correction ladder.

Every method receives one **one-try copy**: one row per bug with the outcome
``y`` (fixed or not), the choice ``t`` (1 = choice *a*) and the bug's features.
It returns an estimate of the effect of *a* vs *b* with a standard error.
Settings are frozen in ``docs/one_try/01_problem.md`` §5.

=====  =========================================================
Rung   Method
=====  =========================================================
0      simple difference in fix rates
1      linear regression on report facts (HC1 errors)
2      double machine learning (DML), report facts + report text
3      DML, + language-model embeddings (exploratory)
4      DML, + human-fix size
5      DML, + other AIs' results on the bug
6      DML, + IRT difficulty from other AIs
=====  =========================================================

DML ("partialling out"): two gradient-boosting models predict ``y`` and ``t``
from the features, each trained on 4/5 of the bugs and used on the remaining
1/5 (cross-fitting). The effect is the slope of the leftover ``y`` on the
leftover ``t``: θ = Σ r_y r_t / Σ r_t².
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import KFold

from agent_tips.onetry.features import TextFeatures

FACT_NUMERIC = ["log_report_chars", "has_code_block", "has_stack_trace", "names_file"]
FACT_CATEGORICAL = ["repo_group", "language", "task_type"]
FIX = ["log_fix_files", "log_fix_lines", "fix_missing"]
OTHER = ["other_rate", "n_other", "other_missing"]
IRT = ["irt_difficulty", "irt_missing"]
RUNGS: dict[int, dict[str, object]] = {
    2: {"text": True, "emb": False, "extra": []},
    3: {"text": True, "emb": True, "extra": []},
    4: {"text": True, "emb": False, "extra": FIX},
    5: {"text": True, "emb": False, "extra": FIX + OTHER},
    6: {"text": True, "emb": False, "extra": FIX + IRT},
}
Z95 = 1.959963984540054


@dataclass(frozen=True)
class Result:
    """One estimate with its standard error and 95% interval."""

    estimate: float
    se: float

    @property
    def ci_lo(self) -> float:
        """Lower end of the 95% interval."""
        return self.estimate - Z95 * self.se

    @property
    def ci_hi(self) -> float:
        """Upper end of the 95% interval."""
        return self.estimate + Z95 * self.se


# ---------------------------------------------------------------- learners


def gbm() -> HistGradientBoostingRegressor:
    """Return the main nuisance learner (frozen settings)."""
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)


def forest() -> RandomForestRegressor:
    """Return the check learner (frozen settings; n_jobs only changes speed)."""
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=0, n_jobs=2)


LEARNERS = {"gbm": gbm, "forest": forest}


# ---------------------------------------------------------------- features


def facts_matrix(d: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Report facts with categories one-hot encoded (fixed column set if given)."""
    x = pd.get_dummies(d[FACT_NUMERIC + FACT_CATEGORICAL], columns=FACT_CATEGORICAL, dtype=float)
    if columns is not None:
        x = x.reindex(columns=list(columns), fill_value=0.0)
    return x.astype(float)


def _fold_features(
    d: pd.DataFrame, train: np.ndarray, test: np.ndarray, rung: int, text: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrices of one fold for one rung (text model fitted on ``train`` only)."""
    spec = RUNGS[rung]
    facts = facts_matrix(d)
    parts_tr = [facts.to_numpy()[train]]
    parts_te = [facts.to_numpy()[test]]
    if spec["text"]:
        parts_tr.append(text["train"])
        parts_te.append(text["test"])
    if spec["emb"]:
        emb = d.filter(like="emb_").to_numpy(float)
        parts_tr.append(emb[train])
        parts_te.append(emb[test])
    extra = list(spec["extra"])  # type: ignore[call-overload]
    if extra:
        e = d[extra].to_numpy(float)
        parts_tr.append(e[train])
        parts_te.append(e[test])
    return np.hstack(parts_tr), np.hstack(parts_te)


def text_folds(
    d: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    dims: int,
    seed: int,
    tokens: list[list[str]] | None = None,
) -> list[dict[str, np.ndarray]]:
    """Report-text numbers per fold, each fitted on that fold's training bugs only.

    ``tokens`` (from :func:`agent_tips.onetry.features.tokenize`, aligned with the
    rows of ``d``) skips the slow word-splitting; the numbers are identical.
    """
    out = []
    for train, test in folds:
        if tokens is not None:
            tf = TextFeatures(dims=dims, seed=seed, pretokenized=True)
            tr = [tokens[i] for i in train]
            te = [tokens[i] for i in test]
            tf.fit(tr)
            out.append({"train": tf.transform(tr), "test": tf.transform(te)})
        else:
            tf = TextFeatures(dims=dims, seed=seed).fit(d.report.iloc[train])
            out.append(
                {
                    "train": tf.transform(d.report.iloc[train]),
                    "test": tf.transform(d.report.iloc[test]),
                }
            )
    return out


def make_folds(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-fitting folds; rows are bugs, so the split is by bug."""
    return list(KFold(n_splits=min(k, n), shuffle=True, random_state=seed).split(np.zeros(n)))


# ---------------------------------------------------------------- rungs 0 and 1


def simple(d: pd.DataFrame) -> Result:
    """Rung 0: difference in fix rates between the two choices."""
    a, b = d[d.t == 1].y, d[d.t == 0].y
    se = np.sqrt(a.var(ddof=0) / max(len(a), 1) + b.var(ddof=0) / max(len(b), 1))
    return Result(float(a.mean() - b.mean()), float(se))


def linear(d: pd.DataFrame) -> Result:
    """Rung 1: ordinary least squares y ~ t + report facts, HC1 standard error of t."""
    x = facts_matrix(d)
    x = x.loc[:, x.std(ddof=0) > 0]
    xm = np.column_stack([np.ones(len(d)), d.t.to_numpy(float), x.to_numpy()])
    y = d.y.to_numpy(float)
    coef, *_ = np.linalg.lstsq(xm, y, rcond=None)
    resid = y - xm @ coef
    n, p = xm.shape
    bread = np.linalg.pinv(xm.T @ xm)
    meat = (xm * resid[:, None]).T @ (xm * resid[:, None])
    cov = bread @ meat @ bread * n / max(n - p, 1)
    return Result(float(coef[1]), float(np.sqrt(max(cov[1, 1], 0.0))))


# ---------------------------------------------------------------- DML rungs


def dml(
    d: pd.DataFrame,
    rung: int,
    seed: int,
    folds_k: int = 5,
    learner: str = "gbm",
    text: list[dict[str, np.ndarray]] | None = None,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
    text_dims: int = 100,
    tokens: list[list[str]] | None = None,
) -> Result:
    """Rungs 2-6: double machine learning by partialling out."""
    folds = folds or make_folds(len(d), folds_k, seed)
    text = text or text_folds(d, folds, text_dims, seed, tokens)
    y, t = d.y.to_numpy(float), d.t.to_numpy(float)
    ry, rt = np.zeros(len(d)), np.zeros(len(d))
    for (train, test), tx in zip(folds, text, strict=True):
        xtr, xte = _fold_features(d, train, test, rung, tx)
        for target, out in ((y, ry), (t, rt)):
            m = LEARNERS[learner]()
            m.fit(xtr, target[train])
            out[test] = target[test] - m.predict(xte)
    return _partial_out(ry, rt)


def _partial_out(ry: np.ndarray, rt: np.ndarray) -> Result:
    j = float(np.mean(rt * rt))
    if j <= 0:
        return Result(float("nan"), float("nan"))
    theta = float(np.sum(ry * rt) / np.sum(rt * rt))
    psi = (ry - theta * rt) * rt / j
    return Result(theta, float(np.sqrt(np.mean(psi * psi) / len(ry))))


def aipw(
    d: pd.DataFrame,
    rung: int,
    seed: int,
    folds_k: int = 5,
    text: list[dict[str, np.ndarray]] | None = None,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
    text_dims: int = 100,
    tokens: list[list[str]] | None = None,
) -> Result:
    """Check: doubly robust (AIPW) estimate with a clipped propensity model."""
    folds = folds or make_folds(len(d), folds_k, seed)
    text = text or text_folds(d, folds, text_dims, seed, tokens)
    y, t = d.y.to_numpy(float), d.t.to_numpy(int)
    psi = np.zeros(len(d))
    for (train, test), tx in zip(folds, text, strict=True):
        xtr, xte = _fold_features(d, train, test, rung, tx)
        if len(np.unique(t[train])) < 2:
            return Result(float("nan"), float("nan"))
        e = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
        e.fit(xtr, t[train])
        p = np.clip(e.predict_proba(xte)[:, 1], 0.02, 0.98)
        mu = {}
        for arm in (0, 1):
            m = gbm()
            m.fit(xtr[t[train] == arm], y[train][t[train] == arm])
            mu[arm] = m.predict(xte)
        tt, yy = t[test], y[test]
        psi[test] = mu[1] - mu[0] + tt * (yy - mu[1]) / p - (1 - tt) * (yy - mu[0]) / (1 - p)
    return Result(float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi))))


def ladder(
    d: pd.DataFrame,
    seed: int,
    rungs: Sequence[int] = (0, 1, 2, 4, 5, 6),
    folds_k: int = 5,
    text_dims: int = 100,
    checks: bool = False,
    tokens: list[list[str]] | None = None,
) -> dict[str, Result]:
    """Run the requested rungs on one one-try copy (text model shared across rungs)."""
    out: dict[str, Result] = {}
    if 0 in rungs:
        out["rung0"] = simple(d)
    if 1 in rungs:
        out["rung1"] = linear(d)
    dml_rungs = [r for r in rungs if r >= 2]
    if dml_rungs:
        folds = make_folds(len(d), folds_k, seed)
        text = text_folds(d, folds, text_dims, seed, tokens)
        for r in dml_rungs:
            out[f"rung{r}"] = dml(d, r, seed, folds=folds, text=text)
        if checks:
            for r in (2, 4, 5):
                if r in dml_rungs:
                    out[f"rung{r}_forest"] = dml(
                        d, r, seed, learner="forest", folds=folds, text=text
                    )
                    out[f"rung{r}_aipw"] = aipw(d, r, seed, folds=folds, text=text)
    return out
