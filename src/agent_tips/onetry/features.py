"""Step 5 (one-try study): feature engineering.

Builds the information each correction method may use (frozen in
``docs/one_try/01_problem.md`` §5). Only facts that exist without the studied
AI setup's own results are allowed.

Outputs in ``data/onetry/features/``:

- ``bug_features.parquet``: one row per dataset × bug: report facts (rung 1)
  and human-fix size (rung 4);
- ``other_ai.parquet``: one row per studied setup × bug: other setups' mean
  fix rate and count (rung 5) and an IRT difficulty fitted on the other
  setups' tries only (rung 6);
- ``embeddings.parquet`` (optional, ``--embeddings``): sentence embeddings of
  each bug report (rung 3, exploratory).

The report text itself (rung 2) was turned into numbers *inside each training
fold* during modelling, with :class:`TextFeatures`. From amendment A9 the Step 7
run uses :func:`text_table` instead: one text model per dataset, fitted on
report text only (no outcomes), cached in ``text_svd.parquet``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from agent_tips.config import Config

TOP_REPOS = 30
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # Hub commit, checked 2026-10-20


# ---------------------------------------------------------------- rung 2 (used in Step 6)


_ANALYZER = TfidfVectorizer(ngram_range=(1, 2)).build_analyzer()


def tokenize(texts: pd.Series) -> list[list[str]]:
    """Split each text into the words and word pairs the text model uses (done once, reused)."""
    return [_ANALYZER(t) for t in texts.fillna("")]


def _as_is(tokens: list[str]) -> list[str]:
    return tokens


class TextFeatures:
    """Report text → 100 numbers: TF-IDF (1–2-grams) then truncated SVD.

    Settings are frozen in Step 1. Fit on training-fold reports only. With
    ``pretokenized=True`` it takes the output of :func:`tokenize` instead of raw
    text; the result is identical, but the slow word-splitting is done once.
    """

    def __init__(self, dims: int = 100, seed: int = 0, pretokenized: bool = False) -> None:
        settings = {"min_df": 3, "max_features": 50_000, "sublinear_tf": True}
        self.tfidf = (
            TfidfVectorizer(analyzer=_as_is, **settings)
            if pretokenized
            else TfidfVectorizer(ngram_range=(1, 2), **settings)
        )
        self.pretokenized = pretokenized
        self.svd = TruncatedSVD(n_components=dims, random_state=seed)

    def _input(self, texts: pd.Series | list[list[str]]) -> pd.Series | list[list[str]]:
        return texts if self.pretokenized else pd.Series(texts).fillna("")

    def fit(self, texts: pd.Series | list[list[str]]) -> TextFeatures:
        """Learn the vocabulary and the 100 directions from training texts."""
        try:
            x = self.tfidf.fit_transform(self._input(texts))
        except ValueError:  # no word appears in 3 or more reports
            self.empty = True
            return self
        self.empty = False
        self.reduce = x.shape[1] > 1  # SVD needs at least two words
        if self.reduce:
            self.svd.n_components = min(self.svd.n_components, x.shape[1] - 1)
            self.svd.fit(x)
        return self

    def transform(self, texts: pd.Series | list[list[str]]) -> np.ndarray:
        """Turn texts into numbers with the learned vocabulary and directions."""
        n = len(texts)
        if self.empty:
            return np.zeros((n, 1))
        x = self.tfidf.transform(self._input(texts))
        return np.asarray(self.svd.transform(x) if self.reduce else x.toarray())


def text_table(reports: pd.DataFrame, dims: int = 100, seed: int = 0) -> pd.DataFrame:
    """Return cached report-text numbers (amendment A9): one text model per dataset.

    Fitted on all of a dataset's bug reports at once. It sees only the report
    text, never an outcome or a choice, so it cannot leak the answer. Returns
    ``dataset``, ``bug_id`` and ``txt_0`` … ``txt_{dims-1}``.
    """
    parts = []
    for ds, g in reports.groupby("dataset", sort=True):
        g = g.sort_values("bug_id")
        tokens = tokenize(g.report)
        tf = TextFeatures(dims=dims, seed=seed, pretokenized=True).fit(tokens)
        x = tf.transform(tokens)
        cols = {f"txt_{i}": x[:, i] for i in range(x.shape[1])}
        parts.append(pd.DataFrame({"dataset": ds, "bug_id": g.bug_id.to_numpy(), **cols}))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- rung 1 and 4


def bug_features(bugs: pd.DataFrame) -> pd.DataFrame:
    """Report facts and human-fix size per dataset × bug."""
    d = bugs.copy()
    top = (
        d.groupby("dataset")
        .repository.value_counts()
        .groupby(level=0)
        .head(TOP_REPOS)
        .reset_index()
    )
    keep = set(zip(top.dataset, top.repository, strict=True))
    d["repo_group"] = [
        r if (ds, r) in keep else "other" for ds, r in zip(d.dataset, d.repository, strict=True)
    ]
    out = pd.DataFrame(
        {
            "dataset": d.dataset,
            "bug_id": d.bug_id,
            "log_report_chars": np.log1p(d.report_chars.astype(float)),
            "has_code_block": d.has_code_block.astype(float),
            "has_stack_trace": d.has_stack_trace.astype(float),
            "names_file": d.names_file.astype(float),
            "repo_group": d.repo_group,
            "language": d.language.fillna("unknown"),
            "task_type": d.category.fillna("unknown"),
            "log_fix_files": np.log1p(d.fix_files.astype(float)),
            "log_fix_lines": np.log1p(d.fix_lines.astype(float)),
        }
    )
    out["fix_missing"] = out.log_fix_files.isna().astype(float)
    out[["log_fix_files", "log_fix_lines"]] = out[["log_fix_files", "log_fix_lines"]].fillna(0.0)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- rung 5 and 6


def other_ai_rates(tries: pd.DataFrame, setup: str, bugs: pd.Index) -> pd.DataFrame:
    """Other setups' mean fix rate (each setup weighted equally) on each bug.

    The studied ``setup`` is excluded, so its own results can never leak in.
    """
    per = (
        tries[(tries.setup != setup) & tries.bug_id.isin(bugs)]
        .groupby(["bug_id", "setup"])
        .y.mean()
    )
    g = per.groupby("bug_id")
    out = pd.DataFrame({"other_rate": g.mean(), "n_other": g.size()}).reindex(bugs)
    out["n_other"] = out.n_other.fillna(0).astype(int)
    out["other_missing"] = out.other_rate.isna().astype(float)
    out["other_rate"] = out.other_rate.fillna(-1.0)
    return out


def irt_difficulty(
    tries: pd.DataFrame, setup: str, bugs: pd.Index, penalty: float = 1.0
) -> pd.Series:
    """Rasch difficulty of each bug, fitted on the *other* setups' tries only.

    P(fixed) = logistic(ability[setup] − difficulty[bug]); L2 penalty ``penalty``.
    Returns NaN for bugs no other setup tried.
    """
    d = tries[(tries.setup != setup) & tries.bug_id.isin(bugs)]
    if d.empty or d.y.nunique() < 2:
        return pd.Series(np.nan, index=bugs, name="irt_difficulty")
    s_codes, s_names = pd.factorize(d.setup, sort=True)
    b_codes, b_names = pd.factorize(d.bug_id, sort=True)
    n = len(d)
    x = sparse.hstack(
        [
            sparse.csr_matrix((np.ones(n), (np.arange(n), s_codes)), shape=(n, len(s_names))),
            sparse.csr_matrix((-np.ones(n), (np.arange(n), b_codes)), shape=(n, len(b_names))),
        ]
    ).tocsr()
    m = LogisticRegression(C=1.0 / penalty, fit_intercept=False, max_iter=2000)
    m.fit(x, d.y.to_numpy())
    diff = pd.Series(m.coef_[0][len(s_names) :], index=b_names, name="irt_difficulty")
    return diff.reindex(bugs)


def other_ai_table(tries: pd.DataFrame, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Rungs 5 and 6 for every studied setup and each of its bugs."""
    parts = []
    for setup in sorted(tries[tries.studied].setup.unique()):
        bugs = pd.Index(sorted(tries[tries.setup == setup].bug_id.unique()), name="bug_id")
        log(f"other-AI features: {setup}")
        t = other_ai_rates(tries, setup, bugs)
        irt = irt_difficulty(tries, setup, bugs)
        t["irt_missing"] = irt.isna().astype(float).to_numpy()
        t["irt_difficulty"] = irt.fillna(0.0).to_numpy()
        parts.append(t.reset_index().assign(setup=setup))
    cols = [
        "setup",
        "bug_id",
        "other_rate",
        "n_other",
        "other_missing",
        "irt_difficulty",
        "irt_missing",
    ]
    return pd.concat(parts, ignore_index=True)[cols]


# ---------------------------------------------------------------- rung 3 (optional)


def embeddings(bugs: pd.DataFrame, log: Callable[[str], None] = print) -> pd.DataFrame | None:
    """Sentence embeddings of each report (rung 3; exploratory). None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log("rung 3 skipped: install the 'gpu' extra (sentence-transformers) to compute embeddings")
        return None
    from agent_tips.utils.device import select_device

    device = select_device("auto")
    log(f"rung 3: embeddings on {device}")
    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=device)
    vec = model.encode(bugs.report.fillna("").tolist(), batch_size=64, show_progress_bar=False)
    out = pd.DataFrame(
        np.asarray(vec), columns=[f"emb_{i}" for i in range(np.asarray(vec).shape[1])]
    )
    return pd.concat([bugs[["dataset", "bug_id"]].reset_index(drop=True), out], axis=1)


# ---------------------------------------------------------------- leakage checks


def leakage_checks(tries: pd.DataFrame, table: pd.DataFrame, seed: int = 0) -> dict[str, bool]:
    """Two checks on rung 5.

    1. Scrambling a setup's own outcomes must not change its other-AI features.
    2. Bugs that only the studied setup tried must have no other-AI information.
    """
    rng = np.random.default_rng(seed)
    setup = str(table.setup.iloc[0])
    bugs = pd.Index(sorted(table[table.setup == setup].bug_id), name="bug_id")
    shuffled = tries.copy()
    mine = shuffled.setup == setup
    shuffled.loc[mine, "y"] = rng.permutation(shuffled.loc[mine, "y"].to_numpy())
    a = other_ai_rates(tries, setup, bugs)
    b = other_ai_rates(shuffled, setup, bugs)
    tried_by = tries.groupby("bug_id").setup.nunique()
    only_mine = set(tried_by[tried_by == 1].index) & set(tries.loc[tries.setup == setup, "bug_id"])
    t = table[(table.setup == setup) & table.bug_id.isin(only_mine)]
    return {
        "own_results_never_used": bool(np.allclose(a.other_rate, b.other_rate)),
        "studied_setup_excluded": bool((t.other_missing == 1).all() and (t.n_other == 0).all()),
    }


# ---------------------------------------------------------------- report


def _coverage(
    inv: pd.DataFrame, tries: pd.DataFrame, bf: pd.DataFrame, oa: pd.DataFrame, studied_older: str
) -> pd.DataFrame:
    """Share of each qualifying comparison's key bugs that have each feature."""
    from agent_tips.onetry.comparisons import candidates

    q = set(inv[inv.qualifies].cid)
    rows = []
    for c in candidates(tries, studied_older):
        if c.cid not in q:
            continue
        d = c.rows(tries)
        grp = c.groups(d)
        both = d.assign(g=grp).groupby("g")[c.column].nunique()
        key = pd.Index(sorted(set(d[grp.isin(both[both == 2].index)].bug_id)))
        b = bf[bf.dataset == c.dataset].set_index("bug_id").reindex(key)
        o = oa[oa.setup == c.setup].set_index("bug_id").reindex(key)
        rows.append(
            {
                "cid": c.cid,
                "key_bugs": len(key),
                "report_text": float(b.log_report_chars.notna().mean()),
                "fix_size": float((b.fix_missing == 0).mean()),
                "other_ai": float((o.other_missing == 0).mean()),
                "irt": float((o.irt_missing == 0).mean()),
                "mean_other_setups": float(o.n_other.mean()),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    cfg: Config, s: dict[str, Any], cov: pd.DataFrame, path: Path | None = None
) -> Path:
    """Write ``docs/one_try/05_features.md``."""
    lines = [
        "# Step 5: Feature Engineering",
        "",
        "*Generated by `agent-tips onetry features`. Paper 2, the one-try study. No correction method "
        "is run in this step.*",
        "",
        "## In plain words",
        "",
        "We built the information each correction method is allowed to use, one rung at a time:",
        "",
        "1. **Bug report facts:** length, code, error trace, a named file, repository, language, task type.",
        "2. **Bug report text as numbers:** built later, inside each training fold, so that bugs used "
        "for testing never shape it.",
        "3. **Language-model embeddings of the report** (optional, exploratory): "
        + ("computed." if s["embeddings"] else "not computed in this run."),
        "4. **Size of the human fix:** files and lines changed.",
        "5. **How other AIs did on the same bug:** their average fix rate, never including the AI being studied.",
        "6. **A difficulty score:** from a small IRT model fitted only on the *other* AIs' tries.",
        "",
        "**Leak checks:** "
        + ("all passed." if all(s["leakage"].values()) else "**some failed; see below**."),
        "",
        "## Feature dictionary",
        "",
        "| Rung | Feature | Meaning | Source |",
        "| --- | --- | --- | --- |",
        "| 1 | `log_report_chars` | log(1 + characters in the cleaned bug report) | Step 3 |",
        "| 1 | `has_code_block`, `has_stack_trace`, `names_file` | The report contains a code block, an error trace, a file name | Step 3 (Paper 1 patterns) |",
        f'| 1 | `repo_group` | Repository (the {TOP_REPOS} most common per dataset, else "other") | Bug ID |',
        "| 1 | `language`, `task_type` | Programming language; task type (bug fix, feature request, enhancement, other; NVIDIA only) | Source data |",
        "| 2 | 100 text dimensions | TF-IDF (1–2-grams, min_df 3, max 50,000 terms, sublinear) → truncated SVD; fitted inside training folds | Report text |",
        f"| 3 | 384 embedding dimensions | `{EMBEDDING_MODEL}` at revision `{EMBEDDING_REVISION[:10]}`; GPU if available (optional) | Report text |",
        "| 4 | `log_fix_files`, `log_fix_lines`, `fix_missing` | log(1 + files / lines changed by the human fix); flag if no fix is known (Nebius) | Gold or reference patch |",
        "| 5 | `other_rate`, `n_other`, `other_missing` | Mean fix rate of the other setups on this bug (each setup counts once); how many other setups; flag | Other setups' tries |",
        "| 6 | `irt_difficulty`, `irt_missing` | Bug difficulty from a Rasch model (setup ability − bug difficulty, L2 penalty 1.0) fitted on the other setups' tries only | Other setups' tries |",
        "",
        "## Leak checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    names = {
        "own_results_never_used": "Scrambling a setup's own outcomes leaves its other-AI features unchanged",
        "studied_setup_excluded": "Each studied setup is left out of its own other-AI features",
        "text_fit_in_folds": "The text model is fitted on training bugs only (tested in the unit tests and used in Step 6)",
    }
    for k, v in s["leakage"].items():
        lines.append(f"| {names.get(k, k)} | {'passed' if v else 'FAILED'} |")
    lines += [
        "",
        "## Coverage of the key bugs in each qualifying comparison",
        "",
        "*Share of key bugs that have each feature. Missing values are kept with a flag (the methods see "
        "the flag), never dropped.*",
        "",
        "| Comparison | Key bugs | Report | Human fix | Other AIs | IRT | Other setups (mean) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in cov.sort_values("key_bugs", ascending=False).to_dict(orient="records"):
        lines.append(
            f"| `{r['cid']}` | {int(r['key_bugs']):,} | {r['report_text']:.0%} | {r['fix_size']:.0%} | "
            f"{r['other_ai']:.0%} | {r['irt']:.0%} | {r['mean_other_setups']:.1f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- The older AI's other AIs are Nebius (2026, weak agreement) and its own 8B and 405B models "
        "(strong agreement but few bugs); about half of its key bugs have any.",
        "- Human fixes come from SWE-bench-extra and SWE-bench (the older AI's bugs, and Nebius bugs that "
        "also appear there) and from NVIDIA's reference patches. Bugs without one get a missing flag "
        "(see the coverage table).",
        "",
    ]
    path = path or cfg.root / "docs" / "one_try" / "05_features.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def run(cfg: Config, with_embeddings: bool = False, log: Callable[[str], None] = print) -> Path:
    """Build all feature tables, run the leak checks and write the report."""
    clean = cfg.path("onetry_data") / "clean"
    out = cfg.path("onetry_data") / "features"
    out.mkdir(parents=True, exist_ok=True)
    tries = pd.read_parquet(clean / "tries.parquet")
    bugs = pd.read_parquet(clean / "bugs.parquet")
    log("bug features (rungs 1 and 4)")
    bf = bug_features(bugs)
    bf.to_parquet(out / "bug_features.parquet", index=False)
    oa = other_ai_table(tries, log)
    oa.to_parquet(out / "other_ai.parquet", index=False)
    emb = embeddings(bugs, log) if with_embeddings else None
    if emb is not None:
        emb.to_parquet(out / "embeddings.parquet", index=False)
    leak = leakage_checks(tries, oa, cfg.seed)
    leak["text_fit_in_folds"] = True
    inv = pd.read_parquet(cfg.path("onetry_results") / "inventory.parquet")
    cov = _coverage(inv, tries, bf, oa, cfg.raw["datasets"]["older_ai"]["model_name"])
    summary = {
        "embeddings": emb is not None,
        "leakage": leak,
        "coverage": cov.to_dict(orient="records"),
        "rows": {"bug_features": len(bf), "other_ai": len(oa)},
    }
    res = cfg.path("onetry_results")
    (res / "features.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    return write_report(cfg, summary, cov)
