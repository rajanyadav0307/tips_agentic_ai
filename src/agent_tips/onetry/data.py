"""Load the one-try study's clean and feature tables, and attach features to a copy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agent_tips.config import Config
from agent_tips.onetry.comparisons import Comparison, candidates


@dataclass
class StudyData:
    """Everything Steps 6-7 need, loaded once."""

    tries: pd.DataFrame
    bug_features: pd.DataFrame
    reports: pd.DataFrame
    other_ai: pd.DataFrame
    embeddings: pd.DataFrame | None
    comparisons: list[Comparison]
    text: pd.DataFrame | None = None  # cached report-text numbers (amendment A9)


def text_path(cfg: Config) -> Path:
    """Where the cached report-text numbers live (amendment A9)."""
    return cfg.path("onetry_data") / "features" / "text_svd.parquet"


def load(cfg: Config, qualifying_only: bool = True) -> StudyData:
    """Load the clean tries, features, cached text (if built) and the comparisons."""
    base = cfg.path("onetry_data")
    tries = pd.read_parquet(base / "clean" / "tries.parquet")
    bugs = pd.read_parquet(base / "clean" / "bugs.parquet", columns=["dataset", "bug_id", "report"])
    feats = base / "features"
    emb_path = feats / "embeddings.parquet"
    inv = pd.read_parquet(cfg.path("onetry_results") / "inventory.parquet")
    keep = set(inv[inv.qualifies].cid) if qualifying_only else set(inv.cid)
    comps = [
        c for c in candidates(tries, cfg.raw["datasets"]["older_ai"]["model_name"]) if c.cid in keep
    ]
    return StudyData(
        tries=tries,
        bug_features=pd.read_parquet(feats / "bug_features.parquet"),
        reports=bugs,
        other_ai=pd.read_parquet(feats / "other_ai.parquet"),
        embeddings=pd.read_parquet(emb_path) if emb_path.is_file() else None,
        comparisons=comps,
        text=pd.read_parquet(text_path(cfg)) if text_path(cfg).is_file() else None,
    )


def with_features(
    copy: pd.DataFrame,
    c: Comparison,
    s: StudyData,
    other_ai: pd.DataFrame | None = None,
    embeddings: bool = False,
    text: bool = False,
) -> pd.DataFrame:
    """Attach every rung's features to a one-try copy (one row per bug).

    ``text=True`` adds the cached report-text numbers ``txt_*`` (amendment A9).
    """
    bf = s.bug_features[s.bug_features.dataset == c.dataset].drop(columns="dataset")
    rep = s.reports[s.reports.dataset == c.dataset].drop(columns="dataset")
    oa = other_ai if other_ai is not None else s.other_ai
    oa = oa[oa.setup == c.setup].drop(columns="setup")
    d = copy.merge(bf, on="bug_id", how="left").merge(rep, on="bug_id", how="left")
    d = d.merge(oa, on="bug_id", how="left")
    if embeddings and s.embeddings is not None:
        emb = s.embeddings[s.embeddings.dataset == c.dataset].drop(columns="dataset")
        d = d.merge(emb, on="bug_id", how="left")
    if text:
        if s.text is None:
            raise FileNotFoundError("cached report text missing; build it with text_table()")
        txt = s.text[s.text.dataset == c.dataset].drop(columns="dataset")
        d = d.merge(txt, on="bug_id", how="left")
        cols = [x for x in d.columns if x.startswith("txt_")]
        d[cols] = d[cols].fillna(0.0)
    d["report"] = d["report"].fillna("")
    for col, fill in (
        ("other_rate", -1.0),
        ("n_other", 0),
        ("other_missing", 1.0),
        ("irt_difficulty", 0.0),
        ("irt_missing", 1.0),
    ):
        d[col] = d[col].fillna(fill)
    return d.sort_values("bug_id").reset_index(drop=True)
