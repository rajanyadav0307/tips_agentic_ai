"""Load the one-try study's clean and feature tables, and attach features to a copy."""

from __future__ import annotations

from dataclasses import dataclass

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


def load(cfg: Config, qualifying_only: bool = True) -> StudyData:
    """Load the clean tries, features and the fixed list of comparisons."""
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
    )


def with_features(
    copy: pd.DataFrame,
    c: Comparison,
    s: StudyData,
    other_ai: pd.DataFrame | None = None,
    embeddings: bool = False,
) -> pd.DataFrame:
    """Attach every rung's features to a one-try copy (one row per bug)."""
    bf = s.bug_features[s.bug_features.dataset == c.dataset].drop(columns="dataset")
    rep = s.reports[s.reports.dataset == c.dataset].drop(columns="dataset")
    oa = other_ai if other_ai is not None else s.other_ai
    oa = oa[oa.setup == c.setup].drop(columns="setup")
    d = copy.merge(bf, on="bug_id", how="left").merge(rep, on="bug_id", how="left")
    d = d.merge(oa, on="bug_id", how="left")
    if embeddings and s.embeddings is not None:
        emb = s.embeddings[s.embeddings.dataset == c.dataset].drop(columns="dataset")
        d = d.merge(emb, on="bug_id", how="left")
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
