"""Step 5 of the one-try study: features and leak checks."""

import numpy as np
import pandas as pd

from agent_tips.config import Config
from agent_tips.onetry import clean, collect, eda, features


def _quiet(_: str) -> None:
    pass


def _tries() -> pd.DataFrame:
    rows = []
    # bug "easy" is fixed by everyone, bug "hard" by no one; setup "me" is the studied one
    for setup in ("me", "o1", "o2"):
        for bug, y in (("easy", 1.0), ("hard", 0.0), ("mixed", 1.0 if setup == "o1" else 0.0)):
            rows += [(setup, bug, y)] * (3 if setup == "o1" else 1)
    rows.append(("me", "only_me", 1.0))
    d = pd.DataFrame(rows, columns=["setup", "bug_id", "y"])
    d["studied"] = d.setup == "me"
    return d


def test_text_features_learn_only_from_training_texts() -> None:
    train = pd.Series([f"error in parser module number {i} crash" for i in range(30)])
    tf = features.TextFeatures(dims=5).fit(train)
    vocab = dict(tf.tfidf.vocabulary_)
    out = tf.transform(pd.Series(["completely unseen words here", None]))
    assert out.shape == (2, 5)
    assert dict(tf.tfidf.vocabulary_) == vocab  # transforming never changes the fitted model


def test_other_ai_rate_excludes_the_studied_setup_and_weights_setups_equally() -> None:
    t = _tries()
    out = features.other_ai_rates(t, "me", pd.Index(["easy", "hard", "mixed", "only_me"]))
    assert out.loc["easy", "other_rate"] == 1.0
    assert out.loc["hard", "other_rate"] == 0.0
    # o1 fixed "mixed" in all 3 tries, o2 never: equal weight per setup gives 0.5, not 0.75
    assert out.loc["mixed", "other_rate"] == 0.5
    assert out.loc["only_me", "other_missing"] == 1.0 and out.loc["only_me", "n_other"] == 0


def test_irt_ranks_the_hard_bug_as_harder() -> None:
    t = _tries()
    d = features.irt_difficulty(t, "me", pd.Index(["easy", "hard", "mixed", "only_me"]))
    assert d["hard"] > d["mixed"] > d["easy"]
    assert np.isnan(d["only_me"])


def test_leakage_checks_pass_on_clean_features() -> None:
    t = _tries()
    table = features.other_ai_table(t, _quiet)
    checks = features.leakage_checks(t, table)
    assert all(checks.values())


def test_features_step_end_to_end(big_project: Config) -> None:
    collect.run(big_project, log=_quiet)
    clean.run(big_project, log=_quiet)
    eda.run(big_project, log=_quiet)
    report = features.run(big_project, log=_quiet)
    f = big_project.path("onetry_data") / "features"
    bf = pd.read_parquet(f / "bug_features.parquet")
    oa = pd.read_parquet(f / "other_ai.parquet")
    assert not bf.duplicated(["dataset", "bug_id"]).any()
    assert {"other_rate", "irt_difficulty", "n_other"} <= set(oa.columns)
    # in the fake project two NVIDIA setups share their bugs, so each has one other setup
    nv = oa[oa.setup.str.contains("/")]
    assert (nv.n_other >= 1).all()
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Step 5: Feature Engineering")
    assert "**Leak checks:** all passed." in text
