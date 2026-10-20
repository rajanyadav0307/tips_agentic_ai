"""Probes used to compare project options on the older AI (swe-agent-llama-70b).

1. Simple vs. fair comparisons of first moves (each vs. 'search_dir').
2. Does the reproduce-first effect differ by type of issue?
3. Does the agent's final claim of success match reality?

"Fair" = same issue and same evaluation day (our proxy for model version).

Usage (from the project root):
    python -I analysis/exploratory/options_probes.py > analysis/exploratory/results/options_probes.txt
"""
import argparse

import duckdb
import numpy as np
import pandas as pd

SEARCH = ["search_dir", "find_file", "ls", "grep", "find", "search_file"]


def fair(d, col, a, b, keys, rng, y="resolved", reps=1000):
    """Weighted within-group difference (a minus b), 95% interval from resampling issues."""
    d = d[d[col].isin([a, b])]
    g = d.groupby(keys + [col])[y].agg(["size", "sum"]).unstack(col, fill_value=0)
    st = pd.DataFrame({"i": g.index.get_level_values(0), "na": g[("size", a)].values, "nb": g[("size", b)].values,
                       "ka": g[("sum", a)].values, "kb": g[("sum", b)].values})
    st = st[(st.na > 0) & (st.nb > 0)]
    w = st.na * st.nb / (st.na + st.nb)
    dd = st.ka / st.na - st.kb / st.nb
    _, idx = np.unique(st.i, return_inverse=True)
    num, den = np.bincount(idx, weights=w * dd), np.bincount(idx, weights=w)
    draws = rng.integers(0, len(num), size=(reps, len(num)))
    lo, hi = np.percentile(num[draws].sum(1) / den[draws].sum(1), [2.5, 97.5])
    return (f"{np.sum(w * dd) / np.sum(w) * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]"
            f" (issues={st.i.nunique()}, runs={int((st.na + st.nb).sum())})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="analysis/exploratory/data/runs_70b.parquet")
    p.add_argument("--data", default="data/SWE-agent-trajectories/data")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    con = duckdb.connect()
    raw = con.execute(rf"""
    SELECT regexp_extract(filename, 'train-(\d+)', 1) || ':' || file_row_number AS run_id,
      regexp_extract(trajectory[2].text, '(?s)ISSUE:\n(.*?)\n\nINSTRUCTIONS:', 1) AS issue,
      list_filter(trajectory, e -> e.role = 'ai')[-1].text AS last_ai
    FROM read_parquet('{args.data}/*.parquet', filename = true, file_row_number = true)
    WHERE model_name = 'swe-agent-llama-70b'
    """).df()
    df = pd.read_parquet(args.runs).merge(raw, on="run_id")
    modal = df.groupby("instance_id").issue_hash.agg(lambda s: s.value_counts().index[0])
    df = df[df.issue_hash == df.instance_id.map(modal)].copy()
    first = df.cmd0.fillna("").str.split().str[0].fillna("")
    df["move"] = np.where(first.isin(["create", "open"] + SEARCH), first, "other")
    df["arm"] = np.where(first == "create", "R", np.where(first.isin(SEARCH), "S", "O"))
    df["day"] = df.eval_ts.dt.floor("D")
    dated = df[df.day.notna()]

    print("=== 1. First move vs 'search_dir': simple comparison vs fair comparison (points) ===")
    for a in ["create", "find_file", "ls", "open", "grep"]:
        simple = (df[df.move == a].resolved.mean() - df[df.move == "search_dir"].resolved.mean()) * 100
        print(f"{a:>10}: simple {simple:+.1f} | same issue {fair(df, 'move', a, 'search_dir', ['instance_id'], rng)}"
              f" | fair {fair(dated, 'move', a, 'search_dir', ['instance_id', 'day'], rng)}")

    print("\n=== 2. Reproduce first (R) vs search first (S), fair comparison, by type of issue ===")
    print("all:", fair(dated, "arm", "R", "S", ["instance_id", "day"], rng))
    iss = df.groupby("instance_id").issue.first()
    feats = {
        "has a code block": iss.str.contains("```", regex=False),
        "has a traceback": iss.str.contains("Traceback (most recent call last)", regex=False),
        "names a .py file": iss.str.contains(r"[\w/]+\.py\b", regex=True),
        "longer than median": iss.str.len() > iss.str.len().median(),
    }
    for name, f in feats.items():
        m = dated.instance_id.map(f)
        print(f"{name:>20}: yes {fair(dated[m], 'arm', 'R', 'S', ['instance_id', 'day'], rng)}"
              f" | no {fair(dated[~m], 'arm', 'R', 'S', ['instance_id', 'day'], rng)}")

    print("\n=== 3. Final claim vs reality (runs that submitted) ===")
    sub = df[df.exit_status == "submitted"].copy()
    thought = sub.last_ai.fillna("").str.split("```").str[0].str.lower()
    sub["claim"] = np.where(thought.str.contains(
        r"(?:has been|have been|is now|are now|been successfully) (?:fixed|resolved|addressed)"
        r"|fix(?:es)? (?:works|worked)|successfully (?:fixed|resolved)|issue is (?:fixed|resolved)"), "C", "N")
    print(f"submitted runs={len(sub)}; share claiming the fix works={(sub.claim == 'C').mean():.1%}")
    print("resolved | claim:", round(sub[sub.claim == "C"].resolved.mean(), 3),
          "| no claim:", round(sub[sub.claim == "N"].resolved.mean(), 3))
    print("same issue, claim minus no claim:", fair(sub, "claim", "C", "N", ["instance_id"], rng))


if __name__ == "__main__":
    main()
