"""More tips, older AI (swe-agent-llama-70b): does checking the fix help?

The prompt says: "When you think you've fixed the bug, re-run the bug
reproduction script to make sure that the bug has indeed been fixed."
Among tries that edited the project's code, compare tries that re-ran a
reproduction script (or the test suite) after their first code edit with
tries that did not. Same-issue comparisons use the same evaluation day.

Usage (from the project root):
    python -I analysis/exploratory/older_ai_more.py > analysis/exploratory/results/older_ai_more.txt
"""
import argparse
import re

import duckdb
import numpy as np
import pandas as pd

CMD_RE = r"(?s)```[a-z]*\n(.*?)\n?```"
OPEN_RE = r"\(Open file: ([^)]*)\)"
RERUN_RE = r"^(python3?|bash|sh)\s+\S*(repro|test|bug|issue)"
SUITE_RE = r"(pytest|python3? -m (pytest|unittest)|tox\b)"


def after_first_fix(fe, steps, pattern, anchored):
    """Does any command after the first code edit (1-based position fe) match the pattern?"""
    if fe is None or pd.isna(fe):
        return False
    test = re.match if anchored else re.search
    return any(test(pattern, s["cmd"] or "") for s in steps[int(fe):])


def fair(d, col, a, b, keys, rng, y="resolved", reps=1000, scale=100, unit=" pts"):
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
    return (f"{np.sum(w * dd) / np.sum(w) * scale:+.1f}{unit} [{lo * scale:+.1f}, {hi * scale:+.1f}]"
            f" (issues={st.i.nunique()}, runs={int((st.na + st.nb).sum())})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/SWE-agent-trajectories/data")
    p.add_argument("--runs", default="analysis/exploratory/data/runs_70b.parquet")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    con = duckdb.connect()
    # For each AI turn: the command, and which file was open just before it.
    ev = con.execute(rf"""
    WITH t AS (
      SELECT regexp_extract(filename, 'train-(\d+)', 1) || ':' || file_row_number AS run_id, trajectory AS tr
      FROM read_parquet('{args.data}/*.parquet', filename = true, file_row_number = true)
      WHERE model_name = 'swe-agent-llama-70b'),
    s AS (
      SELECT run_id, list_transform(generate_series(3, len(tr)::BIGINT, 2), j -> struct_pack(
          cmd := trim(regexp_extract(tr[j].text, '{CMD_RE}', 1)),
          open := regexp_extract(tr[j - 1].text, '{OPEN_RE}', 1))) AS steps
      FROM t)
    SELECT run_id,
      list_min(list_filter(list_transform(steps, (s, i) -> CASE WHEN s.cmd LIKE 'edit %'
               AND NOT regexp_matches(lower(s.open), '(repro|test|bug|issue)') THEN i END), v -> v IS NOT NULL)) AS fe,
      steps
    FROM s
    """).df()
    ev["rerun_after_fix"] = [after_first_fix(fe, s, RERUN_RE, True) for fe, s in zip(ev.fe, ev.steps)]
    ev["suite_after_fix"] = [after_first_fix(fe, s, SUITE_RE, False) for fe, s in zip(ev.fe, ev.steps)]
    runs = pd.read_parquet(args.runs).merge(ev[["run_id", "fe", "rerun_after_fix", "suite_after_fix"]], on="run_id")
    modal = runs.groupby("instance_id").issue_hash.agg(lambda s: s.value_counts().index[0])
    runs = runs[(runs.issue_hash == runs.instance_id.map(modal)) & runs.fe.notna()].copy()
    runs["day"] = runs.eval_ts.dt.floor("D")
    runs["created_repro"] = runs.first_create_repro_step.notna()
    dated = runs[runs.day.notna()]

    print("=== Checking the fix (older AI; tries that edited project code) ===")
    print(f"tries that edited code={len(runs)}; re-ran a repro/test script after the first fix: "
          f"{runs.rerun_after_fix.mean():.1%}; ran the test suite after the first fix: {runs.suite_after_fix.mean():.1%}")
    for col, label in [("rerun_after_fix", "re-ran repro/test script after fixing"),
                       ("suite_after_fix", "ran the test suite after fixing")]:
        d = dated.assign(arm=np.where(dated[col], "yes", "no"))
        simple = (d[d.arm == "yes"].resolved.mean() - d[d.arm == "no"].resolved.mean()) * 100
        print(f"{label}: simple {simple:+.1f} | fair (same issue x day) {fair(d, 'arm', 'yes', 'no', ['instance_id', 'day'], rng)}")
    d = dated[dated.created_repro].assign(arm=lambda x: np.where(x.rerun_after_fix, "yes", "no"))
    simple = (d[d.arm == "yes"].resolved.mean() - d[d.arm == "no"].resolved.mean()) * 100
    print(f"(only tries that wrote a repro script) re-ran it after fixing: simple {simple:+.1f} | fair "
          f"{fair(d, 'arm', 'yes', 'no', ['instance_id', 'day'], rng)}")

    print("\n=== Cost of reproducing first (older AI, same issue x day) ===")
    allr = pd.read_parquet(args.runs)
    allr = allr[allr.issue_hash == allr.instance_id.map(modal)].copy()
    first = allr.cmd0.fillna("").str.split().str[0].fillna("")
    allr["arm"] = np.where(first == "create", "R", np.where(
        first.isin(["search_dir", "find_file", "ls", "grep", "find", "search_file"]), "S", "O"))
    allr["day"] = allr.eval_ts.dt.floor("D")
    da = allr[allr.day.notna()]
    print("steps (commands):", fair(da, "arm", "R", "S", ["instance_id", "day"], rng, y="n_commands", scale=1, unit=" cmds"))


if __name__ == "__main__":
    main()
