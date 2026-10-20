"""Replication checks on the newer AI (OpenHands + Qwen3-Coder-480B).

1. Does it write a test/reproduction script before its first edit of the
   project's code? (Only if it varies can we compare fairly.)
2. Fair coin-flip comparisons where tries of the same issue had exactly the
   same history so far and then chose differently (steps 0-3).

Usage (from the project root):
    python -I analysis/exploratory/newer_ai_checks.py \
        --data data/SWE-rebench-openhands-trajectories/trajectories.parquet
"""
import argparse

import duckdb
import numpy as np
import pandas as pd

MAX_DEPTH = 4  # look for forks in the first 4 steps


def load(path):
    con = duckdb.connect()
    df = con.execute(f"""
    WITH a AS (
      SELECT trajectory_id, instance_id, resolved::INT AS resolved, exit_status,
             md5(trajectory[2].content) AS issue_hash,
             list_filter(trajectory, m -> m.role = 'assistant') AS msgs
      FROM read_parquet('{path}')),
    s AS (
      SELECT *, list_transform(msgs, m -> struct_pack(
          fn := coalesce(m.tool_calls[1].function.name, 'none'),
          cmd := coalesce(json_extract_string(m.tool_calls[1].function.arguments, '$.command'), ''),
          path := coalesce(json_extract_string(m.tool_calls[1].function.arguments, '$.path'), ''),
          h := md5(coalesce(m.content, '') || '|' || coalesce(array_to_string(
                   list_transform(m.tool_calls, t -> t.function.name || ':' || t.function.arguments), '||'), '')))) AS steps
      FROM a),
    c AS (SELECT *, list_transform(list_filter(steps, x -> x.cmd = 'create'), x -> x.path) AS created FROM s)
    SELECT trajectory_id, instance_id, resolved, exit_status, issue_hash, len(steps) AS n_steps,
      list_min(list_filter(list_transform(steps, (x, i) -> CASE WHEN x.cmd = 'create' THEN i END),
                           v -> v IS NOT NULL)) - 1 AS first_create,
      list_min(list_filter(list_transform(steps, (x, i) -> CASE WHEN x.cmd IN ('str_replace', 'insert')
                           AND NOT list_contains(created, x.path) THEN i END), v -> v IS NOT NULL)) - 1 AS first_src_edit,
      list_transform(steps[1:{MAX_DEPTH + 1}], x -> x.h) AS hashes,
      list_transform(steps[1:{MAX_DEPTH + 1}], x ->
        CASE WHEN x.fn = 'think' THEN 'think'
             WHEN x.fn = 'task_tracker' THEN 'plan'
             WHEN x.fn = 'finish' THEN 'finish'
             WHEN x.fn = 'str_replace_editor' AND x.cmd = 'view' THEN 'view'
             WHEN x.fn = 'str_replace_editor' AND x.cmd = 'create' THEN 'create'
             WHEN x.fn = 'str_replace_editor' THEN 'edit'
             WHEN x.fn = 'execute_bash' AND regexp_matches(x.cmd, '(python|pytest)') THEN 'run'
             WHEN x.fn = 'execute_bash' THEN 'bash-look'
             ELSE 'other' END) AS kinds
    FROM c
    """).df()
    modal = df.groupby("instance_id").issue_hash.agg(lambda s: s.value_counts().index[0])
    return df[df.issue_hash == df.instance_id.map(modal)].reset_index(drop=True)


def weighted_diff(n_a, n_b, k_a, k_b):
    w = n_a * n_b / (n_a + n_b)
    return np.sum(w * (k_a / n_a - k_b / n_b)) / np.sum(w)


def compare(d, group_cols, arm_col, a, b, rng, y="resolved", reps=2000):
    """Within-group difference (a minus b) with a 95% interval from resampling issues."""
    d = d[d[arm_col].isin([a, b])]
    g = d.groupby(group_cols + [arm_col])[y].agg(["size", "sum"]).unstack(arm_col, fill_value=0)
    if ("size", a) not in g or ("size", b) not in g:
        return None
    st = pd.DataFrame({"inst": g.index.get_level_values(0),
                       "n_a": g[("size", a)].values, "n_b": g[("size", b)].values,
                       "k_a": g[("sum", a)].values, "k_b": g[("sum", b)].values})
    st = st[(st.n_a > 0) & (st.n_b > 0)]
    if st.empty:
        return None
    est = weighted_diff(st.n_a, st.n_b, st.k_a, st.k_b)
    _, idx = np.unique(st.inst, return_inverse=True)
    w = st.n_a * st.n_b / (st.n_a + st.n_b)
    num = np.bincount(idx, weights=w * (st.k_a / st.n_a - st.k_b / st.n_b))
    den = np.bincount(idx, weights=w)
    draws = rng.integers(0, len(num), size=(reps, len(num)))
    lo, hi = np.percentile(num[draws].sum(1) / den[draws].sum(1), [2.5, 97.5])
    return dict(est=est, lo=lo, hi=hi, issues=st.inst.nunique(), groups=len(st),
                runs=int(st.n_a.sum() + st.n_b.sum()))


def fmt(r):
    if r is None:
        return "not enough data"
    return (f"{r['est']*100:+.2f} pts  [{r['lo']*100:+.2f}, {r['hi']*100:+.2f}]  "
            f"issues={r['issues']} groups={r['groups']} runs={r['runs']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/SWE-rebench-openhands-trajectories/trajectories.parquet")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    df = load(args.data)
    df["max_iter"] = df.exit_status.str.contains("maximum iteration").astype(int)
    print(f"runs={len(df)} issues={df.instance_id.nunique()} resolved={df.resolved.mean():.3f}")

    print("\n=== 1. Test script before the first fix? ===")
    fc, fe = df.first_create.astype(float), df.first_src_edit.astype(float)
    df["order"] = np.select([fe.isna(), fc.notna() & (fc < fe)], ["never edits", "test first"], "fix first")
    print("share of runs:", df.order.value_counts(normalize=True).round(3).to_dict())
    print("median step of the first test script:", df.loc[df.order == "test first", "first_create"].median())
    simple = df[df.order == "test first"].resolved.mean() - df[df.order == "fix first"].resolved.mean()
    print(f"simple comparison (all issues pooled), test first minus fix first: {simple*100:+.2f} pts")
    print("test first minus fix first, within issue (few runs; not a coin flip):")
    print("   resolved:          ", fmt(compare(df, ["instance_id"], "order", "test first", "fix first", rng)))
    print("   hit the 100-step cap:", fmt(compare(df, ["instance_id"], "order", "test first", "fix first", rng,
                                                   y="max_iter")))

    print("\n=== 2. Fair coin flips: same issue, identical history so far, different next action ===")
    for depth in range(MAX_DEPTH):
        ok = df.kinds.apply(len) > depth
        d = df[ok].copy()
        d["node"] = d.hashes.apply(lambda h: "|".join(h[:depth]))
        d["next"] = d.kinds.apply(lambda k: k[depth])
        groups = d.groupby(["instance_id", "node"])
        d = d[groups.next.transform("nunique") > 1]
        if d.empty:
            print(f"step {depth}: no forks")
            continue
        pair = d.next.value_counts().index[:2].tolist()
        n_nodes = d.groupby(["instance_id", "node"]).ngroups
        print(f"step {depth}: {n_nodes} forks in {d.instance_id.nunique()} issues; "
              f"most common choices {d.next.value_counts().head(4).to_dict()}")
        print(f"   {pair[0]} minus {pair[1]}: {fmt(compare(d, ['instance_id', 'node'], 'next', pair[0], pair[1], rng))}")
        if depth == 0:
            first = df.kinds.apply(lambda k: k[0] if len(k) else "")
            simple = df[first == pair[0]].resolved.mean() - df[first == pair[1]].resolved.mean()
            print(f"   simple comparison (all issues pooled), {pair[0]} minus {pair[1]}: {simple*100:+.2f} pts")


if __name__ == "__main__":
    main()
