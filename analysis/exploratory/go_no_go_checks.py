"""Go/no-go checks for the 'reproduce the bug first' natural experiment.

Compares runs of the same issue that started by creating a reproduction
script (arm R) with runs that started by searching the code (arm S), only
inside groups of runs that likely came from the same model version
(same issue + same evaluation batch). Positive numbers mean R does better.

Usage (from the project root):
    python -I analysis/exploratory/go_no_go_checks.py --runs analysis/data/runs_70b.parquet
"""
import argparse

import numpy as np
import pandas as pd

SEARCH_RE = r"(search_dir|search_file|find_file|grep|find|ls)\b"
BATCHES = {  # name -> how runs are grouped into "same model version" batches
    "day": "b_day",
    "6 hours": "b_6h",
    "hour": "b_hour",
    "gap > 60 min": "b_gap60",
    "gap > 10 min": "b_gap10",
}


def load(path):
    df = pd.read_parquet(path)
    cmd0 = df.cmd0.fillna("")
    df["arm"] = np.select(
        [cmd0.str.match(r"(?i)create\s"), cmd0.str.match(SEARCH_RE), cmd0.str.match(r"(open|goto)\b")],
        ["R", "S", "O"], "X")
    # Identical input: keep runs that saw the most common text of their issue.
    modal = df.groupby("instance_id").issue_hash.agg(lambda s: s.value_counts().index[0])
    df["same_input"] = df.issue_hash == df.instance_id.map(modal)
    t = df.eval_ts
    df["b_day"], df["b_6h"], df["b_hour"] = t.dt.floor("D"), t.dt.floor("6h"), t.dt.floor("h")
    # Gap batches: within an issue, a new batch starts after a pause in evaluation times.
    d = df[t.notna()].sort_values(["instance_id", "eval_ts"])
    for name, gap in [("b_gap60", "60min"), ("b_gap10", "10min")]:
        new = (d.instance_id != d.instance_id.shift()) | (d.eval_ts.diff() > pd.Timedelta(gap))
        df.loc[d.index, name] = new.cumsum().astype(float)
    df["no_patch"] = (~df.has_patch).astype(int)
    df["ctx_submit"] = (df.exit_status == "submitted (exit_context)").astype(int)
    df["ctx_any"] = df.exit_status.str.contains("exit_context").astype(int)
    return df


def strata(d, key, y="resolved", a="R", b="S"):
    """Per stratum (issue x batch): run counts and outcome sums for arms a and b."""
    d = d[d.arm.isin([a, b]) & d[key].notna()]
    g = d.groupby(["instance_id", key, "arm"])[y].agg(["size", "sum"]).unstack("arm", fill_value=0)
    st = pd.DataFrame({"instance_id": g.index.get_level_values(0),
                       "n_a": g[("size", a)].values, "n_b": g[("size", b)].values,
                       "k_a": g[("sum", a)].values, "k_b": g[("sum", b)].values})
    return st[(st.n_a > 0) & (st.n_b > 0)].reset_index(drop=True)


def weighted_diff(n_a, n_b, k_a, k_b):
    """Average within-stratum difference (a minus b), weighting strata by n_a*n_b/(n_a+n_b)."""
    w = n_a * n_b / (n_a + n_b)
    return np.sum(w * (k_a / n_a - k_b / n_b)) / np.sum(w)


def cluster_ci(st, rng, reps=2000):
    """95% interval from resampling whole issues."""
    _, idx = np.unique(st.instance_id, return_inverse=True)
    w = st.n_a * st.n_b / (st.n_a + st.n_b)
    num = np.bincount(idx, weights=w * (st.k_a / st.n_a - st.k_b / st.n_b))
    den = np.bincount(idx, weights=w)
    draws = rng.integers(0, len(num), size=(reps, len(num)))
    return np.percentile(num[draws].sum(1) / den[draws].sum(1), [2.5, 97.5])


def effect(d, key, rng, y="resolved", a="R", b="S"):
    st = strata(d, key, y, a, b)
    est = weighted_diff(st.n_a.values, st.n_b.values, st.k_a.values, st.k_b.values)
    lo, hi = cluster_ci(st, rng)
    return dict(est=est, lo=lo, hi=hi, strata=len(st), issues=st.instance_id.nunique(),
                runs=int(st.n_a.sum() + st.n_b.sum()))


def fmt(r, scale=100, unit=" pts"):
    return (f"{r['est']*scale:+.2f}{unit}  [{r['lo']*scale:+.2f}, {r['hi']*scale:+.2f}]  "
            f"issues={r['issues']} groups={r['strata']} runs={r['runs']}")


def permutation_p(d, key, rng, reps=1000):
    """If the first action did not matter, how often would shuffling the R/S labels
    inside each group give a difference at least this large?"""
    d = d[d.arm.isin(["R", "S"]) & d[key].notna()]
    sid = d.groupby(["instance_id", key]).ngroup().values
    isR = (d.arm == "R").values.astype(float)
    y = d.resolved.values.astype(float)
    n_all = np.bincount(sid)
    nR_all = np.bincount(sid, weights=isR)
    keep = ((nR_all > 0) & (nR_all < n_all))[sid]
    _, sid = np.unique(sid[keep], return_inverse=True)
    isR, y = isR[keep], y[keep]
    n, nR = np.bincount(sid), np.bincount(sid, weights=isR)
    kT = np.bincount(sid, weights=y)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    est = lambda kR: weighted_diff(nR, n - nR, kR, kT - kR)
    obs = est(np.bincount(sid, weights=y * isR))
    null = np.empty(reps)
    for r in range(reps):
        order = np.lexsort((rng.random(len(sid)), sid))
        s = sid[order]
        assigned_R = (np.arange(len(s)) - start[s]) < nR[s]
        null[r] = est(np.bincount(s, weights=y[order] * assigned_R, minlength=len(n)))
    return obs, float(np.mean(np.abs(null) >= abs(obs)))


def hour_balance_p(d, rng, reps=1000):
    """Inside each issue x day group, does the hour predict the first action?
    If yes, different model versions are probably mixed within a day."""
    d = d[d.arm.isin(["R", "S"]) & d.b_day.notna()].copy()
    d["sid"] = d.groupby(["instance_id", "b_day"]).ngroup()
    d["hid"] = d.groupby(["instance_id", "b_day", "b_hour"]).ngroup()
    multi = d.groupby("sid").hid.transform("nunique") > 1
    mixed = d.groupby("sid").arm.transform("nunique") > 1
    d = d[multi & mixed]
    sid = pd.factorize(d.sid)[0]
    hid = pd.factorize(d.hid)[0]
    isR = (d.arm == "R").values.astype(float)
    pR_s = np.bincount(sid, weights=isR) / np.bincount(sid)
    n_h = np.bincount(hid)
    s_of_h = np.zeros(len(n_h), dtype=int)
    s_of_h[hid] = sid
    stat = lambda aR: np.sum((np.bincount(hid, weights=aR, minlength=len(n_h)) - n_h * pR_s[s_of_h]) ** 2 / n_h)
    obs = stat(isR)
    base = np.argsort(sid, kind="stable")
    labels = isR[base]
    null = np.empty(reps)
    for r in range(reps):
        order = np.lexsort((rng.random(len(sid)), sid))
        aR = np.empty_like(isR)
        aR[order] = labels
        null[r] = stat(aR)
    return dict(groups=int(len(np.unique(sid))), runs=int(len(sid)), ratio=obs / null.mean(),
                p=float(np.mean(null >= obs)))


def missing_scenarios(df, key, rng, reps=200):
    """Runs with no evaluation log (always unresolved) cannot be placed in a batch.
    Re-estimate after adding them back under different assumptions."""
    base = df[df.same_input & df.arm.isin(["R", "S"])]
    ts, nolog = base[base[key].notna()], base[base.eval_ts.isna()]
    g = ts.groupby(["instance_id", key, "arm"]).resolved.agg(["size", "sum"]).unstack("arm", fill_value=0)
    g = g.reset_index()
    inst = g.instance_id.values
    nR0, nS0 = g[("size", "R")].values.astype(float), g[("size", "S")].values.astype(float)
    kR0, kS0 = g[("sum", "R")].values.astype(float), g[("sum", "S")].values.astype(float)
    failR0, failS0 = nR0 - kR0, nS0 - kS0
    # one extra empty "unseen batch" per issue that has runs without a log
    extra = np.unique(nolog.instance_id)
    inst = np.concatenate([inst, extra])
    pad = np.zeros(len(extra))
    nR0, nS0, kR0, kS0 = (np.concatenate([x, pad]) for x in (nR0, nS0, kR0, kS0))
    failR0, failS0 = np.concatenate([failR0, pad]), np.concatenate([failS0, pad])
    is_new = np.concatenate([np.zeros(len(g), bool), np.ones(len(extra), bool)])
    by_inst = pd.Series(np.arange(len(inst))).groupby(inst).apply(np.array).to_dict()
    todo = nolog.groupby(["instance_id", "arm"]).size()

    def estimate(nR, nS):
        ok = (nR > 0) & (nS > 0)
        return weighted_diff(nR[ok], nS[ok], kR0[ok], kS0[ok])

    out = {"left out (main analysis)": estimate(nR0, nS0)}

    # Assumption: a missing run came from a batch in proportion to that batch's
    # failed runs with the same first action (failures look alike).
    ests = []
    for _ in range(reps):
        nR, nS = nR0.copy(), nS0.copy()
        for (iid, arm), m in todo.items():
            idx = by_inst[iid]
            old = idx[~is_new[idx]]
            if len(old) == 0:
                cand, p = idx[is_new[idx]], np.array([1.0])
            else:
                fails = (failR0 if arm == "R" else failS0)[old]
                if fails.sum() == 0:
                    fails = failR0[old] + failS0[old]
                if fails.sum() == 0:
                    fails = np.ones(len(old))
                cand, p = old, fails / fails.sum()
            add = rng.multinomial(m, p)
            if arm == "R":
                nR[cand] += add
            else:
                nS[cand] += add
        ests.append(estimate(nR, nS))
    out["added back, like other failures (median of 200 draws)"] = float(np.median(ests))

    # Rough worst and best cases: put each missing run where it moves the result most.
    runs = [(iid, arm) for (iid, arm), m in todo.items() for _ in range(m)]
    rng.shuffle(runs)
    for label, pick in [("added back, worst case for the finding", np.argmax),
                        ("added back, best case for the finding", np.argmin)]:
        nR, nS = nR0.copy(), nS0.copy()
        def contrib(j):
            if nR[j] > 0 and nS[j] > 0:
                w = nR[j] * nS[j] / (nR[j] + nS[j])
                return w * (kR0[j] / nR[j] - kS0[j] / nS[j]), w
            return 0.0, 0.0
        num = den = 0.0
        for j in range(len(nR)):
            c, w = contrib(j)
            num, den = num + c, den + w
        for iid, arm in runs:
            cand = by_inst[iid]
            vals = []
            for j in cand:
                c0, w0 = contrib(j)
                (nR if arm == "R" else nS)[j] += 1
                c1, w1 = contrib(j)
                (nR if arm == "R" else nS)[j] -= 1
                vals.append(((num - c0 + c1) / (den - w0 + w1), c0, w0, c1, w1))
            best = pick([v[0] for v in vals])
            _, c0, w0, c1, w1 = vals[best]
            (nR if arm == "R" else nS)[cand[best]] += 1
            num, den = num - c0 + c1, den - w0 + w1
        out[label] = num / den
    return out, len(nolog)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="analysis/exploratory/data/runs_70b.parquet")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    df = load(args.runs)
    same = df[df.same_input]
    rs = same[same.arm.isin(["R", "S"])]
    dated = rs[rs.eval_ts.notna()]

    print("=== 0. Data ===")
    print(f"runs={len(df)}  same-input runs={len(same)}  issues={df.instance_id.nunique()}")
    print("first-action arms (share of same-input runs):",
          same.arm.value_counts(normalize=True).round(3).to_dict())
    print("runs with an evaluation timestamp:", int(df.eval_ts.notna().sum()),
          "| without (all unresolved):", int(df.eval_ts.isna().sum()))

    print("\n=== 1+2. Finer batches: does the result hold, and how much data is left? ===")
    print("Positive = starting with reproduction does better. 95% interval from resampling issues.")
    for name, key in BATCHES.items():
        r = effect(dated, key, rng)
        share = r["runs"] / len(dated)
        print(f"{name:>13}: {fmt(r)}  ({share:.0%} of dated R/S runs usable)")

    print("\n=== 3. Chance check: shuffle R/S labels inside each group (1,000 times) ===")
    for name in ["day", "hour", "gap > 10 min"]:
        obs, pval = permutation_p(dated, BATCHES[name], rng)
        print(f"{name:>13}: observed {obs*100:+.2f} pts, share of shuffles at least as extreme = {pval:.3f}")

    print("\n=== 4. Are versions still mixed within a day? (does the hour predict the first action?) ===")
    hb = hour_balance_p(dated, rng)
    print(f"groups={hb['groups']} runs={hb['runs']}  hour-to-hour variation = {hb['ratio']:.2f}x what chance gives,"
          f" p = {hb['p']:.3f}")

    print("\n=== 5. Placebo: 'create reproduce.py' vs 'create reproduce_bug.py' (both are R; expect ~0) ===")
    plc = dated.copy()
    plc["arm"] = np.select([plc.cmd0.str.strip() == "create reproduce.py",
                            plc.cmd0.str.strip() == "create reproduce_bug.py"], ["P1", "P2"], "other")
    pl_all = same[same.eval_ts.notna()].copy()
    pl_all["arm"] = np.select([pl_all.cmd0.str.strip() == "create reproduce.py",
                               pl_all.cmd0.str.strip() == "create reproduce_bug.py"], ["P1", "P2"], "other")
    for name in ["day", "hour"]:
        print(f"{name:>13}: {fmt(effect(pl_all, BATCHES[name], rng, a='P1', b='P2'))}")

    print("\n=== 6. Runs with no evaluation log: add them back under different assumptions ===")
    for name in ["day", "hour"]:
        res, n_missing = missing_scenarios(df, BATCHES[name], rng)
        print(f"batches by {name} ({n_missing} R/S runs without a log):")
        for k, v in res.items():
            print(f"   {k:<55} {v*100:+.2f} pts")

    print("\n=== 7. Label check: do the arms mean what we think? ===")
    R, S = same[same.arm == "R"], same[same.arm == "S"]
    print("R: most common first commands:", R.cmd0.str.strip().value_counts().head(5).to_dict())
    print(f"R: share that run the reproduction script within the first 5 commands: "
          f"{(R.first_run_repro_step <= 4).mean():.1%}")
    later = S.first_create_repro_step.notna()
    print(f"S: share that create a reproduction script later: {later.mean():.1%}"
          f" (median step {S.first_create_repro_step[later].median():.0f});"
          f" never: {(~later).mean():.1%}")
    print("S: most common first command types:", S.cmd0.str.split().str[0].value_counts().head(5).to_dict())
    for arm, d in [("R", R), ("S", S)]:
        print(f"{arm}: creates any file after step 1: {d.later_create_any.mean():.1%};"
              f" runs a script of any kind: {d.runs_a_script.mean():.1%};"
              f" runs the project's test suite: {d.runs_test_suite.mean():.1%}")

    print("\n=== 8. First look at why (exploratory; same R vs S comparison, other outcomes) ===")
    print("within issue x day (dated runs):")
    print(f"   submitted because context ran out: {fmt(effect(dated, 'b_day', rng, y='ctx_submit'))}")
    r = effect(dated, "b_day", rng, y="n_commands")
    print(f"   number of commands: {fmt(r, scale=1, unit=' cmds')}")
    allr = rs.assign(one=1)
    print("within issue, all runs (versions not separated):")
    print(f"   resolved: {fmt(effect(allr, 'one', rng))}")
    print(f"   ended with no patch: {fmt(effect(allr, 'one', rng, y='no_patch'))}")
    print(f"   ran out of context (any exit): {fmt(effect(allr, 'one', rng, y='ctx_any'))}")


if __name__ == "__main__":
    main()
