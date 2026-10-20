"""Does writing a test/reproduction script before fixing help modern AIs?
Data: NVIDIA Open-SWE-Traces (run table from build_nvidia_runs.py).

Arms: 'test first' = created a test/reproduction file before its first edit of
existing project code; 'fix first' = edited existing code first.

Caveat: NVIDIA removed tries that hit the step limit, ended early, or produced
no patch, so every comparison here is among tries that finished normally.

Usage (from the project root):
    python -I analysis/exploratory/nvidia_checks.py > analysis/exploratory/results/nvidia_checks.txt
"""
import argparse

import numpy as np
import pandas as pd


def load(path):
    d = pd.read_parquet(path)
    fr, fe = d.first_repro.astype(float), d.first_edit.astype(float)
    d["arm"] = np.select([fe.isna(), fr.notna() & (fr < fe)], ["no edit found", "test first"], "fix first")
    fx = d.first_run.astype(float)
    d["ran_before_edit"] = np.where(fx.notna() & (fe.isna() | (fx < fe)), "ran code first", "edited before running")
    d["labeled"] = d.resolved >= 0
    d["y"] = (d.resolved == 1).astype(int)
    d["issue"] = d.subset + "::" + d.instance_id
    return d


def within(d, col, a, b, y="y", group="issue", rng=None, reps=2000, scale=100, unit=" pts"):
    """Weighted within-group difference (a minus b) with a 95% interval from resampling issues."""
    d = d[d[col].isin([a, b])]
    g = d.groupby([group, col])[y].agg(["size", "sum"]).unstack(col, fill_value=0)
    st = pd.DataFrame({"g": g.index.get_level_values(0), "na": g[("size", a)].values, "nb": g[("size", b)].values,
                       "ka": g[("sum", a)].values, "kb": g[("sum", b)].values})
    st = st[(st.na > 0) & (st.nb > 0)]
    if st.empty:
        return "not enough data"
    w = st.na * st.nb / (st.na + st.nb)
    dd = st.ka / st.na - st.kb / st.nb
    est = np.sum(w * dd) / np.sum(w)
    _, idx = np.unique(st.g, return_inverse=True)
    num, den = np.bincount(idx, weights=w * dd), np.bincount(idx, weights=w)
    draws = rng.integers(0, len(num), size=(reps, len(num)))
    lo, hi = np.percentile(num[draws].sum(1) / den[draws].sum(1), [2.5, 97.5])
    return (f"{est*scale:+.1f}{unit} [{lo*scale:+.1f}, {hi*scale:+.1f}]"
            f"  (issues={len(num)}, runs={int((st.na + st.nb).sum())})")


def forks(d):
    """Pairs of tries of the same issue whose action histories are identical up to
    the step where they first differ. Records what each did at that step."""
    rows = []
    for issue, g in d.groupby("issue"):
        if len(g) < 2:
            continue
        H, K, Y = list(g.hash30), list(g.kinds30), list(g.y)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                h1, h2 = list(H[i]), list(H[j])
                t = next((s for s in range(min(len(h1), len(h2))) if h1[s] != h2[s]), None)
                if t is not None:
                    rows.append((issue, t, K[i][t], K[j][t], Y[i], Y[j]))
    return pd.DataFrame(rows, columns=["issue", "step", "k1", "k2", "y1", "y2"])


def fork_effect(f, rng, a, b=None):
    """Among fork pairs where one try does `a` and the other does `b` (or anything
    other than `a` if b is None), outcome difference (a minus other)."""
    if b is None:
        m = (f.k1 == a) ^ (f.k2 == a)
    else:
        m = ((f.k1 == a) & (f.k2 == b)) | ((f.k1 == b) & (f.k2 == a))
    g = f[m]
    if g.empty:
        return "no such forks"
    ya = np.where(g.k1 == a, g.y1, g.y2)
    yo = np.where(g.k1 == a, g.y2, g.y1)
    diff = ya.mean() - yo.mean()
    issues = g.issue.values
    u = np.unique(issues)
    idx = {k: np.where(issues == k)[0] for k in u}
    boot = []
    for _ in range(2000):
        s = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
        boot.append(ya[s].mean() - yo[s].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (f"{diff*100:+.1f} pts [{lo*100:+.1f}, {hi*100:+.1f}]  (pairs={len(g)}, issues={len(u)},"
            f" median step={np.median(g.step):.0f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="analysis/exploratory/data/nvidia_runs.parquet")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    d = load(args.runs)
    print(f"tries={len(d)} issues={d.issue.nunique()} labeled={d.labeled.mean():.1%}")

    print("\n=== 1. How often does each AI write a test/repro script before fixing? ===")
    t = d.groupby("subset").arm.value_counts(normalize=True).unstack().round(3)
    t["tries"] = d.groupby("subset").size()
    print(t.to_string())
    print("\nlabel missing (resolved = -1) by arm:", d.groupby("arm").labeled.apply(lambda s: round(1 - s.mean(), 3)).to_dict())

    lab = d[d.labeled]
    print("\n=== 2. Test first minus fix first: simple comparison vs same-issue comparison ===")
    for s, g in list(lab.groupby("subset")) + [("ALL SUBSETS", lab)]:
        simple = g[g.arm == "test first"].y.mean() - g[g.arm == "fix first"].y.mean()
        print(f"{s:<45} simple {simple*100:+.1f} | same issue {within(g, 'arm', 'test first', 'fix first', rng=rng)}")

    print("\n=== 3. Near coin flips: same issue, identical action history, then a different next action ===")
    f = forks(lab)
    print(f"fork pairs={len(f)} in {f.issue.nunique()} issues; median fork step={f.step.median():.0f}")
    pairs = f.apply(lambda r: " vs ".join(sorted([r.k1, r.k2])), axis=1)
    print("most common choices at the fork:", pairs.value_counts().head(6).to_dict())
    print("test script now vs edit code now:", fork_effect(f, rng, "repro_create", "edit"))
    print("test script now vs anything else now:", fork_effect(f, rng, "repro_create"))

    print("\n=== 4. Same-issue comparison for 'ran code before the first edit' ===")
    print("ran code first minus edited before running:",
          within(lab, "ran_before_edit", "ran code first", "edited before running", rng=rng))

    print("\n=== 5. Does testing first cost extra steps? (same issue; tries that finished normally) ===")
    print("steps, test first minus fix first:",
          within(lab, "arm", "test first", "fix first", y="n_steps", rng=rng, scale=1, unit=" steps"))

    print("\n=== 6. Hints of removed tries (NVIDIA dropped tries that hit the step limit or had no patch) ===")
    k = d.groupby("issue").size()
    full = d.groupby("subset").apply(lambda g: g.groupby("issue").size().max())
    d["n_kept"] = d.issue.map(k)
    d["fewer_than_max"] = d.n_kept < d.subset.map(full)
    print("share of issues with fewer tries than the subset maximum:",
          round(d.groupby("issue").fewer_than_max.first().mean(), 3))
    print("test-first share | issues with all tries kept:",
          round((d[~d.fewer_than_max].arm == "test first").mean(), 3),
          "| issues missing some tries:", round((d[d.fewer_than_max].arm == "test first").mean(), 3))

    print("\n=== 7. Other tips after the fix (tries that edited code; same issue) ===")
    ed = lab[lab.first_edit.notna()].copy()
    tips = {
        "ran code after fixing": ed.n_runs_after_edit > 0,
        "ran the project's tests after fixing": ed.n_suite_after_edit > 0,
        "wrote extra tests after fixing": ed.n_tests_written_after_edit > 0,
        "cleaned up files (rm)": ed.n_cleanup > 0,
        "ran the project's tests before fixing": ed.n_suite_before_edit > 0,
    }
    for name, flag in tips.items():
        x = ed.assign(arm=np.where(flag, "yes", "no"))
        simple = x[x.arm == "yes"].y.mean() - x[x.arm == "no"].y.mean()
        print(f"{name:<40} share yes {flag.mean():.1%} | simple {simple*100:+.1f} | same issue "
              f"{within(x, 'arm', 'yes', 'no', rng=rng)}")

    print("\n=== 8. Cost of testing first (same issue; tries that finished normally) ===")
    for col, label, scale in [("out_chars", "text written by the AI (thousand chars)", 1e-3),
                              ("in_chars", "text read by the AI (thousand chars)", 1e-3)]:
        print(f"{label}: {within(lab, 'arm', 'test first', 'fix first', y=col, rng=rng, scale=scale, unit='')}")

    print("\n=== 9. Test first minus fix first by programming language (same issue, all AIs) ===")
    lang_names = lab.language.replace({"js": "javascript", "ts": "typescript"})
    for lang, g in lab.groupby(lang_names):
        if g.issue.nunique() >= 300:
            print(f"{lang:<12} test-first share {(g.arm == 'test first').mean():.0%} | "
                  f"{within(g, 'arm', 'test first', 'fix first', rng=rng)}")


if __name__ == "__main__":
    main()
