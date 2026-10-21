"""Candidate comparisons of the one-try study, exactly as frozen in Step 1.

A comparison is one behaviour (two choices *a* vs *b*) for one studied AI
setup. Its **key bugs** are the bugs where both choices occur among the clean
tries; for the older AI the groups are bug × evaluation day (hidden model
versions) and only dated tries count. A comparison qualifies when it has at
least ``min_bugs`` key bugs (Step 4 lists them; Step 6 uses that list).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

OLDER, NEBIUS, NVIDIA = "older_ai", "newer_ai_nebius", "newer_ai_nvidia"
AFTER_EDIT = [
    ("rerun_after_fix", "Check your fix (re-ran own test script after the first code edit)"),
    ("suite_after_fix", "Ran the project's test suite after the first code edit"),
    ("run_after_fix", "Ran any code after the first code edit"),
    ("tests_after_fix", "Wrote extra tests after the first code edit"),
    ("suite_before_fix", "Ran the project's test suite before the first code edit"),
]


@dataclass(frozen=True)
class Comparison:
    """One behaviour × one studied AI setup."""

    cid: str
    dataset: str
    setup: str
    behaviour: str
    column: str
    a: str
    b: str
    experiment: bool
    by_day: bool = False  # older AI: groups are bug × evaluation day
    edited_only: bool = False  # only tries that edited code (after-fix behaviours)

    def rows(self, tries: pd.DataFrame) -> pd.DataFrame:
        """Return the tries of this setup with choice a or b (and the comparison's filters)."""
        d = tries[(tries.setup == self.setup) & tries[self.column].isin([self.a, self.b])]
        if self.by_day:
            d = d[d.eval_day.notna()]
        if self.edited_only:
            d = d[d["order"] != "no edit"]
        return d

    def groups(self, d: pd.DataFrame) -> pd.Series:
        """Return the group key of every try: the bug, or bug × day for the older AI."""
        if self.by_day:
            return d.bug_id + "|" + d.eval_day.astype(str)
        return d.bug_id


def candidates(tries: pd.DataFrame, studied_older: str) -> list[Comparison]:
    """All candidate comparisons from the frozen list (before the size filter)."""
    out: list[Comparison] = [
        Comparison(
            f"{OLDER}|reproduce_first",
            OLDER,
            studied_older,
            "Reproduce first (create) vs search first",
            "first_move_older",
            "R",
            "S",
            experiment=True,
            by_day=True,
        ),
    ]
    for tool in ("find_file", "ls", "open", "grep"):
        out.append(
            Comparison(
                f"{OLDER}|first_{tool}",
                OLDER,
                studied_older,
                f"First move {tool} vs search_dir",
                "first_tool_older",
                tool,
                "search_dir",
                experiment=True,
                by_day=True,
            )
        )
    for col, label in AFTER_EDIT[:2]:
        out.append(
            Comparison(
                f"{OLDER}|{col}",
                OLDER,
                studied_older,
                label,
                col,
                "yes",
                "no",
                experiment=False,
                by_day=True,
                edited_only=True,
            )
        )
    out.append(
        Comparison(
            f"{NEBIUS}|think_first",
            NEBIUS,
            NEBIUS,
            "First move think vs view",
            "first_move_nebius",
            "think",
            "view",
            experiment=True,
        )
    )
    nv = tries[(tries.dataset == NVIDIA) & tries.studied]
    for setup, g in sorted(nv.groupby("setup"), key=lambda kv: str(kv[0])):
        top = g.first_move_nvidia.value_counts()
        if len(top) >= 2:
            a, b = str(top.index[1]), str(top.index[0])
            out.append(
                Comparison(
                    f"{setup}|first_move",
                    NVIDIA,
                    str(setup),
                    f"First move {a} vs {b}",
                    "first_move_nvidia",
                    a,
                    b,
                    experiment=True,
                )
            )
        out.append(
            Comparison(
                f"{setup}|test_first",
                NVIDIA,
                str(setup),
                "Test first vs fix first",
                "order",
                "test first",
                "fix first",
                experiment=False,
            )
        )
        for col, label in [*AFTER_EDIT, ("cleanup", "Cleaned up created files")]:
            out.append(
                Comparison(
                    f"{setup}|{col}",
                    NVIDIA,
                    str(setup),
                    label,
                    col,
                    "yes",
                    "no",
                    experiment=False,
                    edited_only=True,
                )
            )
    return out


def key_counts(c: Comparison, tries: pd.DataFrame) -> dict[str, float]:
    """How many tries and key bugs a comparison has (counts only, no outcomes)."""
    d = c.rows(tries)
    if d.empty:
        return {
            "tries_a": 0,
            "tries_b": 0,
            "key_groups": 0,
            "key_bugs": 0,
            "key_tries": 0,
            "share_a": 0.0,
        }
    grp = c.groups(d)
    both = d.assign(g=grp).groupby("g")[c.column].nunique()
    key = set(both[both == 2].index)
    kd = d[grp.isin(key)]
    return {
        "tries_a": int((d[c.column] == c.a).sum()),
        "tries_b": int((d[c.column] == c.b).sum()),
        "key_groups": len(key),
        "key_bugs": int(kd.bug_id.nunique()),
        "key_tries": len(kd),
        "share_a": float((d[c.column] == c.a).mean()),
    }
