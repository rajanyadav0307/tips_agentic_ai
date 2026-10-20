"""Shared fixtures: a tiny fake project with raw files in each real format."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agent_tips.config import Config, load_config

TOOL_CALL = {"function": {"arguments": "{}", "name": "x"}, "id": "c0", "type": "function"}


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _older_ai_rows() -> list[dict[str, Any]]:
    rows = []
    for i, (inst, target) in enumerate([("a__b-1", True), ("a__b-1", False), ("c__d-2", False)]):
        rows.append(
            {
                "instance_id": inst,
                "model_name": "swe-agent-llama-70b",
                "target": target,
                "trajectory": [
                    {
                        "cutoff_date": "01.01.2023",
                        "mask": False,
                        "role": "system",
                        "system_prompt": "SETTING",
                        "text": "",
                    },
                    {
                        "cutoff_date": "",
                        "mask": False,
                        "role": "user",
                        "system_prompt": "",
                        "text": "ISSUE:\nbug\n\nINSTRUCTIONS:\n...",
                    },
                    {
                        "cutoff_date": "",
                        "mask": True,
                        "role": "ai",
                        "system_prompt": "",
                        "text": "Reproduce it.\n```\ncreate reproduce.py\n```",
                    },
                    {
                        "cutoff_date": "",
                        "mask": False,
                        "role": "user",
                        "system_prompt": "",
                        "text": "[File: /r/reproduce.py (1 lines total)]\n(Open file: /r/reproduce.py)",
                    },
                    {
                        "cutoff_date": "",
                        "mask": True,
                        "role": "ai",
                        "system_prompt": "",
                        "text": "Edit it.\n```\nedit 1:1\nprint(1)\nend_of_edit\n```",
                    },
                ],
                "exit_status": "submitted",
                "generated_patch": "diff" if target else "",
                "eval_logs": f"2024-10-0{i + 1} 10:00:00,000 - root - INFO - run",
            }
        )
    return rows


def _nebius_rows() -> list[dict[str, Any]]:
    msg = {
        "content": "hi",
        "name": "",
        "role": "assistant",
        "tool_call_id": "",
        "tool_calls": [TOOL_CALL],
    }
    bad = {**msg, "tool_calls": [{**TOOL_CALL, "function": {"arguments": "", "name": "finish"}}]}
    return [
        {
            "trajectory_id": f"t{i}",
            "instance_id": "o__p-1",
            "repo": "o/p",
            "trajectory": [msg, bad],
            "model_patch": "diff",
            "exit_status": "submit",
            "resolved": i % 2,
        }
        for i in range(2)
    ]


def _nvidia_rows() -> list[dict[str, Any]]:
    patch = {"patch": "diff", "num_modified_files": 1, "num_modified_lines": 2}
    meta = {
        "category": "bug",
        "teacher_model": {"name": "m", "enable_thinking": True, "reasoning_effort": "high"},
        "reference_patch": patch,
        "model_patch": patch,
    }
    msg = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [TOOL_CALL]}
    return [
        {
            "instance_id": "q__r-1",
            "repo": "q/r",
            "license": "MIT",
            "language": "python",
            "trajectory_id": f"n{i}",
            "messages": [msg],
            "tools": ["x"],
            "resolved": r,
            "metadata": meta,
            "hf_dataset_name": "x",
        }
        for i, r in enumerate([1, 0, -1])
    ]


@pytest.fixture()
def fake_project(tmp_path: Path) -> Config:
    """A project root in a temp folder with tiny raw files for every dataset."""
    raw_cfg = copy.deepcopy(load_config().raw)
    ds = raw_cfg["datasets"]
    ds["newer_ai_nvidia"]["patterns"] = ["data/sweagent/minimax_m25/swe-rebench-v2/*.parquet"]
    raw = tmp_path / "data" / "raw"
    _write(raw / "older_ai" / "data" / "train-00000-of-00001.parquet", _older_ai_rows())
    _write(raw / "newer_ai_nebius" / "trajectories.parquet", _nebius_rows())
    _write(
        raw / "newer_ai_nvidia" / "data/sweagent/minimax_m25/swe-rebench-v2/train-0.parquet",
        _nvidia_rows(),
    )
    gold = [{"instance_id": "a__b-1", "patch": "diff", "created_at": "2024-01-01"}]
    _write(raw / "gold_extra" / "data" / "train-00000-of-00001.parquet", gold)
    _write(
        raw / "gold_dev" / "data" / "dev-00000-of-00001.parquet",
        [{"instance_id": "c__d-2", "patch": "diff", "created_at": "2023-01-01"}],
    )
    return Config(root=tmp_path, raw=raw_cfg)


# ---------------------------------------------------------------- a larger fake project
# Same raw formats, but enough issues and repeated tries for every comparison in
# Steps 6-8 to exist. Outcomes are random, with known within-issue structure.

GOLD = "diff --git a/src/f.py b/src/f.py\n"


def _call(name: str, args: dict[str, Any], i: int) -> dict[str, Any]:
    import json

    return {
        "function": {"arguments": json.dumps(args), "name": name},
        "id": f"c{i}",
        "type": "function",
    }


def _older_big(rng: Any, n_issues: int = 60, tries: int = 8) -> list[dict[str, Any]]:
    def ai(cmd: str) -> dict[str, Any]:
        return {
            "cutoff_date": "",
            "mask": True,
            "role": "ai",
            "system_prompt": "",
            "text": f"Next.\n```\n{cmd}\n```",
        }

    def obs(text: str) -> dict[str, Any]:
        return {"cutoff_date": "", "mask": False, "role": "user", "system_prompt": "", "text": text}

    rows = []
    for i in range(n_issues):
        base = rng.uniform(0.1, 0.7)
        for k in range(tries):
            repro = bool(rng.random() < 0.5)
            first = (
                rng.choice(["create reproduce.py", "create reproduce_bug.py"])
                if repro
                else rng.choice(["search_dir foo", "find_file f.py", "open src/f.py"])
            )
            steps = [
                {
                    "cutoff_date": "01.01.2023",
                    "mask": False,
                    "role": "system",
                    "system_prompt": "SETTING",
                    "text": "",
                },
                obs(f"ISSUE:\nbug {i}\n\nINSTRUCTIONS:\n..."),
                ai(first),
                obs("[File: /r/reproduce.py (1 lines total)]\n(Open file: /r/x.py)"),
                ai("open src/f.py"),
                obs("(Open file: /r/src/f.py)"),
                ai("edit 1:1\nprint(1)\nend_of_edit"),
                obs("File updated.\n(Open file: /r/src/f.py)"),
            ]
            if rng.random() < 0.5:
                steps += [ai("python reproduce.py"), obs("ok\n(Open file: /r/src/f.py)")]
            if rng.random() < 0.3:
                steps += [ai("pytest tests/"), obs("==== 1 passed ====")]
            steps += [ai("submit"), obs("submitted")]
            p = base - (0.05 if repro else 0.0)
            dated = rng.random() < 0.9
            day = 1 + (k % 2)
            rows.append(
                {
                    "instance_id": f"o__r-{i}",
                    "model_name": "swe-agent-llama-70b",
                    "target": bool(rng.random() < p) if dated else False,
                    "trajectory": steps,
                    "exit_status": "submitted (exit_context)"
                    if rng.random() < 0.2
                    else "submitted",
                    "generated_patch": "diff",
                    "eval_logs": (
                        f"2024-10-0{day} {rng.integers(0, 24):02d}:00:00,000 - root - INFO - run"
                        if dated
                        else "no time"
                    ),
                }
            )
    return rows


def _tool_try(rng: Any, test_first: bool, bash: str) -> list[dict[str, Any]]:
    """An OpenHands / SWE-agent style message list (system, issue, actions, tool replies)."""
    calls = [("str_replace_editor", {"command": "view", "path": "/testbed"})]
    create = ("str_replace_editor", {"command": "create", "path": "/testbed/reproduce_issue.py"})
    fix = ("str_replace_editor", {"command": "str_replace", "path": "/testbed/src/f.py"})
    calls += [create, fix] if test_first else [fix, create]
    if rng.random() < 0.5:
        calls.append((bash, {"command": "python /testbed/reproduce_issue.py"}))
    if rng.random() < 0.3:
        calls.append((bash, {"command": "rm /testbed/reproduce_issue.py"}))
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "SYS", "reasoning_content": "", "tool_calls": None},
        {"role": "user", "content": "ISSUE", "reasoning_content": "", "tool_calls": None},
    ]
    for j, (name, args) in enumerate(calls):
        msgs.append(
            {
                "role": "assistant",
                "content": "x",
                "reasoning_content": "",
                "tool_calls": [_call(name, args, j)],
            }
        )
        reply = "File created successfully" if args.get("command") == "create" else "ok"
        msgs.append({"role": "tool", "content": reply, "reasoning_content": "", "tool_calls": None})
    return msgs


def _nvidia_big(rng: Any, n_issues: int, bash: str) -> list[dict[str, Any]]:
    patch = {"patch": GOLD, "num_modified_files": 1, "num_modified_lines": 2}
    meta = {
        "category": "bug",
        "teacher_model": {"name": "m", "enable_thinking": True, "reasoning_effort": "high"},
        "reference_patch": patch,
        "model_patch": patch,
    }
    rows = []
    for i in range(n_issues):
        base = rng.uniform(0.2, 0.8)
        for k in range(4):
            tf = bool(rng.random() < 0.6)
            rows.append(
                {
                    "instance_id": f"q__r-{i}",
                    "repo": "q/r",
                    "license": "MIT",
                    "language": "python" if i % 3 else "go",
                    "trajectory_id": f"n{i}-{k}",
                    "messages": _tool_try(rng, tf, bash),
                    "tools": ["x"],
                    "resolved": int(rng.random() < base),
                    "metadata": meta,
                    "hf_dataset_name": "x",
                }
            )
    return rows


def _nebius_big(rng: Any, n_issues: int = 30) -> list[dict[str, Any]]:
    rows = []
    for i in range(n_issues):
        for k in range(4):
            msgs = _tool_try(rng, True, "execute_bash")
            if rng.random() < 0.5:  # first move: think vs view
                msgs[2]["tool_calls"] = [_call("think", {"thought": "hmm"}, 0)]
            for m in msgs:
                m.pop("reasoning_content")
            rows.append(
                {
                    "trajectory_id": f"t{i}-{k}",
                    "instance_id": f"o__p-{i}",
                    "repo": "o/p",
                    "trajectory": msgs,
                    "model_patch": "diff",
                    "exit_status": "submit"
                    if rng.random() < 0.9
                    else "RuntimeError: Agent reached maximum iteration",
                    "resolved": int(rng.random() < 0.5),
                }
            )
    return rows


@pytest.fixture(scope="session")
def big_project(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """A fake project large enough for the whole pipeline (Steps 3-8)."""
    import numpy as np

    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("big")
    raw_cfg = copy.deepcopy(load_config().raw)
    subsets = ["sweagent/minimax_m25/swe-rebench-v2", "openhands/minimax_m25/swe-rebench-v2"]
    raw_cfg["datasets"]["newer_ai_nvidia"]["patterns"] = [f"data/{s}/*.parquet" for s in subsets]
    raw_cfg["analysis"].update({"bootstrap_reps": 50, "permutation_reps": 20})
    raw = root / "data" / "raw"
    older = _older_big(rng)
    _write(raw / "older_ai" / "data" / "train-00000-of-00001.parquet", older)
    _write(raw / "newer_ai_nebius" / "trajectories.parquet", _nebius_big(rng))
    for s in subsets:
        bash = "execute_bash" if s.startswith("openhands") else "bash"
        _write(
            raw / "newer_ai_nvidia" / "data" / s / "train-0.parquet", _nvidia_big(rng, 120, bash)
        )
    gold = [
        {"instance_id": iid, "patch": GOLD, "created_at": "2024-01-01"}
        for iid in sorted({r["instance_id"] for r in older})
    ]
    _write(raw / "gold_extra" / "data" / "train-00000-of-00001.parquet", gold)
    return Config(root=root, raw=raw_cfg)
