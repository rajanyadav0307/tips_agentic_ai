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
    return [
        {
            "trajectory_id": f"t{i}",
            "instance_id": "o__p-1",
            "repo": "o/p",
            "trajectory": [msg],
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
