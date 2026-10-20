"""Step 3: the shared table formats produced by cleaning.

Every dataset is turned into two tables with identical columns:

- ``tries``: one row per attempt by an AI at an issue.
- ``steps``: one row per action the AI took within a try.

Rows are never dropped during cleaning. Instead, flag columns (``in_scope``,
``same_input``, ``outcome_known``, ``dated``) say which rows each analysis
uses, and the cleaning funnel counts them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    """One column of a clean table."""

    name: str
    dtype: str
    description: str


TRIES: tuple[Column, ...] = (
    Column("try_id", "string", "Unique id: `<dataset>:<source-specific id>`"),
    Column("dataset", "string", "`older_ai`, `newer_ai_nebius` or `newer_ai_nvidia`"),
    Column("subset", "string", "Group with one AI configuration (NVIDIA: scaffold/AI/task source)"),
    Column("ai", "string", "Model that generated the try"),
    Column("scaffold", "string", "Agent framework (SWE-agent 2024, OpenHands, SWE-agent)"),
    Column("issue_id", "string", "Issue (task) identifier from the source"),
    Column("issue_key", "string", "`subset::issue_id` - the issue within one AI configuration"),
    Column("resolved", "int8", "1 if hidden tests pass, 0 if not, null if unknown"),
    Column("outcome_known", "bool", "False only for NVIDIA tries with resolved = -1"),
    Column("exit_status", "string", "How the try ended, as recorded by the source"),
    Column("n_steps", "int32", "Number of AI actions (assistant turns)"),
    Column("has_patch", "bool", "The try produced a non-empty patch"),
    Column("eval_ts", "timestamp", "Evaluation time from the eval log (older AI only)"),
    Column("dated", "bool", "eval_ts is known (older AI); true elsewhere"),
    Column("issue_hash", "string", "MD5 of the issue text the AI received"),
    Column("same_input", "bool", "issue_hash equals the most common text for this issue_key"),
    Column("in_scope", "bool", "Row belongs to an AI analysed in the project"),
    Column("language", "string", "Programming language of the repository"),
    Column("chars_out", "int64", "Characters written by the AI (thoughts, reasoning, tool calls)"),
    Column("chars_in", "int64", "Characters the AI read (observations and tool outputs)"),
    Column("batch_day", "timestamp", "Older AI: evaluation day (model-version proxy)"),
    Column("batch_6h", "timestamp", "Older AI: 6-hour evaluation window"),
    Column("batch_hour", "timestamp", "Older AI: evaluation hour"),
    Column("batch_gap60", "int32", "Older AI: evaluation burst (new batch after a 60-min pause)"),
    Column("batch_gap10", "int32", "Older AI: evaluation burst (new batch after a 10-min pause)"),
)

STEPS: tuple[Column, ...] = (
    Column("try_id", "string", "Links to `tries.try_id`"),
    Column("step", "int32", "0-based position of the AI action within the try"),
    Column("tool", "string", "Tool or command name (e.g. `create`, `str_replace_editor`, `bash`)"),
    Column("action", "string", "Editor sub-command or first word of a shell command"),
    Column("command", "string", "Command text, truncated to 500 characters"),
    Column("path", "string", "File path the action refers to, if any"),
    Column("open_file", "string", "Older AI: file open in the editor before this action"),
    Column("args_hash", "string", "MD5 of the full tool call (to find identical histories)"),
    Column("n_calls", "int32", "Number of tool calls in this turn (only the first is described)"),
    Column("thought_len", "int32", "Characters of reasoning text in this turn"),
    Column("obs_len", "int32", "Characters of the environment's reply"),
    Column("obs_head", "string", "First 300 characters of the environment's reply"),
    Column("obs_tail", "string", "Last 300 characters of the environment's reply (test summaries)"),
)


def markdown_dictionary(columns: tuple[Column, ...]) -> str:
    """Render a data dictionary as a Markdown table."""
    lines = ["| Column | Type | Description |", "| --- | --- | --- |"]
    lines += [f"| `{c.name}` | {c.dtype} | {c.description} |" for c in columns]
    return "\n".join(lines)


def names(columns: tuple[Column, ...]) -> list[str]:
    """Column names in order."""
    return [c.name for c in columns]
