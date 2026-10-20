"""Step 3: SQL that turns each raw dataset into the shared ``tries`` and ``steps`` formats.

Each function takes the SQL list of local Parquet globs (``src``) and returns a
SELECT statement. Columns follow :mod:`agent_tips.cleaning.schema`, except the
derived flags and batches, which :mod:`agent_tips.cleaning.pipeline` adds.
"""

from __future__ import annotations

# The older AI writes its command in a fenced code block after its reasoning.
CMD_RE = r"(?s)```[a-z]*\n(.*?)\n?```"
# Timestamp logged by the evaluation harness just before running the tests.
TS_RE = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - root - INFO"
# The older AI's editor reports the open file in every observation.
OPEN_RE = r"\(Open file: ([^)]*)\)"
FIRST_WORD_RE = r"^\s*(\S+)"
NVIDIA_PATH_RE = r"data/(sweagent|openhands|minisweagent)/([^/]+)/([^/]+)/"

OLDER_TRY_ID = (
    r"'older_ai:' || regexp_extract(filename, 'train-(\d+)', 1) || ':' || file_row_number"
)


def older_ai_tries(src: str, model: str) -> str:
    """Return the tries query for nebius/SWE-agent-trajectories (``in_scope`` marks ``model``)."""
    return f"""
    SELECT {OLDER_TRY_ID} AS try_id,
      'older_ai' AS dataset, 'older_ai' AS subset, model_name AS ai,
      'swe-agent-2024' AS scaffold, instance_id AS issue_id,
      'older_ai::' || instance_id AS issue_key,
      CAST(target AS TINYINT) AS resolved, TRUE AS outcome_known, exit_status,
      CAST(len(list_filter(trajectory, e -> e.role = 'ai')) AS INTEGER) AS n_steps,
      length(coalesce(generated_patch, '')) > 0 AS has_patch,
      try_strptime(regexp_extract(eval_logs, '{TS_RE}', 1), '%Y-%m-%d %H:%M:%S') AS eval_ts,
      md5(coalesce(trajectory[2].text, '')) AS issue_hash,
      model_name = '{model}' AS in_scope,
      'python' AS language,
      CAST(coalesce(list_sum(list_transform(list_filter(trajectory, e -> e.role = 'ai'),
        e -> length(coalesce(e.text, '')))), 0) AS BIGINT) AS chars_out,
      CAST(coalesce(list_sum(list_transform(list_filter(trajectory, e -> e.role = 'user'),
        e -> length(coalesce(e.text, '')))), 0) AS BIGINT) AS chars_in
    FROM read_parquet({src}, filename = true, file_row_number = true)
    """


def older_ai_steps(src: str, model: str) -> str:
    """Return the steps query for the in-scope model of the older AI dataset."""
    return f"""
    WITH r AS (
      SELECT {OLDER_TRY_ID} AS try_id, trajectory
      FROM read_parquet({src}, filename = true, file_row_number = true)
      WHERE model_name = '{model}'),
    e AS (
      SELECT try_id, unnest(generate_series(1, len(trajectory))) AS pos,
             unnest(trajectory) AS ent FROM r),
    w AS (
      SELECT try_id, pos, ent.role AS role, coalesce(ent.text, '') AS text,
        lead(ent.text) OVER (PARTITION BY try_id ORDER BY pos) AS obs,
        lag(ent.text) OVER (PARTITION BY try_id ORDER BY pos) AS prev
      FROM e),
    a AS (
      SELECT *, trim(regexp_extract(text, '{CMD_RE}', 1)) AS cmd FROM w WHERE role = 'ai')
    SELECT try_id,
      CAST(row_number() OVER (PARTITION BY try_id ORDER BY pos) - 1 AS INTEGER) AS step,
      regexp_extract(cmd, '{FIRST_WORD_RE}', 1) AS tool,
      regexp_extract(cmd, '{FIRST_WORD_RE}', 1) AS action,
      left(cmd, 500) AS command,
      CASE WHEN regexp_extract(cmd, '{FIRST_WORD_RE}', 1) IN ('create', 'open')
           THEN split_part(regexp_replace(cmd, '\\s+', ' ', 'g'), ' ', 2) ELSE '' END AS path,
      regexp_extract(coalesce(prev, ''), '{OPEN_RE}', 1) AS open_file,
      md5(cmd) AS args_hash,
      CAST(1 AS INTEGER) AS n_calls,
      CAST(greatest(length(text) - length(cmd), 0) AS INTEGER) AS thought_len,
      CAST(length(coalesce(obs, '')) AS INTEGER) AS obs_len,
      left(coalesce(obs, ''), 300) AS obs_head
    FROM a
    """


def _tool_steps(try_id_sql: str, from_sql: str, messages: str, reasoning: bool) -> str:
    """Return the steps query for tool-calling formats (OpenHands, SWE-agent)."""
    thought = (
        "length(coalesce(m.content, '')) + length(coalesce(m.reasoning_content, ''))"
        if reasoning
        else "length(coalesce(m.content, ''))"
    )
    return f"""
    WITH r AS (SELECT {try_id_sql} AS try_id, {messages} AS msgs {from_sql}),
    e AS (SELECT try_id, unnest(generate_series(1, len(msgs))) AS pos, unnest(msgs) AS m FROM r),
    w AS (SELECT try_id, pos, m,
            lead(m.content) OVER (PARTITION BY try_id ORDER BY pos) AS obs FROM e),
    a AS (
      SELECT *, coalesce(m.tool_calls[1].function.name, 'none') AS fn,
             coalesce(m.tool_calls[1].function.arguments, '') AS args,
             -- some tool calls carry empty or malformed arguments
             CASE WHEN json_valid(m.tool_calls[1].function.arguments)
                  THEN m.tool_calls[1].function.arguments ELSE '{{}}' END AS aj
      FROM w WHERE m.role = 'assistant')
    SELECT try_id,
      CAST(row_number() OVER (PARTITION BY try_id ORDER BY pos) - 1 AS INTEGER) AS step,
      fn AS tool,
      CASE WHEN fn = 'str_replace_editor'
           THEN coalesce(json_extract_string(aj, '$.command'), '')
           ELSE regexp_extract(coalesce(json_extract_string(aj, '$.command'), ''),
                               '{FIRST_WORD_RE}', 1) END AS action,
      left(coalesce(json_extract_string(aj, '$.command'), args), 500) AS command,
      coalesce(json_extract_string(aj, '$.path'), '') AS path,
      '' AS open_file,
      md5(fn || '|' || args) AS args_hash,
      CAST(coalesce(len(m.tool_calls), 0) AS INTEGER) AS n_calls,
      CAST({thought} AS INTEGER) AS thought_len,
      CAST(length(coalesce(obs, '')) AS INTEGER) AS obs_len,
      left(coalesce(obs, ''), 300) AS obs_head
    FROM a
    """


def nebius_tries(src: str) -> str:
    """Return the tries query for nebius/SWE-rebench-openhands-trajectories."""
    return f"""
    SELECT 'newer_ai_nebius:' || trajectory_id AS try_id,
      'newer_ai_nebius' AS dataset, 'newer_ai_nebius' AS subset,
      'qwen3-coder-480b' AS ai, 'openhands-0.54' AS scaffold,
      instance_id AS issue_id, 'newer_ai_nebius::' || instance_id AS issue_key,
      CAST(resolved AS TINYINT) AS resolved, TRUE AS outcome_known, exit_status,
      CAST(len(list_filter(trajectory, m -> m.role = 'assistant')) AS INTEGER) AS n_steps,
      length(coalesce(model_patch, '')) > 0 AS has_patch,
      CAST(NULL AS TIMESTAMP) AS eval_ts,
      md5(coalesce(trajectory[2].content, '')) AS issue_hash,
      TRUE AS in_scope, 'python' AS language,
      CAST(coalesce(list_sum(list_transform(list_filter(trajectory, m -> m.role = 'assistant'),
        m -> length(coalesce(m.content, ''))
             + length(coalesce(m.tool_calls[1].function.arguments, '')))), 0) AS BIGINT) AS chars_out,
      CAST(coalesce(list_sum(list_transform(list_filter(trajectory, m -> m.role IN ('tool', 'user')),
        m -> length(coalesce(m.content, '')))), 0) AS BIGINT) AS chars_in
    FROM read_parquet({src})
    """


def nebius_steps(src: str) -> str:
    """Return the steps query for the Nebius OpenHands dataset."""
    return _tool_steps(
        "'newer_ai_nebius:' || trajectory_id",
        f"FROM read_parquet({src})",
        "trajectory",
        reasoning=False,
    )


def _nvidia_subset() -> str:
    f = r"replace(filename, '\', '/')"
    return (
        f"regexp_extract({f}, '{NVIDIA_PATH_RE}', 1) || '/' || "
        f"regexp_extract({f}, '{NVIDIA_PATH_RE}', 2) || '/' || "
        f"regexp_extract({f}, '{NVIDIA_PATH_RE}', 3)"
    )


def nvidia_tries(src: str) -> str:
    """Return the tries query for the configured nvidia/Open-SWE-Traces subsets."""
    sub = _nvidia_subset()
    f = r"replace(filename, '\', '/')"
    return f"""
    SELECT 'newer_ai_nvidia:' || {sub} || ':' || trajectory_id AS try_id,
      'newer_ai_nvidia' AS dataset, {sub} AS subset,
      regexp_extract({f}, '{NVIDIA_PATH_RE}', 2) AS ai,
      regexp_extract({f}, '{NVIDIA_PATH_RE}', 1) AS scaffold,
      instance_id AS issue_id, {sub} || '::' || instance_id AS issue_key,
      CAST(CASE WHEN resolved IN (0, 1) THEN resolved END AS TINYINT) AS resolved,
      resolved IN (0, 1) AS outcome_known, '' AS exit_status,
      CAST(len(list_filter(messages, m -> m.role = 'assistant')) AS INTEGER) AS n_steps,
      length(coalesce(metadata.model_patch.patch, '')) > 0 AS has_patch,
      CAST(NULL AS TIMESTAMP) AS eval_ts,
      md5(coalesce(list_filter(messages, m -> m.role = 'user')[1].content, '')) AS issue_hash,
      TRUE AS in_scope,
      CASE language WHEN 'js' THEN 'javascript' WHEN 'ts' THEN 'typescript'
                    ELSE language END AS language,
      CAST(coalesce(list_sum(list_transform(list_filter(messages, m -> m.role = 'assistant'),
        m -> length(coalesce(m.content, '')) + length(coalesce(m.reasoning_content, ''))
             + length(coalesce(m.tool_calls[1].function.arguments, '')))), 0) AS BIGINT) AS chars_out,
      CAST(coalesce(list_sum(list_transform(list_filter(messages, m -> m.role IN ('tool', 'user')),
        m -> length(coalesce(m.content, '')))), 0) AS BIGINT) AS chars_in
    FROM read_parquet({src}, filename = true, union_by_name = true)
    """


def nvidia_steps(src: str) -> str:
    """Return the steps query for the configured NVIDIA subsets."""
    return _tool_steps(
        f"'newer_ai_nvidia:' || {_nvidia_subset()} || ':' || trajectory_id",
        f"FROM read_parquet({src}, filename = true, union_by_name = true)",
        "messages",
        reasoning=True,
    )
