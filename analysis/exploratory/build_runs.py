"""Build a run-level table for the swe-agent-llama-70b runs.

One row per run: whether the issue was resolved, the evaluation timestamp
(our clue for which model version produced the run), the issue-text hash
(to keep only runs that saw identical input), and the agent's commands.

Usage (from the project root):
    python -I analysis/exploratory/build_runs.py --data data/SWE-agent-trajectories/data \
        --out analysis/data/runs_70b.parquet
"""
import argparse
import os

import duckdb

# The agent writes its command in a fenced code block after its reasoning.
CMD_RE = r"(?s)```[a-z]*\n(.*?)\n?```"
# Timestamp the evaluation harness logs right before running the tests.
TS_RE = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - root - INFO"
CREATE_REPRO_RE = r"^create\s+\S*repro"
RUN_REPRO_RE = r"^(python3?|bash|sh)\s+\S*repro"
ANY_CREATE_RE = r"^create\s"
RUN_SCRIPT_RE = r"^(python3?|bash|sh)\s+\S+"
TEST_SUITE_RE = r"(pytest|python3? -m (pytest|unittest)|tox\b)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/SWE-agent-trajectories/data")
    p.add_argument("--out", default="analysis/exploratory/data/runs_70b.parquet")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    con = duckdb.connect()
    con.execute(f"""
    CREATE TABLE runs AS
    SELECT
      regexp_extract(filename, 'train-(\\d+)', 1) || ':' || file_row_number AS run_id,
      instance_id,
      target::INT AS resolved,
      exit_status,
      len(trajectory) AS n_entries,
      length(coalesce(generated_patch, '')) > 0 AS has_patch,
      try_strptime(regexp_extract(eval_logs, '{TS_RE}', 1), '%Y-%m-%d %H:%M:%S') AS eval_ts,
      md5(coalesce(trajectory[2].text, '')) AS issue_hash,
      list_transform(list_filter(trajectory, e -> e.role = 'ai'),
                     e -> trim(regexp_extract(e.text, '{CMD_RE}', 1))) AS cmds
    FROM read_parquet('{args.data}/*.parquet', filename = true, file_row_number = true)
    WHERE model_name = 'swe-agent-llama-70b'
    """)
    con.execute(f"""
    COPY (
      SELECT run_id, instance_id, resolved, exit_status, n_entries, has_patch, eval_ts, issue_hash,
        len(cmds) AS n_commands,
        cmds[1] AS cmd0, cmds[2] AS cmd1, cmds[3] AS cmd2, cmds[4] AS cmd3, cmds[5] AS cmd4,
        list_min(list_filter(list_transform(cmds, (c, i) -> CASE WHEN regexp_matches(c, '{CREATE_REPRO_RE}') THEN i END),
                             x -> x IS NOT NULL)) - 1 AS first_create_repro_step,
        list_min(list_filter(list_transform(cmds, (c, i) -> CASE WHEN regexp_matches(c, '{RUN_REPRO_RE}') THEN i END),
                             x -> x IS NOT NULL)) - 1 AS first_run_repro_step,
        len(list_filter(cmds[2:], c -> regexp_matches(c, '{ANY_CREATE_RE}'))) > 0 AS later_create_any,
        len(list_filter(cmds, c -> regexp_matches(c, '{RUN_SCRIPT_RE}'))) > 0 AS runs_a_script,
        len(list_filter(cmds, c -> regexp_matches(c, '{TEST_SUITE_RE}'))) > 0 AS runs_test_suite
      FROM runs
    ) TO '{args.out}' (FORMAT parquet)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{args.out}')").fetchone()[0]
    print(f"wrote {n} runs to {args.out}")


if __name__ == "__main__":
    main()
