"""Build a run-level table from NVIDIA's Open-SWE-Traces (streamed from Hugging Face).

For each try: issue, outcome, and when the agent first (a) created a
test/reproduction file, (b) edited an existing project file, (c) ran code.
Also the kinds and hashes of its first 30 actions (to find tries of the same
issue with identical action histories).

Only subsets that use the str_replace_editor tool (SWE-agent, OpenHands) and
have repeated tries per issue are included by default.

Usage (from the project root):
    python -I analysis/exploratory/build_nvidia_runs.py --out analysis/data/nvidia_runs.parquet
"""
import argparse
import os

import duckdb

BASE = "hf://datasets/nvidia/Open-SWE-Traces/data"
SUBSETS = [
    "sweagent/minimax_m25/swe-rebench-v2",
    "sweagent/qwen35_122b/swe-rebench-v2",
    "sweagent/qwen36_27b/scale-swe",
    "sweagent/qwen36_27b/swe-rebench-v2",
    "openhands/minimax_m25/swe-rebench-v2",
    "openhands/qwen35_122b/swe-rebench-v2",
    "openhands/deepseek_v4_flash_0731/scale-swe",
]
# A new file whose name suggests a test or reproduction script.
REPRO_NAME = r"(repro|test|debug|issue|bug|verify|check|demo|example|scratch|/tmp/|try_|poc|sample)"
HEREDOC = r"<<-?\s*['\"]?[A-Za-z_]+"
REDIRECT_FILE = r">\s*['\"]?([\w./~-]+\.[A-Za-z0-9]+)"
BASH_EDIT = r"(sed\s+-i|perl\s+-[a-z]*i|git\s+apply|patch\s+-p)"
# Running the project's own test suite (checked before the generic RUN pattern).
TEST_SUITE = (r"(pytest|python3?\s+-m\s+(pytest|unittest)|go\s+test|mvn\s+(\S+\s+)*test|gradlew?\s+(\S+\s+)*test|"
              r"npm\s+(run\s+)?test|yarn\s+(run\s+)?test|pnpm\s+(run\s+)?test|cargo\s+test|phpunit|rspec|tox\b|"
              r"dotnet\s+test|make\s+(\S+\s+)*test|jest|vitest|mocha|ctest)")
CLEANUP = r"(^|[;&|]\s*|\s)rm\s+(-\S+\s+)*\S"
RUN = (r"(^|[;&|(]\s*|\s)(pytest|python3?|go\s+(test|run|build|vet)|mvn|gradle|\./gradlew|npm\s+(test|run)|npx|yarn|"
       r"pnpm|cargo\s+(test|run|build|check)|node|ruby|rspec|bundle\s+exec|phpunit|php|dotnet|make|tox|java|javac|"
       r"gcc|g\+\+|clang|julia|Rscript|bash\s+\S+\.sh|sh\s+\S+\.sh)\b")


def sql_for(path, subset):
    # Quote marks inside the regex patterns must be doubled inside SQL strings.
    REPRO_NAME, HEREDOC, REDIRECT_FILE, BASH_EDIT, RUN, TEST_SUITE, CLEANUP = (
        x.replace("'", "''") for x in (globals()[n] for n in
                                       ("REPRO_NAME", "HEREDOC", "REDIRECT_FILE", "BASH_EDIT", "RUN", "TEST_SUITE", "CLEANUP")))
    return f"""
    WITH a AS (
      SELECT instance_id, trajectory_id, resolved, language,
        list_sum(list_transform(list_filter(messages, m -> m.role = 'assistant'),
          m -> length(coalesce(m.content, '')) + length(coalesce(m.reasoning_content, ''))
               + length(coalesce(m.tool_calls[1].function.arguments, '')))) AS out_chars,
        list_sum(list_transform(list_filter(messages, m -> m.role IN ('tool', 'user')),
          m -> length(coalesce(m.content, '')))) AS in_chars,
        list_transform(list_filter(messages, m -> m.role = 'assistant'), m -> struct_pack(
          fn := coalesce(m.tool_calls[1].function.name, 'none'),
          args := coalesce(m.tool_calls[1].function.arguments, ''),
          cmd := coalesce(json_extract_string(m.tool_calls[1].function.arguments, '$.command'), ''),
          path := coalesce(json_extract_string(m.tool_calls[1].function.arguments, '$.path'), ''))) AS raw
      FROM read_parquet('{path}')),
    k AS (
      SELECT instance_id, trajectory_id, resolved, language, out_chars, in_chars,
        list_transform(raw, s -> struct_pack(fn := s.fn, args := s.args, path := s.path, kind :=
          CASE
            WHEN s.fn = 'str_replace_editor' AND s.cmd = 'create' AND regexp_matches(lower(s.path), '{REPRO_NAME}') THEN 'repro_create'
            WHEN s.fn = 'str_replace_editor' AND s.cmd = 'create' THEN 'create_other'
            WHEN s.fn = 'str_replace_editor' AND s.cmd IN ('str_replace', 'insert', 'undo_edit') THEN 'edit'
            WHEN s.fn = 'str_replace_editor' THEN 'view'
            WHEN s.fn IN ('bash', 'execute_bash') AND regexp_matches(s.cmd, '{HEREDOC}') AND regexp_matches(s.cmd, '{REDIRECT_FILE}')
                 AND regexp_matches(lower(regexp_extract(s.cmd, '{REDIRECT_FILE}', 1)), '{REPRO_NAME}') THEN 'repro_create'
            WHEN s.fn IN ('bash', 'execute_bash') AND regexp_matches(s.cmd, '{HEREDOC}') AND regexp_matches(s.cmd, '{REDIRECT_FILE}') THEN 'create_other'
            WHEN s.fn IN ('bash', 'execute_bash') AND regexp_matches(s.cmd, '{BASH_EDIT}') THEN 'edit'
            WHEN s.fn IN ('bash', 'execute_bash') AND regexp_matches(s.cmd, '{TEST_SUITE}') THEN 'test_suite'
            WHEN s.fn IN ('bash', 'execute_bash') AND regexp_matches(s.cmd, '{RUN}') THEN 'run'
            WHEN s.fn IN ('bash', 'execute_bash') AND regexp_matches(s.cmd, '{CLEANUP}') THEN 'cleanup'
            WHEN s.fn IN ('bash', 'execute_bash') THEN 'look'
            WHEN s.fn = 'think' THEN 'think'
            WHEN s.fn IN ('submit', 'finish') THEN 'submit'
            ELSE 'other' END)) AS steps
      FROM a),
    c AS (
      SELECT *, list_transform(list_filter(steps, s -> s.kind IN ('repro_create', 'create_other') AND s.path <> ''),
                               s -> s.path) AS created
      FROM k),
    e AS (
      SELECT *, list_min(list_filter(list_transform(steps, (s, i) -> CASE WHEN s.kind = 'edit'
                AND NOT list_contains(created, s.path) THEN i END), v -> v IS NOT NULL)) AS fe1
      FROM c)
    SELECT '{subset}' AS subset, instance_id, trajectory_id, resolved, language, out_chars, in_chars,
      len(steps) AS n_steps,
      list_min(list_filter(list_transform(steps, (s, i) -> CASE WHEN s.kind = 'repro_create' THEN i END), v -> v IS NOT NULL)) - 1 AS first_repro,
      fe1 - 1 AS first_edit,
      list_min(list_filter(list_transform(steps, (s, i) -> CASE WHEN s.kind IN ('run', 'test_suite') THEN i END),
                           v -> v IS NOT NULL)) - 1 AS first_run,
      len(list_filter(steps, s -> s.kind = 'repro_create')) AS n_repro_creates,
      len(list_filter(steps, s -> s.kind IN ('run', 'test_suite'))) AS n_runs,
      len(list_filter(list_transform(steps, (s, i) -> CASE WHEN i > fe1 AND s.kind IN ('run', 'test_suite') THEN 1 END),
                      v -> v IS NOT NULL)) AS n_runs_after_edit,
      len(list_filter(list_transform(steps, (s, i) -> CASE WHEN i > fe1 AND s.kind = 'test_suite' THEN 1 END),
                      v -> v IS NOT NULL)) AS n_suite_after_edit,
      len(list_filter(list_transform(steps, (s, i) -> CASE WHEN i < fe1 AND s.kind = 'test_suite' THEN 1 END),
                      v -> v IS NOT NULL)) AS n_suite_before_edit,
      len(list_filter(list_transform(steps, (s, i) -> CASE WHEN i > fe1 AND s.kind = 'repro_create' THEN 1 END),
                      v -> v IS NOT NULL)) AS n_tests_written_after_edit,
      len(list_filter(steps, s -> s.kind = 'cleanup')) AS n_cleanup,
      list_transform(steps[1:30], s -> s.kind) AS kinds30,
      list_transform(steps[1:30], s -> md5(s.fn || '|' || s.args)) AS hash30
    FROM e
    """


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="analysis/exploratory/data/nvidia_runs.parquet")
    p.add_argument("--subsets", nargs="*", default=SUBSETS)
    p.add_argument("--files", default="*.parquet", help="file glob inside each subset (for quick tests)")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    parts = []
    for s in args.subsets:
        tmp = f"{args.out}.{s.replace('/', '__')}.parquet"
        con.execute(f"COPY ({sql_for(f'{BASE}/{s}/{args.files}', s)}) TO '{tmp}' (FORMAT parquet)")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{tmp}')").fetchone()[0]
        print(f"{s}: {n} runs", flush=True)
        parts.append(tmp)
    union = " UNION ALL ".join(f"SELECT * FROM read_parquet('{t}')" for t in parts)
    con.execute(f"COPY ({union}) TO '{args.out}' (FORMAT parquet)")
    for t in parts:
        os.remove(t)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
