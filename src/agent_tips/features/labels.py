"""Step 5: one shared labeler for every AI action, in all three datasets.

Labels are produced by SQL (DuckDB), so the same rules run on 26 million steps
and in the unit tests. Each step gets a raw label from its tool and command.
Edits are then split into ``edit_created`` (a file this try created, e.g. its own
test script) and ``edit_code`` (existing project code).

Labels
------
repro_create   created a test or reproduction script (name looks like one)
create_other   created any other new file
edit_code      edited existing project code
edit_created   edited a file the try itself created
run            ran a script or program
test_suite     ran the project's test suite
look           looked at or searched files (view, open, grep, ls, cat, ...)
think          a "think" tool call (no action)
plan           a planning / task-tracker call
cleanup        deleted files (``rm``)
submit         submitted / finished
other          anything else (installs, git, cd, ...)
"""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "repro_create",
    "create_other",
    "edit_code",
    "edit_created",
    "run",
    "test_suite",
    "look",
    "think",
    "plan",
    "cleanup",
    "submit",
    "other",
)

# A new file whose *file name* suggests a test or reproduction script. Only the
# base name is checked: paths like /testbed/src/x.py must not count as "test".
REPRO_NAME = (
    r"(repro|test|spec|debug|issue|bug|verif|valid|check|demo|example|scratch|try_|poc|sample|"
    r"investigat|trace|compare|simulat|analy|experiment|explor|minimal|temp|tmp|quick|understand)"
)
# Documents are never scripts (e.g. /tmp/SOLUTION_SUMMARY.md).
DOC_EXT = r"\.(md|txt|rst)$"
# Files created in /tmp are scratch scripts.
SCRATCH_DIR = r"^/tmp/"
HEREDOC = r"<<-?\s*['\"]?[A-Za-z_]+"
APPEND = r">>\s*['\"]?[\w./~-]+"
REDIRECT_FILE = r">\s*['\"]?([\w./~-]+\.[A-Za-z0-9]+)"
ECHO_CREATE = r"^\s*(echo|printf)\b[^|;&]*>\s*['\"]?[\w./~-]+\.[A-Za-z0-9]+"
BASH_EDIT = r"(sed\s+-i|perl\s+-[a-z]*i|git\s+apply|patch\s+-p)"
# Test-suite commands; each must start a command word (not appear inside a path or name).
TEST_SUITE = (
    r"(^|[\s;&|(/])(pytest|python3?\s+-m\s+(pytest|unittest)|go\s+test|mvn\s+(\S+\s+)*test|"
    r"gradlew?\s+(\S+\s+)*test|npm\s+(run\s+)?test|yarn\s+(run\s+)?test|pnpm\s+(run\s+)?test|"
    r"cargo\s+test|phpunit|rspec|tox|dotnet\s+test|make\s+(\S+\s+)*test|jest|vitest|mocha|ctest)"
    r"(\s|$|:|;|&|\|)"
)
RUN = (
    r"(^|[;&|(]\s*|\s)(python3?|go\s+(run|build|vet)|mvn|gradle|\./gradlew|npm\s+run|npx|yarn|"
    r"pnpm|cargo\s+(run|build|check)|node|ruby|bundle\s+exec|php|dotnet|make|java|javac|"
    r"gcc|g\+\+|clang|julia|Rscript|bash\s+\S+\.sh|sh\s+\S+\.sh)\b"
)
CLEANUP = r"(^|[;&|]\s*|\s)rm\s+(-\S+\s+)*\S"
# Version or help checks only look at the tool; they run nothing.
VERSION_HELP = r"\s--(version|help)\b"
# A run that re-executes the try's own reproduction/test script ("check your fix").
RERUN = r"^\s*(cd\s+\S+\s*&&\s*)?(python3?|bash|sh|node|go\s+run)\s+\S*(repro|test|bug|issue|verify|check)"

OLDER_LOOK = (
    "open",
    "goto",
    "scroll_up",
    "scroll_down",
    "search_dir",
    "search_file",
    "find_file",
    "ls",
    "cat",
    "grep",
    "find",
    "head",
    "tail",
    "less",
    "wc",
    "tree",
)
OLDER_SUBMIT = ("submit", "exit")


def _q(s: str) -> str:
    """Quote a Python string as a SQL literal."""
    return "'" + s.replace("'", "''") + "'"


def _in(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(_q(v) for v in values) + ")"


def _is_repro(path_sql: str) -> str:
    """Return the SQL condition "the file name looks like a test/repro script, or it is in /tmp".

    Documents (``.md``, ``.txt``, ``.rst``) never count.
    """
    name = f"lower(regexp_extract({path_sql}, '[^/]+$'))"
    return (
        f"((regexp_matches({name}, {_q(REPRO_NAME)}) OR regexp_matches({path_sql}, {_q(SCRATCH_DIR)}))"
        f" AND NOT regexp_matches({name}, {_q(DOC_EXT)}))"
    )


def raw_label_sql() -> str:
    """Return the SQL CASE expression giving each step its raw label.

    Uses the columns ``tool``, ``action``, ``command`` and ``path`` of the
    steps table. ``edit`` is split later into ``edit_code`` / ``edit_created``.
    """
    lower_cmd = "lower(command)"
    return f"""CASE
      WHEN tool IN ('finish', 'submit') OR (tool IN {_in(OLDER_SUBMIT)}) THEN 'submit'
      WHEN tool = 'think' THEN 'think'
      WHEN tool = 'task_tracker' THEN 'plan'
      WHEN tool = 'fetch' OR regexp_matches(tool, '^(cat|ls|find|open|grep|head|tail)[./]') THEN 'look'
      WHEN tool IN ('str_replace_editor', 'create') AND (action = 'create' OR tool = 'create') THEN
        CASE WHEN {_is_repro("path")} THEN 'repro_create' ELSE 'create_other' END
      WHEN tool = 'str_replace_editor' AND action IN ('str_replace', 'insert', 'undo_edit') THEN 'edit'
      WHEN tool = 'str_replace_editor' THEN 'look'
      WHEN tool = 'edit' THEN 'edit'
      WHEN regexp_matches(command, {_q(APPEND)}) AND tool NOT IN {_in(OLDER_LOOK)} THEN 'edit'
      WHEN (regexp_matches(command, {_q(HEREDOC)}) AND regexp_matches(command, {_q(REDIRECT_FILE)}))
           OR regexp_matches(command, {_q(ECHO_CREATE)}) THEN
        CASE WHEN {_is_repro(f"regexp_extract(command, {_q(REDIRECT_FILE)}, 1)")}
             THEN 'repro_create' ELSE 'create_other' END
      WHEN regexp_matches(command, {_q(BASH_EDIT)}) THEN 'edit'
      WHEN regexp_matches(command, {_q(VERSION_HELP)}) THEN 'look'
      WHEN regexp_matches({lower_cmd}, {_q(TEST_SUITE)}) THEN 'test_suite'
      WHEN regexp_matches(command, {_q(RUN)}) THEN 'run'
      WHEN regexp_matches(command, {_q(CLEANUP)}) THEN 'cleanup'
      WHEN tool IN {_in(OLDER_LOOK)} THEN 'look'
      WHEN tool IN ('bash', 'execute_bash') THEN 'look'
      ELSE 'other' END"""


def target_sql() -> str:
    """Return the SQL expression for the file a step acts on (may be empty)."""
    return f"""CASE
      WHEN tool = 'edit' THEN open_file
      WHEN tool IN ('str_replace_editor', 'create', 'open') THEN path
      WHEN regexp_matches(command, {_q(REDIRECT_FILE)})
           AND (regexp_matches(command, {_q(HEREDOC)}) OR regexp_matches(command, {_q(ECHO_CREATE)}))
           THEN regexp_extract(command, {_q(REDIRECT_FILE)}, 1)
      WHEN regexp_matches(command, {_q(BASH_EDIT)}) THEN regexp_extract(command, '(\\S+)\\s*$', 1)
      ELSE '' END"""


def labels_query(steps_sql: str) -> str:
    """Return a query that labels every step of the given steps relation.

    ``steps_sql`` is a SQL relation (table name, view or ``read_parquet(...)``)
    with the columns of :data:`agent_tips.cleaning.schema.STEPS`.
    """
    base = "regexp_extract(target, '[^/]+$')"
    return f"""
    WITH l AS (
      SELECT try_id, step, tool, action, command, path, open_file, obs_head,
             {raw_label_sql()} AS raw_label, coalesce({target_sql()}, '') AS target
      FROM {steps_sql}),
    c AS (
      SELECT try_id, list(DISTINCT {base}) AS created
      FROM l WHERE raw_label IN ('repro_create', 'create_other') AND target <> ''
      GROUP BY try_id)
    SELECT l.try_id, l.step, l.tool, l.action, l.command, l.target, l.obs_head,
      CASE WHEN l.raw_label = 'edit' AND list_contains(coalesce(c.created, []), {base})
           THEN 'edit_created'
           WHEN l.raw_label = 'edit' THEN 'edit_code'
           ELSE l.raw_label END AS label
    FROM l LEFT JOIN c USING (try_id)
    """
