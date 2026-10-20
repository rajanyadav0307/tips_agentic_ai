"""The shared labeler on hand-written cases covering all three scaffolds and 9 languages."""

import duckdb
import pandas as pd
import pytest

from agent_tips.features.labels import LABELS, labels_query, raw_label_sql

E = "str_replace_editor"
B = "execute_bash"
S = "bash"  # SWE-agent's shell tool name

# (tool, action, command, path, open_file, expected raw label)
CASES: list[tuple[str, str, str, str, str, str]] = [
    # ---- older AI (SWE-agent 2024): commands are the tool names --------------
    ("create", "create", "create reproduce.py", "reproduce.py", "", "repro_create"),
    ("create", "create", "create reproduce_bug.py", "reproduce_bug.py", "", "repro_create"),
    ("create", "create", "create test_issue.py", "test_issue.py", "", "repro_create"),
    ("create", "create", "create reproduce.sh", "reproduce.sh", "", "repro_create"),
    ("create", "create", "create reproduce.sql", "reproduce.sql", "", "repro_create"),
    ("create", "create", "create src/utils/helpers.py", "src/utils/helpers.py", "", "create_other"),
    ("create", "create", "create setup.cfg", "setup.cfg", "", "create_other"),
    ("edit", "edit", "edit 1:1\nimport x\nend_of_edit", "", "/r/reproduce.py", "edit"),
    ("edit", "edit", "edit 10:12\n    return a\nend_of_edit", "", "/r/src/core.py", "edit"),
    ("python", "python", "python reproduce.py", "", "", "run"),
    ("python3", "python3", "python3 reproduce_bug.py", "", "", "run"),
    ("python", "python", "python -c 'import x; print(x)'", "", "", "run"),
    ("pytest", "pytest", "pytest tests/test_core.py", "", "", "test_suite"),
    ("python", "python", "python -m pytest tests/ -x", "", "", "test_suite"),
    ("python", "python", "python -m unittest discover", "", "", "test_suite"),
    ("tox", "tox", "tox -e py39", "", "", "test_suite"),
    ("search_dir", "search_dir", 'search_dir "def parse"', "", "", "look"),
    ("search_file", "search_file", 'search_file "foo"', "", "", "look"),
    ("find_file", "find_file", 'find_file "core.py"', "", "", "look"),
    ("open", "open", "open src/core.py", "src/core.py", "", "look"),
    ("open", "open", "open src/core.py 120", "src/core.py", "", "look"),
    ("goto", "goto", "goto 583", "", "", "look"),
    ("scroll_down", "scroll_down", "scroll_down", "", "", "look"),
    ("scroll_up", "scroll_up", "scroll_up", "", "", "look"),
    ("ls", "ls", "ls -F", "", "", "look"),
    ("cat", "cat", "cat setup.py", "", "", "look"),
    ("grep", "grep", "grep -rn 'foo' src", "", "", "look"),
    ("find", "find", "find . -name '*.py'", "", "", "look"),
    ("submit", "submit", "submit", "", "", "submit"),
    ("rm", "rm", "rm reproduce.py", "", "", "cleanup"),
    ("rm", "rm", "rm -f reproduce.py", "", "", "cleanup"),
    ("pip", "pip", "pip install -e .", "", "", "other"),
    ("cd", "cd", "cd src", "", "", "other"),
    ("git", "git", "git status", "", "", "other"),
    ("", "", "", "", "", "other"),
    # ---- OpenHands / SWE-agent editor tool -----------------------------------
    (E, "create", "create", "/workspace/repo/reproduce_issue.py", "", "repro_create"),
    (E, "create", "create", "/workspace/repo/test_fix.py", "", "repro_create"),
    (E, "create", "create", "/workspace/repo/comprehensive_test.py", "", "repro_create"),
    (E, "create", "create", "/workspace/repo/final_verification.py", "", "repro_create"),
    (E, "create", "create", "/workspace/repo/debug_issue.py", "", "repro_create"),
    (E, "create", "create", "/testbed/edge_case_test.go", "", "repro_create"),
    (E, "create", "create", "/testbed/src/test/java/FooTest.java", "", "repro_create"),
    (E, "create", "create", "/testbed/tests/repro.rs", "", "repro_create"),
    (E, "create", "create", "/tmp/check.ts", "", "repro_create"),
    (E, "create", "create", "/testbed/example.php", "", "repro_create"),
    (E, "create", "create", "/testbed/src/new_module.py", "", "create_other"),
    (E, "create", "create", "/testbed/pkg/util/format.go", "", "create_other"),
    (E, "create", "create", "/testbed/src/lib.rs", "", "create_other"),
    (E, "create", "create", "/testbed/src/components/Button.tsx", "", "create_other"),
    (E, "create", "create", "/testbed/CHANGELOG.md", "", "create_other"),
    (E, "str_replace", "str_replace", "/testbed/src/core.py", "", "edit"),
    (E, "str_replace", "str_replace", "/testbed/pkg/server/handler.go", "", "edit"),
    (E, "str_replace", "str_replace", "/testbed/src/index.ts", "", "edit"),
    (E, "insert", "insert", "/testbed/src/core.py", "", "edit"),
    (E, "undo_edit", "undo_edit", "/testbed/src/core.py", "", "edit"),
    (E, "view", "view", "/testbed", "", "look"),
    (E, "view", "view", "/testbed/src/core.py", "", "look"),
    (E, "view", "view", "/testbed/src/Foo.java", "", "look"),
    # ---- shell: creating files ------------------------------------------------
    (
        B,
        "cat",
        "cat > /workspace/repo/reproduce.py << 'EOF'\nimport x\nEOF",
        "",
        "",
        "repro_create",
    ),
    (B, "cat", "cat << 'EOF' > /tmp/test_bug.go\npackage main\nEOF", "", "", "repro_create"),
    (B, "cat", "cat > /testbed/repro_test.rs <<EOF\nfn main() {}\nEOF", "", "", "repro_create"),
    (B, "cat", "cat <<- END > check.js\nconsole.log(1)\nEND", "", "", "repro_create"),
    (B, "cat", "cat > src/new_helper.py << 'EOF'\ndef f(): pass\nEOF", "", "", "create_other"),
    (S, "cat", "cat > /testbed/src/util.ts << 'EOF'\nexport {}\nEOF", "", "", "create_other"),
    (B, "echo", "echo 'print(1)' > /tmp/test_x.py", "", "", "repro_create"),
    (B, "printf", "printf 'x' > debug.py", "", "", "repro_create"),
    (B, "echo", "echo 'a' > config.yaml", "", "", "create_other"),
    # ---- shell: edits -----------------------------------------------------------
    (B, "sed", "sed -i 's/foo/bar/' src/core.py", "", "", "edit"),
    (B, "cd", "cd /testbed && sed -i '12d' pkg/a.go", "", "", "edit"),
    (S, "git", "git apply fix.patch", "", "", "edit"),
    (B, "patch", "patch -p1 < fix.diff", "", "", "edit"),
    (B, "perl", "perl -pi -e 's/a/b/' lib/x.pm", "", "", "edit"),
    # ---- shell: test suites (9 languages) --------------------------------------
    (B, "cd", "cd /workspace/repo && python -m pytest tests/test_core.py -v", "", "", "test_suite"),
    (B, "pytest", "pytest -xvs tests/", "", "", "test_suite"),
    (B, "python", "python -m unittest tests.test_x", "", "", "test_suite"),
    (B, "go", "go test ./pkg/...", "", "", "test_suite"),
    (B, "cd", "cd /testbed && go test -run TestFoo ./...", "", "", "test_suite"),
    (B, "mvn", "mvn -q test -Dtest=FooTest", "", "", "test_suite"),
    (B, "mvn", "mvn test", "", "", "test_suite"),
    (B, "./gradlew", "./gradlew test --tests Foo", "", "", "test_suite"),
    (B, "npm", "npm test", "", "", "test_suite"),
    (B, "npm", "npm run test -- --watch=false", "", "", "test_suite"),
    (B, "npx", "npx jest src/foo.test.ts", "", "", "test_suite"),
    (B, "npx", "npx vitest run", "", "", "test_suite"),
    (B, "yarn", "yarn test", "", "", "test_suite"),
    (B, "pnpm", "pnpm test", "", "", "test_suite"),
    (B, "cargo", "cargo test --lib", "", "", "test_suite"),
    (B, "cd", "cd /testbed && cargo test foo", "", "", "test_suite"),
    (B, "vendor/bin/phpunit", "vendor/bin/phpunit tests/FooTest.php", "", "", "test_suite"),
    (B, "bundle", "bundle exec rspec spec/foo_spec.rb", "", "", "test_suite"),
    (B, "dotnet", "dotnet test", "", "", "test_suite"),
    (B, "make", "make test", "", "", "test_suite"),
    (B, "ctest", "ctest --output-on-failure", "", "", "test_suite"),
    (B, "tox", "tox -e py311", "", "", "test_suite"),
    (B, "cd", "cd /workspace/r && python -m pytest --version", "", "", "look"),
    (B, "go", "go version", "", "", "look"),
    (B, "npm", "npm --version", "", "", "look"),
    # ---- shell: running scripts / programs --------------------------------------
    (B, "cd", "cd /workspace/repo && python reproduce_issue.py", "", "", "run"),
    (B, "python", "python /tmp/check.py", "", "", "run"),
    (B, "python3", "python3 -c 'import sys; print(sys.version)'", "", "", "run"),
    (B, "go", "go run /tmp/repro.go", "", "", "run"),
    (B, "go", "go build ./...", "", "", "run"),
    (B, "go", "go vet ./...", "", "", "run"),
    (B, "node", "node check.js", "", "", "run"),
    (B, "npx", "npx tsc --noEmit", "", "", "run"),
    (B, "cargo", "cargo build", "", "", "run"),
    (B, "cargo", "cargo check", "", "", "run"),
    (B, "javac", "javac Foo.java && java Foo", "", "", "run"),
    (B, "php", "php repro.php", "", "", "run"),
    (B, "ruby", "ruby repro.rb", "", "", "run"),
    (B, "gcc", "gcc -o t t.c && ./t", "", "", "run"),
    (B, "bash", "bash run_check.sh", "", "", "run"),
    (B, "make", "make", "", "", "run"),
    # ---- shell: looking around ---------------------------------------------------
    (B, "find", "find /testbed -name '*.py' | head", "", "", "look"),
    (B, "grep", "grep -rn 'parse' /testbed/src", "", "", "look"),
    (B, "cd", "cd /testbed && grep -rn foo .", "", "", "look"),
    (B, "ls", "ls -la /workspace/repo", "", "", "look"),
    (B, "cat", "cat /testbed/setup.py", "", "", "look"),
    (B, "head", "head -50 src/core.py", "", "", "look"),
    (B, "git", "git diff", "", "", "look"),
    (B, "git", "git status", "", "", "look"),
    (B, "cd", "cd /testbed", "", "", "look"),
    (B, "pip", "pip install -e .", "", "", "look"),
    (B, "C-c", "C-c", "", "", "look"),
    (S, "find", "find . -type f -name '*.go'", "", "", "look"),
    # ---- shell: cleanup -------------------------------------------------------------
    (B, "rm", "rm /workspace/repo/reproduce_issue.py", "", "", "cleanup"),
    (B, "rm", "rm -f test_*.py debug_*.py", "", "", "cleanup"),
    (B, "cd", "cd /testbed && rm -rf /tmp/check", "", "", "cleanup"),
    # ---- other tools -------------------------------------------------------------------
    ("think", "", "", "", "", "think"),
    ("task_tracker", "plan", "plan", "", "", "plan"),
    ("task_tracker", "view", "view", "", "", "plan"),
    ("finish", "", "", "", "", "submit"),
    ("submit", "", "", "", "", "submit"),
    ("none", "", "", "", "", "other"),
    # ---- regressions found by the blind second labeling ------------------------------
    (B, "cd", "cd /workspace/tox-dev__tox__3.8 && rm reproduce_issue.py", "", "", "cleanup"),
    (B, "cd", "cd /workspace/wix__Detox__1.0/detox && node debug_test.js", "", "", "run"),
    ("wget", "wget", "wget https://x.org/jw01409_nirspec_prism.fits", "", "", "other"),
    (B, "cd", "cd /x/site-packages/tox/config && grep -n INFO __init__.py", "", "", "look"),
    (E, "create", "create", "/workspace/r/investigate_further.py", "", "repro_create"),
    (E, "create", "create", "/workspace/r/manual_trace.py", "", "repro_create"),
    (E, "create", "create", "/workspace/r/compare_old_new.py", "", "repro_create"),
    (E, "create", "create", "/tmp/SOLUTION_SUMMARY.md", "", "create_other"),
    (E, "create", "create", "/testbed/IMPLEMENTATION_SUMMARY.md", "", "create_other"),
    (B, "cd", "cd /testbed && cat >> klog.go << 'EOF'\nfunc X() {}\nEOF", "", "", "edit"),
    ("fetch", "", '{"url": "https://example.com"}', "", "", "look"),
    ("ls.dvc", "ls.dvc", "ls.dvc", "", "", "look"),
    ("cat.bashrc", "cat.bashrc", "cat.bashrc", "", "", "look"),
    (B, "npx", "npx jest", "", "", "test_suite"),
    (B, "cd", "cd /r && tox", "", "", "test_suite"),
]


def _run(rows: list[tuple[str, ...]], sql: str) -> pd.DataFrame:
    con = duckdb.connect()
    con.register(
        "df", pd.DataFrame(rows, columns=["tool", "action", "command", "path", "open_file"])
    )
    return con.sql(f"SELECT {sql} AS label FROM df").df()


def test_enough_cases() -> None:
    assert len(CASES) >= 130


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}:{c[2][:30]}" for c in CASES])
def test_raw_label(case: tuple[str, str, str, str, str, str]) -> None:
    got = _run([case[:5]], raw_label_sql()).label[0]
    assert got == case[5], f"{case[:5]} -> {got}, expected {case[5]}"


def test_edits_split_into_created_and_code() -> None:
    con = duckdb.connect()
    steps = pd.DataFrame(
        [
            ("t1", 0, "create", "create", "create reproduce.py", "reproduce.py", "", ""),
            ("t1", 1, "edit", "edit", "edit 1:1\nx\nend_of_edit", "", "/r/reproduce.py", ""),
            ("t1", 2, "edit", "edit", "edit 5:5\ny\nend_of_edit", "", "/r/src/core.py", ""),
            ("t2", 0, E, "create", "create", "/w/check_fix.py", "", ""),
            ("t2", 1, E, "str_replace", "str_replace", "/w/check_fix.py", "", ""),
            ("t2", 2, B, "sed", "sed -i 's/a/b/' /w/lib/x.py", "", "", ""),
            ("t3", 0, E, "str_replace", "str_replace", "/w/reproduce.py", "", ""),
        ],
        columns=["try_id", "step", "tool", "action", "command", "path", "open_file", "obs_head"],
    )
    con.register("steps", steps)
    out = con.sql(labels_query("steps")).df().sort_values(["try_id", "step"])
    assert out.label.tolist() == [
        "repro_create",
        "edit_created",
        "edit_code",
        "repro_create",
        "edit_created",
        "edit_code",
        "edit_code",  # t3 never created reproduce.py itself
    ]
    assert set(out.label) <= set(LABELS)
