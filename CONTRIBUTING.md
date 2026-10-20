# Contributing

## Rules

1. **Follow the locked procedure.** Work goes step by step (problem → data → clean → explore → features → model → evaluate → communicate). Each step ends with a written report in `docs/`. See [`docs/project/PROJECT_PLAN.md`](docs/project/PROJECT_PLAN.md).
2. **Code lives in the package.** Put new code in `src/agent_tips/<step>/` and expose it through the `agent-tips` CLI. Loose scripts are not allowed; `analysis/exploratory/` is a frozen record.
3. **Use no hard-coded paths or seeds.** Read them from `configs/*.yaml` and build paths with `pathlib`.
4. **Use cross-platform tooling only.** Every dependency must have wheels for Linux x86_64/ARM64, Windows x86_64 and macOS ARM64. Don't add shell-only scripts.
5. **The GPU stays optional.** Use `agent_tips.utils.device.select_device()`; code must also run on CPU.
6. **Add docs and tests with the code.** Give every public function type hints and a NumPy-style docstring, and add tests for new logic. Estimators get a test on synthetic data with a known answer.

## Checks

```bash
uv sync
uv run pre-commit install
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

CI runs the same checks on all four platforms for every push and pull request.

## Commits

Make small commits whose messages say what changed and why. Do each phase on its own branch and merge it into `main` after review.
