# Changelog

## [Unreleased]

### Added

- Phase B:
  - `agent-tips clean` (shared `tries`/`steps` tables, flags, version batches, funnel, generated `docs/03_cleaning_report.md`);
  - `agent-tips eda` (7 charts, `docs/04_eda_report.md`);
  - CI actions upgraded (checkout v7, setup-uv v10).

- Phase A:
  - frozen hypotheses and analysis plan (`docs/01_problem_and_hypotheses.md`, tag `analysis-plan-v1`);
  - `agent-tips download` with pinned revisions and SHA-256 verification;
  - DuckDB raw-data views (`agent_tips.data.load`);
  - data inventory (`docs/02_data_inventory.md`);
  - data cards moved to `docs/data/`.

- Phase 0 project setup:
  - `agent_tips` package with the `agent-tips` CLI;
  - config loading, seed and device helpers;
  - uv lockfile for four platforms;
  - ruff, mypy, pytest and pre-commit;
  - MkDocs site;
  - GitHub Actions CI matrix (Linux x86_64 and ARM64, Windows x86_64, macOS ARM64).
- MIT license and `CITATION.cff`.
- Research documents in `docs/project/`. Exploratory scripts and their results in `analysis/exploratory/`.
