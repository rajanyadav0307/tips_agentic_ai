# Changelog

## [1.0.0] - 2026-10-20

### Added

- Phase E:
  - `agent-tips figures`: paper figures F1–F8 (PDF and PNG), tables T1–T4 (LaTeX and Markdown, `docs/08_tables.md`) and `numbers.tex`, so every number in the paper comes from the results files;
  - `agent-tips run-all`: Steps 2–8 in order;
  - paper draft in `paper/` (ACM format), compiled to `paper/main.pdf`;
  - related-work refresh (2026-10-20) in `docs/project/DS_PROJECT_PROPOSAL.md`;
  - end-to-end test of Steps 3–8 on a larger fake project; test coverage 96%.
- Phase D:
  - `agent-tips model`: fair (within-group) estimator with issue-cluster bootstrap and permutation test, fixed-effects regression with clustered errors, interactions, survival analysis, exact-history fork pairs, simple-vs-fair gaps, design checks, Benjamini–Hochberg; generated `docs/06_results.md`;
  - `agent-tips evaluate`: robustness checks (batch definitions, undated tries, leave one AI out, label errors, NVIDIA's removed tries, conditional logistic regression) and a two-run reproducibility check; generated `docs/07_evaluation.md`.

- Phase C:
  - `agent-tips features`: a shared action labeler (SQL), try features (arms, counts, localization) and issue features;
  - label validation: observation check, plus a blind second labeling in two rounds (`docs/data/label_validation/`);
  - generated `docs/05_feature_report.md` with the tip-compliance chart.

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

### Fixed

- Bootstrap p-values can no longer be exactly 0 (they use (count + 1) / (reps + 1)).
- P3 counts "opposite sign" only when both estimates are non-zero; it is also reported without underpowered pairs.
- Reports create the `docs/` folder if it is missing.
- `agent-tips download` (and so `run-all`) works offline when every pinned file is already downloaded and verified.
- Paper figure PDFs carry no creation date, so re-running gives identical files.
- Analysis rows now load in a fixed order, so random draws (and results) are identical across runs.
- File names inside `/testbed` no longer count as test scripts. The exploratory code had this bug.
- Test-tool names inside paths no longer count as test-suite runs.
