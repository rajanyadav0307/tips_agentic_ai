# Project Plan: From Current State to Paper-Ready Results

**Version:** 2 (detailed), 2026-10-20
**Project:** *Do Tips for AI Coding Agents Actually Help?* (results so far in [DS_PROJECT_PROPOSAL.md](DS_PROJECT_PROPOSAL.md))
**Rules:**
- All work follows the locked 8-step data science procedure, in order.
- All code follows the locked engineering standards.
- **The assistant does all the work. The project owner reviews and checks it** at the end of each phase, and the next phase starts only after the owner approves.

## In plain words

- **Where we are:** we have strong first results (the tip doesn't help or even hurts, and simple analysis is badly misleading). The project is set up professionally on GitHub, and Phase 0 is done.
- **What happens next:** we rebuild every result inside the clean, tested package, one step at a time. The final numbers in the paper come only from this pipeline, never from the quick early scripts.
- **No work for you:** the label check that needed a person is replaced by two automatic checks:
  1. each action is checked against what the computer actually replied (for example, "file created");
  2. an independent second labeling by the assistant.

  The paper will say honestly that the label check was done this way, not by a human.
- **The finish line:**
  - every research question answered;
  - 8 figures and 4 tables;
  - a full paper draft;
  - a tagged release on GitHub that reproduces everything with one command.
- **Who does what:** I do all the work. You review it. At the end of each phase, I give you a plain-words summary and a short **"what to check"** list (which files to open, which numbers to look at). I start the next phase only after you approve.

## Contents

1. Locked procedure
2. Current state
3. Decisions (defaults chosen by the assistant)
4. Task breakdown by phase (A–E), with inputs, outputs, "done when" rules, dependencies and effort
5. Dependency overview
6. Risk register
7. Engineering standards (locked)
8. Folder layout
9. Definition of done for the whole project

## 1. Locked procedure

| Step | Name | Output of the step |
| --- | --- | --- |
| 1 | Define the problem | Hypotheses and main analyses written down and frozen **before** the final runs |
| 2 | Collect data | Pinned, checksummed downloads and a data inventory |
| 3 | Clean and prepare | Clean tables in one shared format, plus a cleaning report with row counts at every filter |
| 4 | Explore (EDA) | Exploration charts and an EDA report |
| 5 | Create features | One shared labeler; validated labels; issue and localization features |
| 6 | Model and analyze | All research questions answered with fair estimates, regression models and false-discovery correction |
| 7 | Evaluate | Robustness, sensitivity and reproducibility checks |
| 8 | Communicate | Figures, tables, paper draft and release |

## 2. Current state (2026-10-20)

| Step | Status | Notes |
| --- | --- | --- |
| Setup (Phase 0) | ✅ Done | Package, uv lockfile for 4 platforms, ruff/mypy/pytest, MkDocs, CI passing on all 4 platforms, MIT license, CITATION.cff |
| 1 Problem | ✅ Done (Phase A) | Hypotheses and analysis plan frozen in `docs/01_problem_and_hypotheses.md` (tag `analysis-plan-v1`) |
| 2 Data | ✅ Done (Phase A) | 121 pinned, checksummed files (24.35 GB); `agent-tips download`; loaders; inventory in `docs/02_data_inventory.md` |
| 3 Clean | ✅ Done (Phase B) | `agent-tips clean`: 402,043 tries and 26.5M steps in one shared format; funnels match the exploration exactly; `docs/03_cleaning_report.md` |
| 4 EDA | ✅ Done (Phase B) | `agent-tips eda`: 7 charts plus `docs/04_eda_report.md` |
| 5 Features | ⚠️ Exploratory only | Separate regex labelers per script; labels not validated |
| 6 Model | ⚠️ Exploratory only | Same-bug comparisons, bootstrap and permutation exist; no regression models or FDR |
| 7 Evaluate | ⚠️ Partial | Robustness checks exist for the older AI only |
| 8 Communicate | ❌ Missing | — |

## 3. Decisions (defaults chosen by the assistant)

These were open questions. Defaults are now set so work never waits. Any of them can be overridden at any time.

| Decision | Default | Reason |
| --- | --- | --- |
| Label check | Automatic: (a) checking each action against the computer's reply, plus (b) blind second labeling by the assistant on 600 sampled actions. Reported in the paper as AI-assisted, not human, validation. | The assistant does the work; the owner reviews it (and can inspect the sample spreadsheet) |
| Paper format | ACM `acmart` two-column, about 10 pages plus references | Fits MSR, ESEM and FSE-style venues and most workshops; easy to convert |
| Target venue | Kept open; the draft is written to be venue-neutral | Deadlines are not verified; choose when the draft is ready |
| Paid experiment with closed AIs | Not included | Needs money and accounts, and isn't needed for the story |
| Archival DOI (Zenodo) | Not included | Needs the owner's account; can be added later |
| Python version | 3.12 | Already locked in Phase 0 |

## 4. Task breakdown

Each task lists: **What**, **Inputs**, **Outputs** (code module and files), **Done when**, **Depends on**, and **Effort** (assistant working time, plus machine time where it matters).

Every phase ends with a checkpoint:
1. the work is pushed to GitHub from the owner's account;
2. the assistant gives a plain-words summary and a "what to check" list;
3. **the owner reviews and approves** before the next phase starts.

The label-validation sample (C2) is saved as an easy-to-read spreadsheet, so the owner can check any labels they like.

---

### Phase A: Define the problem and collect data (Steps 1–2)

**A1. Write and freeze hypotheses and the analysis plan**
- **What:** turn RQ1–RQ6 into testable hypotheses. For each one, record:
  - the outcome (resolved yes/no);
  - the unit (one try);
  - the comparison groups;
  - the grouping used for fairness (issue × evaluation day for the older AI; issue for the newer AIs);
  - the estimator, the 95% confidence interval method and the decision rule.
- **Primary analyses** (no multiple-testing correction):
  - **P1:** older AI, reproduce-first vs. search-first at step 0.
  - **P2:** newer AIs (NVIDIA), test-first vs. fix-first, pooled across AIs.
  - **P3:** the gap between simple and fair estimates, summarized across all tip × AI pairs.
- **Secondary analyses:** everything else, with Benjamini–Hochberg correction at q = 0.05.
- **Outputs:** `docs/01_problem_and_hypotheses.md`; git tag `analysis-plan-v1`.
- **Done when:** the file is committed and tagged *before* any Phase D run.
- **Depends on:** —
- **Effort:** 1–2 h.

**A2. Pin data versions and write the data inventory**
- **What:** record the Hugging Face revision (commit hash) of each dataset, plus its size, row count, license and intended use.
- **Outputs:** `configs/default.yaml` (with revisions); `docs/02_data_inventory.md`.
- **Done when:** every source has a revision, license and expected row count.
- **Depends on:** —
- **Effort:** 1 h.

**A3. `agent-tips download` command**
- **What:** download the pinned files with `huggingface_hub` (pure Python, so it works on all platforms):
  - the older AI shards;
  - the Nebius OpenHands file;
  - the NVIDIA subsets in the config;
  - the nebius/SWE-bench-extra metadata (gold patches).

  It writes `data/raw/MANIFEST.json` with SHA-256 checksums, and rerunning skips verified files.
- **Outputs:** `src/agent_tips/data/download.py`; `huggingface_hub` added to the dependencies; tests using a mocked hub.
- **Done when:**
  - a fresh download matches the expected row counts;
  - a second run downloads nothing;
  - a corrupted file is detected.
- **Depends on:** A2.
- **Effort:** 3 h, plus about 30–60 min of download time (about 25 GB).

**A4. Data loaders**
- **What:** DuckDB-backed loaders that return typed tables for each raw dataset, reading only the columns that are needed.
- **Outputs:** `src/agent_tips/data/load.py`; unit tests.
- **Done when:** tests pass on fixtures and the real data loads on macOS.
- **Depends on:** A3.
- **Effort:** 2 h.

**A5. Small test fixtures**
- **What:** tiny synthetic files in each raw format (a few issues, a few tries, known answers), so CI can run the whole pipeline without real data.
- **Outputs:** `tests/fixtures/`; a fixture-based end-to-end test.
- **Done when:** CI runs the pipeline on the fixtures on all 4 platforms.
- **Depends on:** A4.
- **Effort:** 2 h.

**Checkpoint A:** hypotheses frozen and tagged; data downloaded and verified; plain-words summary to the owner. ✅ **Completed 2026-10-20**; awaiting the owner's review.

---

### Phase B: Clean, prepare and explore (Steps 3–4)

**B1. Shared table formats**
- **What:** define two tables used by every dataset:
  - `tries`: try ID, dataset, AI, scaffold, issue, batch, resolved, exit status, number of steps, has patch, evaluation time, issue-text hash, language, text read and written.
  - `steps`: try ID, step number, tool, command, file path, argument hash, observation length, thought length, observation text (truncated).
- **Outputs:** `src/agent_tips/cleaning/schema.py`; written documentation of every column.
- **Done when:** the schema is checked by tests.
- **Depends on:** A4.
- **Effort:** 2 h.

**B2. One cleaner per dataset, with funnel logging**
- **What:** each filter records how many rows it kept and removed, and why:
  - **Older AI:** 70B model only → modal issue text → dated vs. undated tries.
  - **Nebius newer AI:** modal issue text.
  - **NVIDIA:** known outcome (not −1); language names normalized (`js` → `javascript`, `ts` → `typescript`).
- **Outputs:** `src/agent_tips/cleaning/{older_ai,nebius,nvidia}.py`; `data/processed/*.parquet`; `outputs/results/funnel.json`; `agent-tips clean`.
- **Done when:** the funnel counts match the exploratory numbers. Examples:
  - 74,792 → 73,738 same-input tries, of which 64,274 are dated (older AI);
  - 254,933 NVIDIA tries, 83.2% with a known outcome.

  Any difference is explained in the report.
- **Depends on:** B1.
- **Effort:** 5 h, plus about 15 min of machine time.

**B3. Batch (AI version) assignment for the older AI**
- **What:** assign day, 6-hour, hour, 60-minute-gap and 10-minute-gap batches from evaluation timestamps.
- **Outputs:** batch columns in `tries`; tests with synthetic timestamps.
- **Done when:** batch counts match the exploratory results.
- **Depends on:** B2.
- **Effort:** 1 h.

**B4. Cleaning report**
- **What:** automatically generate the funnel tables and the data dictionary.
- **Outputs:** `docs/03_cleaning_report.md` (generated, plus a short written explanation).
- **Done when:** every number in the report comes from `funnel.json`.
- **Depends on:** B2, B3.
- **Effort:** 1 h.

**B5. Exploration charts and EDA report**
- **What:** seven charts:
  1. tries per bug;
  2. success rate by AI;
  3. the older AI's success rate and reproduce-first share by evaluation day (the hidden version mixture);
  4. how often each AI follows each tip;
  5. steps per try and how often tries run out of room;
  6. languages, and success by language;
  7. how much of the variation in success comes from the bug vs. chance between tries.
- **Outputs:** `src/agent_tips/eda/`; `outputs/figures/eda/*.png|pdf`; `docs/04_eda_report.md`; `agent-tips eda`.
- **Done when:** the charts render with the non-interactive backend on all platforms, and each chart has a two- or three-sentence interpretation.
- **Depends on:** B2–B4.
- **Effort:** 4 h.

**Checkpoint B:** cleaning report and EDA report; plain-words summary to the owner. ✅ **Completed 2026-10-20**; awaiting the owner's review. Changes made during the phase:
- the cleaning code lives in `cleaning/sources.py` (SQL per dataset), `pipeline.py` and `report.py`;
- the tip-compliance chart moved to Step 5, because it needs labels;
- EDA PNGs are kept in `docs/figures/eda/`.

---

### Phase C: Features and label validation (Step 5)

**C1. One shared action labeler**
- **What:** merge the exploratory rules into one labeler for all three scaffolds. Labels:
  - create test/repro script;
  - create other file;
  - edit existing code;
  - run code;
  - run the project's test suite;
  - look/search;
  - think;
  - clean up;
  - submit.
- **Outputs:**
  - `src/agent_tips/features/labels.py`;
  - at least 150 unit-test cases (real commands with known labels, across 9 languages);
  - `data/processed/steps_labeled.parquet`;
  - `agent-tips features`.
- **Done when:** all unit tests pass, and re-deriving the exploratory arms reproduces their shares within 1 point.
- **Depends on:** B2.
- **Effort:** 5 h, plus about 20 min of machine time.

**C2. Label validation (no human needed)**
- **(a) Check against the computer's reply.** Confirm each label from the tool's response:
  - a "create" label must be followed by a file-created message;
  - an edit must succeed and the file must appear in the final patch;
  - a test-suite run must produce a test summary (e.g. pytest's "passed/failed" line or Go's "ok/FAIL").

  This gives an estimated precision for each label.
- **(b) Blind second labeling.**
  - Draw a stratified random sample of 600 actions: 200 per dataset, balanced across labels and languages.
  - The assistant labels them from the raw text without seeing the rule-based labels.
  - Measure agreement with Cohen's κ.
- **Rule:** the target is κ ≥ 0.8 and precision ≥ 0.95 for the labels used in the primary analyses. If a label falls short, fix the rules and re-validate, up to 3 rounds. After that, keep the measured error rates for the sensitivity analysis in D7.
- **Outputs:** `src/agent_tips/features/validate.py`; `outputs/results/label_validation.json`; `data/interim/label_sample.csv` (sample with both labelings); a section in `docs/05_feature_report.md`.
- **Done when:** the κ and precision table is published, and labels below target are either fixed or carried into D7.
- **Depends on:** C1.
- **Effort:** 6 h (labeling 600 actions takes most of it).

**C3. Issue features**
- **What:** for each issue: has a code block, has a traceback, names a file, text length, language.
- **Outputs:** `src/agent_tips/features/issues.py`; issue table.
- **Done when:** unit tests pass.
- **Depends on:** B2.
- **Effort:** 1 h.

**C4. Localization features (for RQ6, "why")**
- **What:**
  - Find the "gold files" (the files the real fix changed): from SWE-bench-extra patches for the older AI, and from `reference_patch` in NVIDIA's metadata.
  - For each try, record the first step at which it opens, views or edits a gold file. This is "time to find the right place".
- **Outputs:** `src/agent_tips/features/localization.py`; localization columns.
- **Done when:** at least 90% of issues are matched to their gold patch, and unmatched issues are listed.
- **Depends on:** A3, C1.
- **Effort:** 3 h.

**C5. Feature report**
- **What:** describe every feature, the label validation results and the known label errors.
- **Outputs:** `docs/05_feature_report.md`.
- **Done when:** the report is complete.
- **Depends on:** C1–C4.
- **Effort:** 1 h.

**Checkpoint C:** validated labels and the feature report; plain-words summary to the owner.

---

### Phase D: Model, analyze and evaluate (Steps 6–7)

**D1. Estimator library**
- **What:**
  - weighted within-group difference;
  - 95% intervals from resampling whole issues;
  - within-group permutation test;
  - Manski-style bounds for undated tries;
  - comparisons of exact-history pairs (forks).
- **Outputs:** `src/agent_tips/models/estimators.py`; tests on synthetic data with a known true effect (the estimate must fall within its interval about 95% of the time in simulation).
- **Done when:** the simulation tests pass.
- **Depends on:** A5.
- **Effort:** 4 h.

**D2. Regression models**
- **What:**
  - linear probability model with issue (× batch) fixed effects and standard errors clustered by issue;
  - conditional logistic regression as a cross-check;
  - interaction terms for RQ5;
  - discrete-time hazard model and Kaplan–Meier curves for RQ6.

  Everything uses `statsmodels`, which has wheels for every platform.
- **Outputs:** `src/agent_tips/models/regression.py` and `survival.py`; tests on synthetic data.
- **Done when:** the regression estimates agree with D1 on synthetic and real data, within the expected difference.
- **Depends on:** D1.
- **Effort:** 4 h.

**D3. Run all research questions**
- **What:** run P1–P3 and all secondary analyses exactly as frozen in A1, then apply Benjamini–Hochberg to the secondary analyses.

  | RQ | Analyses |
  | --- | --- |
  | RQ1 | positivity table, balance checks, placebo |
  | RQ2 | P1, P2 |
  | RQ3 | P3 and the full simple-vs-fair table |
  | RQ4 | other tips and their cost |
  | RQ5 | interactions by issue type and language |
  | RQ6 | context exhaustion and time to find the gold file |
- **Outputs:** `agent-tips model`; `outputs/results/estimates.parquet` (one row per estimate: RQ, AI, tip, design, estimate, CI, p-value, q-value, number of issues, number of tries); `docs/06_results.md`.
- **Done when:** every hypothesis in A1 has a result row, and every number in the report comes from the results file.
- **Depends on:** A1 (tag), C2–C4, D1, D2.
- **Effort:** 5 h, plus about 30 min of machine time.

**D4. Check against the exploratory results**
- **What:** compare the final estimates with the exploratory numbers (e.g. −3.5, +0.3, +3.0).
- **Outputs:** a table in `docs/06_results.md`.
- **Done when:** every difference larger than 0.5 points is explained (e.g. by a label fix).
- **Depends on:** D3.
- **Effort:** 1 h.

**D5. Evaluation suite**
- **What:**
  - **Older AI:** finer batches, permutation test, placebo, bounds for undated tries.
  - **All AIs:** leave-one-AI-out.
  - **Label errors:** simulate misclassification at the rates measured in C2 and re-estimate.
  - **NVIDIA's removed tries:** a "tipping point" analysis of how many removed failures in the test-first group would be needed to change the conclusion.
  - **Version mixing:** sensitivity to how the older AI's batches are defined.
- **Outputs:** `src/agent_tips/evaluation/`; `agent-tips evaluate`; `docs/07_evaluation.md`.
- **Done when:** every primary result has a robustness table with a plain-words verdict.
- **Depends on:** D3.
- **Effort:** 5 h, plus about 30 min of machine time.

**D6. Reproducibility check**
- **What:** delete all outputs, run `agent-tips run-all` twice, and check that the results files are identical (byte for byte or by hash).
- **Outputs:** a reproducibility section in `docs/07_evaluation.md`.
- **Done when:** the two runs match exactly.
- **Depends on:** D5.
- **Effort:** 1 h, plus about 1–2 h of machine time.

**Checkpoint D:** all research questions answered with final numbers, and robustness done; plain-words summary to the owner.

---

### Phase E: Communicate (Step 8)

**E1. Paper figures**

| # | Figure |
| --- | --- |
| F1 | How the fair comparison works |
| F2 | The hidden mixture of AI versions |
| F3 | Main result for the older AI |
| F4 | Forest plot for the newer AIs |
| F5 | Simple vs. fair estimate for every tip × AI (the headline chart) |
| F6 | Forest plot of the other tips |
| F7 | Cost of testing first |
| F8 | Why it hurt: running out of room, and time to find the right file |

All figures use a colorblind-safe palette and are saved as vector PDF plus PNG.
- **Outputs:** `src/agent_tips/viz/`; `outputs/figures/paper/`; `agent-tips figures`.
- **Done when:** every figure is generated from `estimates.parquet`.
- **Depends on:** D3, D5.
- **Effort:** 5 h.

**E2. Tables**

| # | Table |
| --- | --- |
| T1 | Datasets |
| T2 | Cleaning funnel |
| T3 | Primary estimates |
| T4 | All secondary estimates with q-values |

- **Outputs:** generated as both LaTeX and Markdown.
- **Done when:** the tables are generated, never hand-typed.
- **Depends on:** D3.
- **Effort:** 2 h.

**E3. Refresh related work**
- **What:** search the literature again for papers published since 2026-10-19, and update the positioning.
- **Outputs:** an updated "Closest related work" section.
- **Done when:** the search date is recorded and any new overlapping papers are handled in the text.
- **Depends on:** —
- **Effort:** 2 h.

**E4. Paper draft**
- **What:** write the draft in `paper/` (LaTeX, `acmart`), with these sections:
  1. Introduction
  2. Related work
  3. Data
  4. Fair-comparison method
  5. Results (RQ1–RQ6)
  6. Threats to validity, including the AI-assisted label validation
  7. Discussion
  8. Conclusion
  9. Data and code availability

  Figures and tables are included directly from `outputs/`.
- **Done when:** the draft compiles, and every number in it traces back to the results file.
- **Depends on:** E1–E3.
- **Effort:** 10–12 h.

**E5. Release**
- **What:**
  - update the README;
  - deploy the documentation site to GitHub Pages;
  - update the changelog;
  - tag `v1.0.0` and create a GitHub release with the cleaned summary tables (small files only).
- **Done when:** CI passes on the tag, and someone with a fresh copy can run `uv sync && uv run agent-tips run-all`.
- **Depends on:** E4.
- **Effort:** 2 h.

**Checkpoint E:** the paper draft and the release; plain-words summary to the owner.

## 5. Dependency overview

```
A1 ─────────────────────────────────────────────┐
A2 → A3 → A4 → A5 ──────────────┐               │
             │                  ▼               ▼
             └→ B1 → B2 → B3 → B4 → B5        D1 → D2
                      │                         │
                      ├→ C1 → C2 ──────────────►├→ D3 → D4
                      ├→ C3 ───────────────────►│    │
                      └→ C1 + A3 → C4 ─────────►┘    └→ D5 → D6
                                                           │
E3 ───────────────────────────────────────────► E1, E2 ◄───┘ → E4 → E5
```

Estimated total: about 75–85 hours of assistant work plus 3–5 hours of machine time.

## 6. Risk register

| Risk | Effect | Prevention / response |
| --- | --- | --- |
| A dataset changes or disappears on Hugging Face | Results can't be reproduced | Pinned revisions and checksums (A2–A3); summary tables attached to the release (E5) |
| Download is slow or rate-limited | Delays Phase A | Retry with backoff; read only the needed columns; cache locally |
| Not enough memory for 255k NVIDIA tries with all their steps | Crashes | DuckDB processes data on disk; keep per-step text truncated; process one subset at a time |
| Label agreement (κ) below 0.8 | Weaker evidence | Fix the rules (up to 3 rounds); otherwise report the measured error rates and the D5 sensitivity analysis |
| Final numbers differ from the exploratory ones | Story could change | Analysis frozen in A1; D4 explains every difference; we report what the data says, even if the story changes |
| A new paper overlaps with ours | Novelty claim weakens | E3 refresh; emphasize the distinct parts (coin flips, version audit, six AIs, size of the trap) |
| NVIDIA's removed tries hide harm | Newer-AI result is incomplete | Stated as a limitation; tipping-point analysis (D5) |
| Testing many tip × AI pairs creates false positives | False findings | Primary analyses fixed in advance; Benjamini–Hochberg for secondary analyses |
| Platform differences (Windows paths, line endings, encodings) | CI failures | pathlib, explicit UTF-8, LF line endings via pre-commit, 4-platform CI on fixtures |
| GPU code paths untested in CI | Hidden bugs | GPU is optional and unused by the analyses; device selection has unit tests; CPU fallback always works |
| AI-assisted label validation is questioned by reviewers | Review risk | Disclose fully; publish the sample with both labelings; rely mainly on checks against the computer's replies |

## 7. Engineering standards (locked)

**In plain words:** the code is organized like a professional software project. It installs the same way on every computer with one command, it is checked automatically for mistakes, it is documented, and it is tested on Linux, Windows and macOS (both Intel/AMD and ARM chips). It can use a graphics card (GPU) for any model training, but it never requires one.

| Area | Standard |
| --- | --- |
| **Project structure** | `src/` layout with one installable package (`agent_tips`); command-line entry point `agent-tips <step>`; `tests/`, `docs/`, `configs/`, `notebooks/` kept separate from code |
| **Package management** | [uv](https://docs.astral.sh/uv/) with `pyproject.toml` and a committed `uv.lock`, which works on all target platforms. Python version pinned in `.python-version` (3.12). Optional dependency groups: `dev`, `docs`, `gpu`. |
| **Code quality** | `ruff` for linting and formatting; `mypy` type checks; type hints and docstrings (NumPy style) on all public functions; `pre-commit` hooks run these automatically |
| **Testing** | `pytest` with coverage. Unit tests for the labeler, the estimators (synthetic data with a known answer) and the cleaning funnel counts; an end-to-end test on the fixtures |
| **Documentation** | `README.md`; `docs/` built with MkDocs + mkdocstrings (step reports and API reference); data cards; `CONTRIBUTING.md`; `CHANGELOG.md`; `CITATION.cff`; `LICENSE` (MIT) |
| **Configuration** | Paths, seeds, dataset revisions and settings in `configs/*.yaml`, not hard-coded. All paths use `pathlib`, so Windows paths work. |
| **Platforms** | Linux x86_64, Linux ARM64, Windows x86_64, macOS ARM64. Only libraries with prebuilt wheels for all four. No shell-only scripts: every task runs through the Python CLI. |
| **Continuous integration** | GitHub Actions matrix: `ubuntu-latest` (x86_64), `ubuntu-24.04-arm` (ARM64), `windows-latest`, `macos-latest` (ARM64). Lint, type check and tests on every push, plus a docs build. |
| **GPU support** | Optional `gpu` extra (PyTorch: CUDA 12.8 on Linux and Windows, MPS on macOS). Device chosen automatically by `agent_tips.utils.device.select_device()`. The analyses run on CPU. |
| **Reproducibility** | Fixed seeds; pinned data revisions and checksums; `agent-tips run-all` reproduces every number, figure and table; byte-identical rerun check (D6) |
| **Version control** | Git and GitHub ([rajanyadav0307/tips_agentic_ai](https://github.com/rajanyadav0307/tips_agentic_ai)). Commits authored only by `rajanyadav0307`, with no assistant attribution. Each phase is pushed at its checkpoint. |

## 8. Folder layout

```
pyproject.toml   uv.lock   .python-version   README.md   LICENSE   CITATION.cff   CHANGELOG.md
.pre-commit-config.yaml   .gitignore   .github/workflows/ci.yml   mkdocs.yml
configs/         default.yaml (paths, seeds, dataset revisions)
src/agent_tips/
  cli.py         agent-tips download | clean | eda | features | model | evaluate | figures | run-all
  data/          download.py, load.py                        (Step 2)
  cleaning/      schema.py, older_ai.py, nebius.py, nvidia.py (Step 3)
  eda/           exploration charts                          (Step 4)
  features/      labels.py, validate.py, issues.py, localization.py (Step 5)
  models/        estimators.py, regression.py, survival.py    (Step 6)
  evaluation/    robustness and sensitivity                   (Step 7)
  viz/           paper figures and tables                     (Step 8)
  utils/         paths, seeds, device selection (CPU / CUDA / MPS)
tests/           unit tests, fixtures/, end-to-end test on fixtures
docs/            index.md, 01–07 step reports, project/ (plan, proposal, options, datasets), reference.md
notebooks/       read-only walkthroughs that call the package
data/            raw/ (+ MANIFEST.json), interim/, processed/   — not in git
outputs/         results/, figures/eda/, figures/paper/          — not in git (small summary tables go into the release)
analysis/exploratory/   frozen record of the first scripts and their results
paper/           LaTeX source of the paper
```

## 9. Definition of done (whole project)

- [ ] Every hypothesis in `docs/01_problem_and_hypotheses.md` has a final result in `estimates.parquet`.
- [ ] Every number, figure and table in the paper is produced by `agent-tips run-all` from pinned data.
- [ ] Label validation is reported (κ and precision), and its effect is covered by the sensitivity analysis.
- [ ] CI passes on all 4 platforms. Test coverage is at least 85% for `src/`.
- [ ] Reports 01–07 are in `docs/`, the docs site is deployed, and the README rerun instructions are verified on a fresh copy.
- [ ] Paper draft in `paper/` compiles, and threats to validity include the AI-assisted validation and NVIDIA's filtering.
- [ ] Release `v1.0.0` is tagged on GitHub.
