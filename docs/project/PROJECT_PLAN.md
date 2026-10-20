# Project Plan: From Current State to Paper-Ready Results

**Created:** 2026-10-19
**Project:** *Do Tips for AI Coding Agents Actually Help?* (details in [DS_PROJECT_PROPOSAL.md](DS_PROJECT_PROPOSAL.md))
**Rule:** all work follows the locked 8-step procedure below, in order. Each step ends with a written output.

## In plain words

- **Where we are:** we already have strong first results (the tip doesn't help or even hurts, and simple analysis is badly misleading). They came from quick exploratory scripts.
- **What's missing for a proper data science project:**
  - a documented cleaning step;
  - charts that explore the data;
  - a human check of the labels;
  - proper statistical models;
  - final figures and the paper itself.
- **The plan:** rebuild everything as one clean pipeline, step by step. The final numbers in the paper will come from this pipeline, not from the quick scripts.
- **Your part:** hand-label about 200 examples for each AI recording dataset, which takes roughly 2–3 hours in total. Also choose a target venue.
- **The finish line:** every research question answered, 8 figures, 4 tables, and a full paper draft.
- **Built to professional standards:**
  - one-command install on Linux, Windows and macOS (Intel/AMD and ARM);
  - automatic code checks and tests;
  - full documentation;
  - optional GPU use for any training (see "Engineering standards").

## The locked procedure

| Step | Name | Output of the step |
| --- | --- | --- |
| 1 | Define the problem | Research questions, hypotheses, and the main analysis fixed **before** the final runs |
| 2 | Collect data | Data inventory: sources, versions, licenses, download dates |
| 3 | Clean and prepare | Cleaning report: what was removed, why, and how many rows were left at each stage |
| 4 | Explore (EDA) | Exploration charts and a short EDA report |
| 5 | Create features | One shared labeling module; issue features; human label check with agreement score |
| 6 | Model and analyze | Main estimates, regression models, multiple-testing correction, subgroup and "why" analyses |
| 7 | Evaluate | Robustness checks, sensitivity to label errors, reproducibility test |
| 8 | Communicate | Final figures and tables, paper draft, cleaned code and data release |

## Current state (2026-10-19)

| Step | Status | What exists | What's missing |
| --- | --- | --- | --- |
| 1 Problem | ✅ Mostly done | Six research questions in the proposal | Written hypotheses; the main analysis fixed in advance |
| 2 Data | ✅ Mostly done | Older AI (local, 80k tries); Nebius newer AI (local, 67k); NVIDIA (streamed, 255k summaries); dataset cards | Gold patches from nebius/SWE-bench-extra (needed for RQ6); one inventory file |
| 3 Clean | ⚠️ Partial | Filters exist inside several scripts | One cleaning pipeline and a cleaning report (counts at each stage) |
| 4 EDA | ❌ Missing | Numbers only | Charts and an EDA report |
| 5 Features | ⚠️ Partial | Separate regex labelers in each script; issue features for the older AI | One shared labeler; human label check; localization features |
| 6 Model | ⚠️ Exploratory | Same-bug comparisons, bootstrap intervals, permutation tests | Regression models, multiple-testing correction, survival analysis for RQ6 |
| 7 Evaluate | ⚠️ Partial | Robustness checks for the older AI | Checks for the newer AIs; sensitivity to label errors; full rerun from scratch |
| 8 Communicate | ❌ Missing | Text reports | Figures, tables, paper |

## The plan, step by step

### Step 1: Define the problem (finalize and freeze)

1. Write each research question as a testable hypothesis. Example: *H2a: for the older AI, reproduce-first lowers the resolution rate compared with search-first.*
2. Fix the **main analyses** in advance, so the final results can't be cherry-picked:
   - **Primary 1:** older AI, reproduce-first vs. search-first at step 0; same bug and same evaluation day; outcome = resolved.
   - **Primary 2:** newer AIs (NVIDIA), test-first vs. fix-first; same bug; outcome = resolved.
   - **Primary 3:** the gap between simple and fair estimates across all tips and AIs.
   - Everything else is secondary and gets a false-discovery correction.
3. **Output:** `docs/01_problem_and_hypotheses.md`.

### Step 2: Collect data

1. Download the nebius/SWE-bench-extra metadata (gold patches and issue dates) for the older AI's issues. NVIDIA's data already includes reference patches.
2. Write a data inventory listing each source, its version and download date, size, license, and what it is used for.
3. **Output:** `docs/02_data_inventory.md`.

### Step 3: Clean and prepare

1. Build one pipeline (`analysis/pipeline/step3_clean.py`) that produces three clean tables, one per dataset, with identical column names.
2. Record a **funnel** for each dataset, i.e. the row count after each filter:
   - **Older AI:** all tries → 70B model only → same issue text → dated vs. undated.
   - **Nebius newer AI:** all tries → same issue text.
   - **NVIDIA:** all tries → known outcome (not −1) → normalized language names.
3. Write a data dictionary describing every column.
4. **Outputs:** `analysis/clean/*.parquet`, `docs/03_cleaning_report.md` (funnel tables and data dictionary).

### Step 4: Explore the data (EDA)

Charts, saved to `figures/eda/`:

1. Tries per bug, for each dataset.
2. Success rate by AI.
3. Success rate and reproduce-first share by evaluation day for the older AI. This shows the hidden mixture of AI versions.
4. How often each AI follows each tip (compliance from 10% to 98%).
5. Number of steps per try, and how often tries run out of room.
6. Bugs by programming language, and success rate by language.
7. How much of the variation in success comes from the bug itself vs. chance between tries.

**Output:** `docs/04_eda_report.md`, with each chart and two or three sentences on what it shows.

### Step 5: Create features

1. **One shared labeling module** (`analysis/pipeline/labels.py`) used for all datasets. It labels each action as: create a test or reproduction script, edit code, run code, run the test suite, look or search, clean up, or submit.
2. **Human label check:**
   - Draw a random sample of about 200 actions per dataset, spread across label types.
   - You label them in a simple spreadsheet; I prepare it with clear instructions.
   - Measure agreement with Cohen's κ (target ≥ 0.8) and fix the labeler wherever it disagrees.
3. **Issue features:** has a code block, has a traceback, names a file, length, language.
4. **Localization features** (RQ6): the step at which the AI first opens the file that the real fix changed.
5. **Outputs:** `analysis/features/*.parquet`, `docs/05_feature_report.md` (including the κ table).

### Step 6: Model and analyze

| RQ | Analysis | Method |
| --- | --- | --- |
| RQ1 Fair design | When does a coin flip exist? | Positivity table, balance checks, placebo |
| RQ2 Main tip | Primary 1 and 2 | Stratified same-bug differences, plus **regression with bug fixed effects** (linear probability and conditional logistic) with standard errors clustered by bug; exact-fork estimates |
| RQ3 Simple vs. fair | Bias of the simple method | Simple and fair estimates for every tip × AI, the gap with its interval, and Simpson's paradox across AIs |
| RQ4 Other tips and cost | Check-your-fix, project tests, extra tests, cleanup; steps and text | Same fixed-effects models; **Benjamini–Hochberg** correction across all secondary tests |
| RQ5 Bug types and languages | Does the effect vary? | Interaction terms in the regression; optional causal forest |
| RQ6 Why | Running out of room; finding the right file later | Fixed-effects models on mechanism outcomes; **survival analysis** (Kaplan–Meier and a discrete-time hazard model) of time until the right file is found |

**Output:** `analysis/pipeline/step6_models.py` and `docs/06_results.md`, with all estimates in one table.

### Step 7: Evaluate

1. Robustness for the older AI: finer time batches, permutation test, placebo, bounds for undated tries. These exist and get rerun inside the pipeline.
2. Robustness for the newer AIs: leave-one-AI-out, label-error sensitivity using the κ results, and a check on the share of unknown outcomes.
3. A sensitivity analysis for NVIDIA's removed tries: how large would a hidden harm have to be to change the conclusion?
4. Reproducibility: delete all outputs, rerun the whole pipeline with one command, and confirm identical numbers.
5. **Output:** `docs/07_evaluation.md`.

### Step 8: Communicate

**Figures** (`figures/paper/`):

| # | Figure |
| --- | --- |
| F1 | How the fair comparison works (same bug, many tries, coin flip) |
| F2 | Hidden mixture of AI versions in the older dataset |
| F3 | Main result for the older AI (simple vs. fair, with intervals) |
| F4 | Forest plot: test-first effect for each newer AI |
| F5 | **Simple vs. fair estimate for every tip × AI**, the headline chart for "the trap" |
| F6 | Other tips (forest plot) |
| F7 | Cost of testing first |
| F8 | Why it hurt: running out of room and time until the right file is found |

**Tables:** T1 datasets; T2 cleaning funnel; T3 main estimates; T4 all secondary estimates with FDR.

**Paper draft** (`paper/`):

1. Introduction
2. Related work
3. Data
4. Fair-comparison method
5. Results (RQ1–RQ6)
6. Threats to validity
7. Discussion
8. Conclusion

**Release:** a README with one rerun command, the cleaned tables, and the code.

## Engineering standards (locked)

**In plain words:** the code will be organized like a professional software project. It installs the same way on every computer with one command, it is checked automatically for mistakes, it is documented, and it is tested on Linux, Windows and macOS (both Intel/AMD and ARM chips). It can use a graphics card (GPU) for any model training, but it never requires one.

| Area | Standard |
| --- | --- |
| **Project structure** | `src/` layout with one installable package (`agent_tips`); command-line entry point `agent-tips <step>`; `tests/`, `docs/`, `configs/`, `notebooks/` kept separate from code |
| **Package management** | [uv](https://docs.astral.sh/uv/) with `pyproject.toml` and a committed `uv.lock`, which works on all target platforms. Python version pinned in `.python-version` (3.12). Optional dependency groups: `dev`, `docs`, `gpu`. |
| **Code quality** | `ruff` for linting and formatting; `mypy` type checks; type hints and docstrings (NumPy style) on all public functions; `pre-commit` hooks run these automatically |
| **Testing** | `pytest` with coverage. Unit tests for the labeler (known commands → known labels), the estimators (small synthetic data with a known answer) and the cleaning funnel counts |
| **Documentation** | `README.md` (install, one-command rerun); `docs/` built with MkDocs + mkdocstrings (step reports and API reference); data cards; `CONTRIBUTING.md`; `CHANGELOG.md`; `CITATION.cff`; `LICENSE` |
| **Configuration** | Paths, seeds and settings in `configs/*.yaml`, not hard-coded. All paths use `pathlib`, so Windows paths work. |
| **Platforms** | Linux x86_64, Linux ARM64, Windows x86_64, macOS ARM64. Only libraries with prebuilt wheels for all four (duckdb, pyarrow, pandas/polars, numpy, scipy, statsmodels, scikit-learn, matplotlib). No shell-only scripts: every task runs through the Python CLI. |
| **Continuous integration** | GitHub Actions matrix: `ubuntu-latest` (x86_64), `ubuntu-24.04-arm` (ARM64), `windows-latest`, `macos-latest` (ARM64). Runs lint, type check and tests on a small sample dataset on every push. |
| **GPU support** | Training is **optional**; the current statistical analyses need only a CPU. Any training code (e.g., a learned action labeler) uses PyTorch from the `gpu` group. Devices are picked automatically: CUDA on Linux and Windows x86_64 and Linux ARM64 (NVIDIA), Apple MPS on macOS, otherwise CPU. CUDA wheel sources are configured per platform in `pyproject.toml`. |
| **Reproducibility** | Fixed seeds; raw data downloaded by `agent-tips download` with recorded versions and checksums; raw data, large outputs and the virtual environment are excluded from git; `agent-tips run-all` reproduces every number, figure and table |
| **Version control** | Git repository (the project folder isn't one yet); small commits with clear messages; the work for each phase on its own branch |

## Order of work and checkpoints

| Phase | Steps | Checkpoint (we review together before moving on) |
| --- | --- | --- |
| 0 | Setup | ✅ **Done 2026-10-19.** Repository, package, `uv` environment, linting, tests and CI all pass on Linux x86_64, Linux ARM64, Windows x86_64 and macOS ARM64. Current scripts moved into `analysis/exploratory/`. MIT `LICENSE` and `CITATION.cff` (Rajan Yadav) added. |
| A | 1 + 2 | Hypotheses and the main analysis frozen; data inventory complete |
| B | 3 + 4 | Cleaning report and EDA charts reviewed |
| C | 5 | Label check done by you; κ ≥ 0.8 |
| D | 6 + 7 | All RQs answered with final numbers; robustness passed |
| E | 8 | Figures, tables and paper draft |

Phase C depends on your labeling. Everything else I can run, then show you the results at each checkpoint.

## Folder layout (target)

```
pyproject.toml   uv.lock   .python-version   README.md   LICENSE   CITATION.cff   CHANGELOG.md
.pre-commit-config.yaml   .gitignore   .github/workflows/ci.yml
configs/         default.yaml (paths, seeds, dataset versions)
src/agent_tips/
  cli.py         `agent-tips download | clean | eda | features | model | evaluate | figures | run-all`
  data/          download, load (step 2)
  cleaning/      funnels and clean tables (step 3)
  eda/           exploration charts (step 4)
  features/      labels.py (shared labeler), issue and localization features (step 5)
  models/        estimators, fixed-effects models, survival, FDR (step 6)
  evaluation/    robustness and sensitivity (step 7)
  viz/           paper figures and tables (step 8)
  utils/         paths, seeds, device selection (CPU / CUDA / MPS)
tests/           unit tests plus a small sample dataset
docs/            MkDocs site: step reports 01–07, data cards, API reference
notebooks/       read-only walkthroughs that call the package
data/            raw/ (not in git), interim/, processed/
outputs/         results/, figures/eda/, figures/paper/
analysis/exploratory/   the current quick scripts, kept for the record
paper/
```

The steps in the plan above produce exactly these files. For example, `analysis/pipeline/step3_clean.py` becomes `src/agent_tips/cleaning/`, run with `agent-tips clean`.

## Decisions needed from you

1. **Target venue**, which sets the paper's length and format (e.g., MSR, ESEM, KDD Applied Data Science, or a workshop).
2. **Who does the label check:** you alone, or two people (two people allow a stronger agreement score).
3. **Optional:** whether to include closed AIs (GPT, Claude) through a small paid experiment. This is not needed for the current story.
