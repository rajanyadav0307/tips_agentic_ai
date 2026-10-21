# Project Plan 3: Can Machine Learning Fix One-Try Studies of AI Coding Agents?

**Version:** 2 (final check done), 2026-10-20
**Study:** the "one-try study" (Paper 2). It builds its own data tables and reports; it reuses Paper 1's tested *code* (downloader, fair-comparison estimator) where useful ([PROJECT_PLAN.md](PROJECT_PLAN.md)).
**Rules:**
- This study follows its own complete standard data science procedure (§5: problem, collection, cleaning, EDA, features, modelling, evaluation, communication), with a report at every step.
- The locked engineering standards apply unchanged (PROJECT_PLAN.md §7).
- Work happens on the git branch `one-try-study`.
- The assistant does all the work; the owner reviews at each checkpoint, and the next phase starts only after approval.
- Commits and pushes come only from `rajanyadav0307`, with no assistant attribution.

## In plain words

- **The problem:** most studies of AI coding tools see each bug only **once**. They compare what the AI did with whether it fixed the bug, and draw conclusions like "testing first helps". These conclusions can be wrong, because the AI may do that thing more often on **easy** bugs.
- **What our data can do:** the same AI tried each bug **many times**, so for each behaviour we can work out the right answer.
- **The study:**
  1. find the right answers using all tries;
  2. pretend we have one try per bug, like other researchers;
  3. test which machine-learning corrections get back to the right answer.
- **What a small test already showed:**
  - with one try per bug, the usual comparison was off by about 18 points;
  - machine learning on the bug report was still off by about 8;
  - adding **how other AIs did on the same bug** brought the error under 1 point.
- **What we deliver:** a paper and code that tell researchers when one-try studies can be trusted, and how to correct them.

## How the machine learning works (in plain words)

**The goal:** with only one try per bug, find out whether a habit such as "testing first" really helps, without being fooled by easy bugs.

**Example:** Bug A is easy and Bug B is hard. With one try each, the AI tested first on Bug A and fixed it, and did not test first on Bug B and failed. A simple comparison says "testing first fixes bugs", which is wrong: Bug A was simply easy.

The machine learning does five jobs:

1. **Turn the bug report into numbers.** A computer cannot read text directly.
   - **Simple way:** count the important words (TF-IDF), then squeeze them into 100 numbers (SVD). Similar reports get similar numbers.
   - **Stronger way (optional, uses the GPU if there is one):** a small language model reads the report and gives numbers that capture its meaning (sentence embeddings).
2. **Two helper predictions for every bug,** made with gradient boosting (many small decision trees; random forests as a check):
   - **Model A:** how likely is this bug to be fixed? (How easy does it look?)
   - **Model B:** how likely is the AI to test first on this bug?
3. **Remove the easy-bug effect (double machine learning).** For each try, keep only the *surprise*:
   - was it fixed more or less than Model A expected?
   - did it test first more or less than Model B expected?

   If trying "test first" *more than expected* goes with being fixed *more than expected*, the habit really helps. If not, it only looked helpful because of easy bugs.
4. **Give the models better information, step by step** (the "correction ladder" in §4): only the bug report → plus the size of the human fix → plus **how other AIs did on the same bug**. A small IRT model also turns the other AIs' results into one "difficulty score". In the pilot, the "other AIs" information worked best.
5. **Keep it honest:**
   - **cross-fitting:** each model is trained on some bugs and used on *other* bugs, so it cannot memorise answers;
   - **made-up data first:** every method is tested on fake data where the true answer is known, and must find it;
   - **real answer key:** every method is compared with the answer from our many-tries data, which the methods never see.

## 1. Is our data useful for this? (checked 2026-10-20)

| The study needs | What our data has | Verdict |
| --- | --- | --- |
| Many tries per bug, to know the right answer | Older AI: median 13 tries per bug (3,591 bugs); Nebius: 10 (6,306 bugs); NVIDIA: 2–3 (99,018 bugs) | ✅ |
| Behaviours that vary within a bug, to test | Comparisons with ≥100 usable bugs: NVIDIA 53 (7 AI setups), older AI 10, Nebius 1 | ✅ (mainly NVIDIA and the older AI) |
| Other AIs on the same bugs, for the correction | NVIDIA: 30,194 bugs tried by ≥2 setups; per-bug success agrees strongly between AIs (median correlation 0.73, range 0.64–0.81). Older AI vs Nebius: 1,203 shared bugs, weak agreement (0.28). Older AI's smaller models: 528 bugs (8B), 100 bugs (405B) | ✅ strong in NVIDIA; ⚠️ weak across generations |
| A reason the usual corrections may fail | The bug decides 51–87% of outcomes, but the bug report predicts hopeless bugs only weakly (AUC 0.59–0.64) | ✅ makes the question worth asking |

**Pilot** (NVIDIA, "test first", 4 AI setups, one try per bug, 5 random draws; numbers are points):

| AI setup | Many-tries answer | Simple | ML + report text | + human fix size | + other AIs' results |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6-27B / SWE-agent / scale-swe | −0.4 | +25.7 | +7.7 | +5.2 | +0.6 |
| Qwen3.5-122B / SWE-agent | +1.5 | +21.9 | +13.6 | +8.1 | +2.6 |
| Qwen3.5-122B / OpenHands | −1.0 | +14.1 | +4.2 | −0.4 | −2.2 |
| MiniMax-M2.5 / SWE-agent | +0.3 | +10.2 | +6.2 | +2.3 | +0.3 |
| **Average distance from the answer** | — | **17.9** | **7.8** | **3.7** | **0.8** |

**Weak spots, which become research questions rather than blockers:**
- other-AI information is weak across AI generations (older AI vs Nebius);
- NVIDIA deleted some failed tries;
- Nebius has only one behaviour that varies.

## 2. Novelty (searched 2026-10-20; re-check before writing)

- **No study found that:**
  - tests one-try analyses of AI agents against a many-tries answer;
  - uses other AIs' results on the same task to correct such analyses.
- **The classic design comes from economics:** non-experimental methods tested against a real experiment (LaLonde 1986; Smith & Todd 2005).
- **Closest work:**
  - within-task designs: [Same Task, Different Fate](https://doi.org/10.1145/3844178.3844642), [Beyond Resolution Rates](https://arxiv.org/abs/2604.02547);
  - task-difficulty models: [Agent Psychometrics](https://arxiv.org/abs/2604.00594), [What Makes Tasks Difficult](https://arxiv.org/abs/2608.18280);
  - single-run noise: [Identical Runs, Different Results](https://arxiv.org/abs/2609.33812).
- Full search notes: [ML_BRAINSTORM.md](ML_BRAINSTORM.md).

## 3. Research questions and hypotheses (frozen in Step 1, before any modelling run)

| RQ | Question | Hypothesis and decision rule |
| --- | --- | --- |
| RQ1 | With one try per bug, how far are the usual analyses from the many-tries answer? | **H1:** the average absolute error of the simple comparison is above 2 points (95% CI above 2) |
| RQ2 | Which correction fixes them? | **H2:** corrections that use only the bug report (rungs 1–3) leave an average error above 2 points. **H3:** adding other AIs' results (rung 5) brings the average error within ±2 points (equivalence test: 90% CI inside ±2, as in Paper 1) |
| RQ3 | When does the correction fail? | **H4:** its benefit shrinks when fewer other AIs tried the bug and when the other AIs agree less with this AI about which bugs are hard (pre-specified slope tests). The older AI vs Nebius is the planned hard case |
| RQ4 | How many tries per bug are enough without any correction? | Descriptive: error of the within-bug analysis using 2, 3, 5 and 10 tries per bug (older AI and Nebius only; NVIDIA has at most 3 tries per bug) |

- **Also reported:**
  - **confidence intervals:** do each method's 95% intervals contain the many-tries answer about 95% of the time?
  - **conclusions:** does the method reach the same conclusion (helps, hurts, or no difference)?
- **How averages get their CIs:** the error of each method is first averaged over draws within a comparison, then over comparisons. CIs come from 2,000 bootstrap resamples of comparisons; comparisons for the same AI setup are resampled together.
- **The answer key is not perfect:** each many-tries answer has its own uncertainty, which sets a "noise floor" for the error. It is reported next to every result, and the ±2-point margin is well above it (pilot answer keys: about ±0.6 points).
- **Multiple testing:** H1–H4 are Holm-corrected.

## 4. Study design

- **Comparisons:** behaviour × AI setup with at least 100 bugs where both choices occur. The current count is about 64:
  - NVIDIA 53;
  - older AI 10;
  - Nebius 1;
  - plus first-move choices.

  The final list comes from the EDA inventory (Step 4) and is fixed before modelling.
- **Many-tries answer (reference):** the within-bug fair comparison (Paper 1's estimator), using all tries. For the older AI the groups are bug × evaluation day, because of its hidden model versions. There are two kinds:
  - **true experiments:** first-move choices (step 0), where every try sees identical input;
  - **best available:** choices made mid-run.

  Results are reported separately for the two kinds.
- **One-try copies:**
  - keep one random try per bug, **from the same bugs as the answer key** so both describe the same bugs (secondary: all bugs);
  - repeat with different random draws (default 20, with a check that more draws change averages by less than 0.2 points);
  - for RQ4, keep 2, 3, 5 or 10 tries per bug.
- **Correction ladder** (the same for every comparison):

  | Rung | Correction | Information used |
  | --- | --- | --- |
  | 0 | Simple difference | — |
  | 1 | Linear regression | Bug report facts: length, code block, error trace, names a file, repository, language, task type |
  | 2 | Double machine learning with gradient boosting (random forests as a check) | + bug report text (TF-IDF → 100 dimensions) |
  | 3 | Same, with text embeddings (optional, GPU) | + sentence embeddings of the report |
  | 4 | Same | + size of the human fix (files and lines changed); available to anyone analysing a benchmark |
  | 5 | Same | + other AIs' success rate on the same bug, and how many other AIs tried it |
  | 6 | Same | Rung 5 with an IRT difficulty score fitted on the other AIs' tries |

  - **Check:** a propensity-weighting (AIPW) version of rungs 2–6.
- **No-leakage rules:**
  - features never contain the studied AI's own results on the bug;
  - other-AI features always leave out the AI being studied;
  - cross-fitting folds are split by bug;
  - the many-tries answer is never used to tune anything.
- **Scores per method:**
  - error (estimate minus answer) and absolute error;
  - sign agreement and conclusion agreement;
  - interval coverage.

  Scores are averaged over draws and comparisons, by dataset, by behaviour type, and for experiments vs non-experiments.

## 5. Standard data science procedure for this study

This study runs its **own** complete procedure, with its own code, data files and report at every step. It does not borrow finished tables from Paper 1. It may reuse Paper 1's *code* (for example the downloader and the fair-comparison estimator), but every table it uses is rebuilt and reported here.

| Step | Name | Question it answers | Report |
| --- | --- | --- | --- |
| 1 | Problem definition | What exactly do we test, and how do we judge success? | `docs/one_try/01_problem.md` |
| 2 | Data collection | Which raw data do we need, where does it come from, and is it intact? | `docs/one_try/02_data_collection.md` |
| 3 | Data cleaning | Which records are usable, and why were the others removed? | `docs/one_try/03_cleaning.md` |
| 4 | Exploratory data analysis (EDA) | What does the data look like, and which comparisons are possible? | `docs/one_try/04_eda.md` |
| 5 | Feature engineering | What information may each correction method use, without leaks? | `docs/one_try/05_features.md` |
| 6 | Modelling | The answer key, the one-try copies and the correction methods | `docs/one_try/06_modelling.md` |
| 7 | Evaluation | Which method gets closest to the answer, and how robust is that? | `docs/one_try/07_evaluation.md` |
| 8 | Communication | Figures, tables, paper and release | `paper2/`, `docs/one_try/08_results.md` |

Each step is one command, `agent-tips onetry <step>`. `agent-tips onetry run-all` runs all steps in order. Data for this study lives in `data/onetry/` and results in `outputs/onetry/`.

### Step 1: Problem definition
- **Do:**
  - write the research questions RQ1–RQ4 and hypotheses H1–H4 (§3);
  - define the target ("many-tries answer"), the metrics, the ±2-point margin, the correction ladder (§4) and the decision rules;
  - state the scope and assumptions;
  - re-check novelty.
- **Output:** `01_problem.md`, tagged `one-try-plan-v1` before any modelling run.
- **Done when:** the tag exists; every hypothesis has a test and a decision rule.

### Step 2: Data collection
- **Do:**
  - list the sources this study needs:
    - the three trajectory datasets;
    - the human fixes (gold patches for the older AI; NVIDIA reference patches);
    - all AI sizes in the older dataset (8B, 70B, 405B) as "other AIs";
  - verify the pinned downloads (revisions, SHA-256);
  - extract only the fields this study needs into `data/onetry/raw/`:
    - one row per try: bug ID, AI setup, outcome, actions per step;
    - one row per bug: report text and human fix.
- **Output:** `02_data_collection.md`, with sources, licences, revisions, fields used and row counts per source; extraction manifest with checksums.
- **Done when:** every source is pinned and verified, and row counts match the source datasets.

### Step 3: Data cleaning
- **Do:** apply written rules, each counted in a funnel:
  - **outcome:** keep tries whose outcome is known;
  - **identical input:** keep tries whose bug text matches the most common text for that bug;
  - **bug ID:** harmonise bug IDs across datasets, so "other AIs on the same bug" is matched correctly;
  - **report text:** strip the agent framework's instruction wrapper, so only the real bug report remains;
  - **fix size:** parse the human fix into files and lines changed;
  - **duplicates:** remove them;
  - **checks:** check for missing values and outliers (for example extreme step counts);
  - **version flag:** mark the older AI's evaluation day as the hidden model-version proxy.
- **Output:** `data/onetry/clean/{tries,bugs,setups}.parquet`; `03_cleaning.md` with the funnel, the rules and a data dictionary.
- **Done when:** every removed row is explained by a rule; unit tests cover each rule.

### Step 4: Exploratory data analysis
- **Do:**
  - outcomes by AI setup;
  - how much the bug vs luck decides the outcome;
  - how strongly different AIs agree about which bugs are hard (heat map);
  - how well the report alone hints at difficulty;
  - how often each behaviour happens, and in how many bugs both choices occur;
  - which comparisons qualify (at least 100 such bugs), first-move experiments included;
  - how much other-AI information each comparison has.
- **Output:** `04_eda.md` with charts and a comparison inventory (`outputs/onetry/inventory.parquet`). Counts and descriptions only; no method results.
- **Done when:** every qualifying comparison is listed with its counts, and every chart's numbers are saved.

### Step 5: Feature engineering
- **Do:** build the information the correction methods may use, per bug:
  - report facts (length, code block, error trace, names a file, repository, language, task type);
  - report text (TF-IDF → 100 dimensions; optional sentence embeddings, GPU if available);
  - human-fix size;
  - other-AI features: the leave-this-AI-out success rate, the number of other AIs, and an IRT difficulty score.
- **Leakage tests:**
  - the studied AI's own results never appear in its features;
  - text models are fitted inside training folds only.
- **Output:** `data/onetry/features/`; `05_features.md` (feature dictionary, coverage, leakage checks).
- **Done when:** the leakage tests pass, and the coverage of every feature is reported.

### Step 6: Modelling
- **Do:**
  - **answer key:** the many-tries answer for every qualifying comparison (fair within-bug comparison, with CIs);
  - **one-try simulator:** one try per bug, many random draws; k tries per bug for RQ4;
  - **correction ladder (§4):**
    - simple comparison;
    - linear regression;
    - double machine learning with gradient boosting and with random forests;
    - a propensity-weighted (AIPW) check;
  - fixed hyper-parameters and cross-fitting split by bug;
  - **synthetic data with a known truth:** each method must be right when it has the needed information, and wrong in the expected direction when it does not;
  - **go/no-go:** reproduce the pilot.
- **Output:** `src/agent_tips/onetry/{references,simulate,corrections}.py`; `06_modelling.md`.
- **Done when:** the synthetic tests pass on CI and the pilot is reproduced.

### Step 7: Evaluation
- **Do:**
  - **full run:** all comparisons × draws × methods;
  - **scores:** error, absolute error, sign and conclusion agreement, and interval coverage;
  - **results:** RQ1–RQ4 with Holm correction; the hard case (older AI vs Nebius);
  - **robustness:** learners, text dimensions, number of draws, NVIDIA's filtering, experiments vs mid-run choices;
  - **reproducibility:** two full runs must give identical results.
- **Output:** `outputs/onetry/results.parquet`; `07_evaluation.md`.
- **Done when:** every hypothesis has a result and a plain-words verdict, and the two runs match.

### Step 8: Communication
- **Do:**
  - figures (the easy/hard bug example, error by method, when the correction fails, how many tries are enough, interval coverage) and tables, all generated;
  - `numbers.tex`;
  - Paper 2 draft in `paper2/`;
  - docs site pages and README section;
  - CHANGELOG and release `v2.0.0`.
- **Done when:** the paper compiles, every number comes from the results files, and CI passes on the tag.

## 6. Phases and checkpoints

| Phase | Steps | Checkpoint (owner approves before the next phase) |
| --- | --- | --- |
| 0 | Setup | Branch `one-try-study` created ✅. Paper 1's local work stays unpublished until the owner decides |
| A | 1 Problem, 2 Collection | Rules frozen and tagged; data sources verified and extracted. ✅ Done and approved 2026-10-20 |
| B | 3 Cleaning, 4 EDA | Clean tables with funnel; EDA report and comparison inventory. ✅ Done and approved 2026-10-20 |
| C | 5 Features | Feature tables with leakage checks. ✅ Done 2026-10-20; awaiting the owner's review |
| D | 6 Modelling, 7 Evaluation | Answer key, methods, full results, robustness |
| E | 8 Communication | Paper 2 draft and release |

At each checkpoint: a plain-words summary, a short "what to check" list, and a push of the branch (commits only as `rajanyadav0307`, no assistant attribution).

## 7. Engineering (same standards as Paper 1)

- **Code:**
  - `src/agent_tips/onetry/`: `collect.py`, `clean.py`, `eda.py`, `features.py`, `references.py`, `simulate.py`, `corrections.py`, `evaluate.py`, `figures.py`, `report.py`;
  - CLI `agent-tips onetry <step>`.
- **Config:** a `onetry:` section in `configs/default.yaml` (draws, folds, margin, minimum bugs per comparison, seeds).
- **New dependency:** `scikit-learn` (installers for all 4 platforms); sentence embeddings stay optional in the `gpu` extra.
- **Checks:**
  - unit tests per cleaning rule and per method;
  - synthetic-truth tests;
  - an end-to-end test on a fake project;
  - coverage ≥ 85%;
  - ruff, mypy and pre-commit;
  - CI on 4 platforms;
  - fixed seeds and row order.
- **Docs:** each step report is generated by code and listed in the MkDocs navigation under "Paper 2".

## 8. Risks

| Risk | What we do |
| --- | --- |
| NVIDIA deleted failed tries | The comparison is internal to the same data, so it stays fair; we report this and check the pattern in the unfiltered older-AI data |
| Mid-run answers are not true experiments | Report first-move experiments separately; claims rest mainly on them |
| Other-AI information is weak across generations | This is RQ3; we report honestly when the correction does not help |
| Long computing time | 20 draws by default with a precision check, runs in parallel, staged runs |
| Someone publishes the same idea first | Re-check before writing; our distinct parts are the many-tries answer, the cross-AI correction, and when it fails |
| Other AIs' results are not always available | The paper states when the correction applies (benchmarks with many agents' logs) and when it does not (brand-new tasks) |

## 9. Definition of done

- [ ] RQ1–RQ4 answered from `outputs/onetry/results.parquet`; rules frozen (tag `one-try-plan-v1`) before any modelling run.
- [ ] Results reported separately for true experiments and mid-run choices, and for each dataset.
- [ ] Every number, figure and table in Paper 2 is produced by the pipeline.
- [ ] Tests ≥ 85% coverage; CI passes on 4 platforms; two runs give identical results.
- [ ] Paper 2 draft compiles; release `v2.0.0` tagged.
