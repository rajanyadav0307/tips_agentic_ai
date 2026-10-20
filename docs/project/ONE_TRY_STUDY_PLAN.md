# Project Plan 3: Can Machine Learning Fix One-Try Studies of AI Coding Agents?

**Version:** 1 (draft for the owner's approval), 2026-10-20
**Study:** the "one-try study" (Paper 2). It reuses the data pipeline of Paper 1 ([PROJECT_PLAN.md](PROJECT_PLAN.md)).
**Rules:**
- The locked 8-step data science procedure and the locked engineering standards apply unchanged (PROJECT_PLAN.md §1 and §7).
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

## 3. Research questions and hypotheses (frozen in Phase 1, before any full run)

| RQ | Question | Hypothesis and decision rule |
| --- | --- | --- |
| RQ1 | With one try per bug, how far are the usual analyses from the many-tries answer? | **H1:** the average absolute error of the simple comparison is above 2 points (95% CI above 2) |
| RQ2 | Which correction fixes them? | **H2:** corrections that use only the bug report leave an average error above 2 points. **H3:** adding other AIs' results brings the average error within ±2 points (equivalence test, as in Paper 1) |
| RQ3 | When does the correction fail? | **H4:** its benefit shrinks when fewer other AIs tried the bug and when the other AIs agree less with this AI about which bugs are hard (pre-specified slope tests). The older AI vs Nebius is the planned hard case |
| RQ4 | How many tries per bug are enough without any correction? | Descriptive: error of the within-bug analysis using 2, 3, 5 and 10 tries per bug |

- **Also reported:**
  - **confidence intervals:** do each method's 95% intervals contain the many-tries answer about 95% of the time?
  - **conclusions:** does the method reach the same conclusion (helps, hurts, or no difference)?
- **Multiple testing:** H1–H4 are Holm-corrected.

## 4. Study design

- **Comparisons:** behaviour × AI setup with at least 100 bugs where both choices occur. The current count is about 64:
  - NVIDIA 53;
  - older AI 10;
  - Nebius 1;
  - plus first-move choices, which are added in Phase 1.
- **Many-tries answer (reference):** the within-bug fair comparison from Paper 1, using all tries. There are two kinds:
  - **true experiments:** first-move choices (step 0), where every try sees identical input;
  - **best available:** choices made mid-run.

  Results are reported separately for the two kinds.
- **One-try copies:**
  - keep one random try per bug;
  - repeat with different random draws (default 20, with a check that more draws change averages by less than 0.2 points);
  - for RQ4, keep 2, 3, 5 or 10 tries per bug.
- **Correction ladder** (the same for every comparison):

  | Rung | Correction | Information used |
  | --- | --- | --- |
  | 0 | Simple difference | — |
  | 1 | Linear regression | Bug report facts: length, code block, error trace, names a file, repository, language, task type |
  | 2 | Double machine learning with gradient boosting | + bug report text (TF-IDF → 100 dimensions) |
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

## 5. Phases (each ends with a plain-words summary and the owner's approval)

### Phase 0: Housekeeping
- **What:** Paper 1 is finished but only saved locally (3 commits, not pushed). Default: push it and tag `v1.0.0` first, so Paper 2 starts from a clean, released base.
- **Done when:** the owner decides; then CI passes.

### Phase 1: Problem and data (Steps 1–2)
- **1.1 Freeze the rules:** RQ1–RQ4, hypotheses, the correction ladder, metrics and decision rules go into `docs/one_try/01_problem.md`, tagged `one-try-plan-v1` before any full run.
- **1.2 Data inventory:** list every qualifying comparison (behaviour × AI), add first-move comparisons, and record how many bugs have other-AI results and how strongly the AIs agree. Output: `docs/one_try/02_data.md`.
- **1.3 Novelty re-check,** dated, in the inventory.
- **Done when:** the tag exists and the inventory lists every comparison with its counts.

### Phase 2: Prepare, explore, features (Steps 3–5)
- **2.1 New tables** (raw data → `data/processed/one_try/`):
  - `bug_text.parquet`: report text, repository, language, task type;
  - `fix_size.parquet`: human fix size, from NVIDIA's reference patches and the older AI's gold patches;
  - `cross_ai.parquet`: for each bug and AI, the other AIs' success rate (leave-this-AI-out), the number of other AIs, and their agreement.
- **2.2 Exploration** (`docs/one_try/04_eda.md`):
  - how much the bug vs luck decides outcomes;
  - how strongly AIs agree on difficulty (heat map);
  - how weakly the report predicts difficulty.
- **2.3 Text features:** TF-IDF → SVD (cached); optional sentence embeddings (GPU if available, CPU fallback).
- **Done when:** the tables pass tests (no leakage: the studied AI never appears in its own other-AI features) and the EDA report is generated.

### Phase 3: Model (Step 6)
- **3.1** Many-tries answers for all comparisons (reusing Paper 1's estimator) → `references.parquet`.
- **3.2** One-try simulator (k tries per bug, fixed seeds).
- **3.3** Correction ladder in `corrections.py`.
  - **Synthetic tests with a known truth:** each method must be right when it has the information it needs, and wrong in the expected direction when it does not.
- **3.4 Go/no-go:** reproduce the pilot within the code base on a small subset.
- **Done when:** the synthetic tests pass on CI and the pilot is reproduced.

### Phase 4: Evaluate (Step 7)
- **4.1** Full run: all comparisons × draws × rungs → `one_try_results.parquet`. Runs in the background with checkpoints; the target is under 1 day of laptop time.
- **4.2** RQ1–RQ4, interval coverage, conclusion agreement, and the hard case (older AI vs Nebius).
- **4.3 Robustness:**
  - learners (boosting, random forest, linear);
  - text dimensions;
  - number of draws;
  - NVIDIA's filtering;
  - experiments vs non-experiments.
- **4.4** Reproducibility: two full runs, identical results.
- **Done when:** every hypothesis has a result row and the robustness table has a plain-words verdict.

### Phase 5: Communicate (Step 8)
- **5.1 Figures** (a generated `numbers.tex` holds every paper number):
  - the easy/hard bug example;
  - error by correction (main figure);
  - when the correction fails;
  - how many tries are enough;
  - interval coverage.
- **5.2 Tables:** data, comparisons, main results.
- **5.3** Paper 2 draft in `paper2/` (ACM format).
- **5.4** Docs pages, README section, CHANGELOG, release `v2.0.0`.
- **Done when:** the paper compiles, every number comes from the results files, and CI passes on the tag.

## 6. Engineering (same standards as Paper 1)

- **Code** in the same package:
  - `src/agent_tips/onetry/`: `data.py`, `references.py`, `simulate.py`, `corrections.py`, `evaluate.py`, `report.py`, `figures.py`;
  - CLI `agent-tips onetry prepare | run | evaluate | figures`, also callable from `run-all`.
- **New dependency:** `scikit-learn` (installers for all 4 platforms). Text embeddings stay optional in the `gpu` extra.
- **Checks:**
  - tests with synthetic truths, plus an end-to-end test on the larger fake project;
  - coverage ≥ 85%;
  - ruff, mypy, and CI on 4 platforms;
  - fixed seeds and fixed row order.

## 7. Risks

| Risk | What we do |
| --- | --- |
| NVIDIA deleted failed tries | The comparison is internal to the same data, so it stays fair; we report this and check the pattern in the unfiltered older-AI data |
| Mid-run answers are not true experiments | Report first-move experiments separately; claims rest mainly on them |
| Other-AI information is weak across generations | This is RQ3; we report honestly when the correction does not help |
| Long computing time | 20 draws by default with a precision check, runs in parallel, staged runs |
| Someone publishes the same idea first | Re-check before writing; our distinct parts are the many-tries answer, the cross-AI correction, and when it fails |
| Other AIs' results are not always available | The paper states when the correction applies (benchmarks with many agents' logs) and when it does not (brand-new tasks) |

## 8. Definition of done

- [ ] RQ1–RQ4 answered from `one_try_results.parquet`; hypotheses frozen before the full run.
- [ ] Results reported separately for true experiments and mid-run choices, and for each dataset.
- [ ] Every number, figure and table in Paper 2 is produced by the pipeline.
- [ ] Tests ≥ 85% coverage; CI passes on 4 platforms; two runs give identical results.
- [ ] Paper 2 draft compiles; release `v2.0.0` tagged.
