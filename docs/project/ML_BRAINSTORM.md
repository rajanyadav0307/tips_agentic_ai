# Machine Learning Brainstorm (2026-10-20)

**Purpose:** starting again from the raw datasets, find machine-learning projects within data-science scope that are new. Every idea below was checked against published work on 2026-10-20. This was a focused search, not a systematic review.

## What the raw data offers

| Signal in the data | Where | Why it matters for ML |
| --- | --- | --- |
| The same bug tried many times by the same AI | Older AI: 13 tries per bug (3,591 bugs); Nebius: 10 per bug (6,306 bugs); NVIDIA: 2–3 per bug, after its filtering | Repeated tries are random draws from the AI, which gives natural experiments and exact offline evaluation |
| Every step: action, and the computer's reply | All three | States, decisions, branch points |
| Identical moves, then a split ("natural forks") | Older AI: 24,319 fork points (5,658 with mixed outcomes); Nebius: 12,311 (2,705); NVIDIA: about 152,000 fork pairs (filtered) | Free "what if I had done the other thing?" comparisons |
| Luck-dependent bugs (sometimes fixed, sometimes not) | 22% (older AI) and 28% (Nebius) of bugs | Retries pay off only on these |
| AI's private "thinking" text | NVIDIA (MiniMax, DeepSeek); thinking is tied to the model, never switched within one model | Text signals; thinking on/off cannot be compared fairly |
| Correctness of the AI's self-written tests | Nebius (`gen_tests_correct`, `pred_passes_gen_tests`) | Labels for "can the AI trust its own tests?" |
| Same bugs tried by several AIs | NVIDIA: 30,194 bugs by ≥2 setups, 11,339 by ≥4 | Cross-AI comparisons |
| Bug reports, reference fixes, language, task type (bug fix, feature, enhancement) | NVIDIA; reference fixes also for the older AI | Features describing each bug |
| Evaluation timestamps | Older AI | Reveal a hidden mix of model versions |

## All ideas and their status

| # | Idea (plain words) | ML used | Status after the search |
| --- | --- | --- | --- |
| 1 | **Learn which next action helps, from natural forks**, without re-running the AI | Pairwise / advantage models (gradient boosting; optional text embeddings on GPU) | **Open in the literature, but rejected after the raw-data check below.** The closest work re-runs the AI from saved points: [Executable Prefix Values](https://arxiv.org/abs/2610.17633) (150 tasks); [He et al.](https://arxiv.org/abs/2608.22960) (replay, step effects underpowered). [CodeTracer](https://arxiv.org/abs/2604.11641) uses LLM judges, not learned values. |
| 2 | **Learn the best first move for each bug** (step 0 is an exact coin flip) | Offline contextual bandit / causal ML | **Open.** No per-bug first-move or strategy learning found. |
| 3 | **Smart retry budget**: spend a fixed number of tries on the bugs where another try is most likely to pay off | Calibrated success models, budget allocation | **Partly open.** [CodeRescue](https://arxiv.org/abs/2607.19338) routes after one failure (retry cheaply vs escalate); [Fail-Fast, Restart-Smart](https://arxiv.org/abs/2608.03222) restarts one run; [test-time scaling for agentic coding](https://arxiv.org/abs/2604.16529) uses fixed budgets. Allocation across bugs with exact offline evaluation was not found. |
| 4 | Per-bug effects of named tips, plus a recommender | Causal ML (R-learner, DR-learner) | **Open** (searched earlier the same day). |
| 5 | When can the AI trust its own tests? | Classifier on run features | **Partly open:** [SWT-Bench](https://arxiv.org/abs/2406.12952) and SpecRover ([leaderboard analysis](https://arxiv.org/abs/2506.17208)) study self-written tests. Only Nebius has the labels. |
| 6 | Does the AI's thinking match its actions? | Text models on thinking traces | **Partly open:** general-agent studies exist ([2606.00476](https://arxiv.org/abs/2606.00476), [2604.08401](https://arxiv.org/abs/2604.08401)); measurement is fragile. |
| 7 | Detect hidden model versions without labels | Clustering / mixture models | **Partly open:** labelled fingerprinting exists ([LIDAR](https://arxiv.org/abs/2609.28559), [procedural fingerprints](https://arxiv.org/abs/2606.16988)). |
| 8 | Send each bug to the best AI or setup | Routing classifiers | **Taken:** SWE-Router ([2607.00053](https://arxiv.org/abs/2607.00053)), [Scrouting](https://arxiv.org/abs/2608.04804), CodeRescue |
| 9 | Small, reliable benchmark subsets | Item response theory | **Taken:** [PTA-IRT](https://arxiv.org/abs/2609.01603), [Mid-Range Filter](https://arxiv.org/abs/2603.23749), [2609.17394](https://arxiv.org/abs/2609.17394) |
| 10 | Predict a run's cost before it starts | Regression | **Taken:** [token consumption](https://arxiv.org/abs/2604.22750), [TokenCast](https://arxiv.org/abs/2609.35760) |
| 11 | Predict failure early in a run | Sequence classifiers | **Taken:** [Fail-Fast](https://arxiv.org/abs/2608.03222), [2610.05572](https://arxiv.org/abs/2610.05572) |
| 12 | Which behaviors go with success; failure taxonomies | Correlational models | **Taken:** XEPV ([2606.15579](https://arxiv.org/abs/2606.15579)), [TraceProbe](https://arxiv.org/abs/2607.06184), [2604.02547](https://arxiv.org/abs/2604.02547), [ACT\*ONOMY](https://arxiv.org/abs/2605.13625) |
| 13 | Does thinking mode help? | Causal comparison | **Not possible here:** thinking is fixed per model, so its effect cannot be separated from the model's |
| 14 | Which AI produced a recording? | Classifier | **Taken:** LIDAR, [LLMmap](https://www.usenix.org/system/files/usenixsecurity25-pasquini.pdf) |

## Raw-data check of idea 1 (later the same day): rejected

The check is in `fork_check.py` (session scratchpad); it uses the older AI (issue × day) and Nebius, which both keep every try.

**1. After the first move, tries are almost never in exactly the same situation.** A *strict* fork needs the AI's whole visible context (its own notes, its commands, the computer's replies) to be identical before the split. A *relaxed* fork only needs identical commands.

| Fork points | Older AI: relaxed | Older AI: strict | Nebius: relaxed | Nebius: strict |
| --- | --- | --- | --- | --- |
| Step 0 | 7,204 | 7,204 | 6,111 | 6,111 |
| Steps 1–2 | 9,726 | 630 | 2,444 | 1,196 |
| Steps 3–5 | 8,415 | 162 | 2,539 | 205 |
| Steps 6–14 | 1,357 | 2 | 1,202 | 6 |

After step 0, almost all forks differ in the AI's own notes, so they are not fair comparisons.

**2. The choice at a fork barely changes the result.** We compare the variation in success between branches with the variation expected from luck alone (1.0 = no difference):
- older AI: 1.044 (95% CI 1.032–1.057), about 2% of outcome variation;
- Nebius: 1.000 (95% CI 0.993–1.006), about 0%.

A model would have almost nothing to learn.

**3. The bug itself decides most outcomes.** Share of outcome variation explained by which bug it is (`eda_stats.json`, e7): 51% for the older AI, 71–87% for the newer AIs. The rest is luck. Individual choices matter little, which also makes a per-bug "which bugs a tip helps" model (idea 4) unlikely to find much.

**Where the signal is: the bug.** A ceiling check for "how should a fixed number of tries be spread over bugs?", assuming perfect knowledge of each bug's chance:

| Dataset | Tries | Same number for every bug | Perfect knowledge |
| --- | --- | --- | --- |
| Older AI | 1 per bug | 12.5% fixed | 21.1% fixed |
| Nebius | 1 per bug | 47.6% fixed | 54.8% fixed |

72% (older AI) and 40% (Nebius) of bugs were never fixed in any try. A planner that skips hopeless bugs could reach the same fixes with far fewer tries; real models reach only part of this ceiling. Predicting bug difficulty has been studied ([2608.18280](https://arxiv.org/abs/2608.18280)), so the novelty would lie in the decision and its exact test on recorded tries, which needs a proper check.

## Best idea after the raw-data checks: can machine learning replace repeated runs?

**Why this question:** the bug decides most outcomes (51–87% of the variation), but the bug report hardly reveals how hard the bug is: a pilot predicting "never fixed" from the report alone reached AUC 0.59–0.64 (`hopeless_pilot.py`).

Most agent datasets and studies have **one try per task**, so they cannot tell "this behaviour helps" apart from "this task was easy". Our data has many tries per bug, so it can serve as an **answer key** for testing the corrections such studies use. This is a within-study comparison: the LaLonde (1986) and Smith & Todd (2005) design from economics, here applied to AI agents.

**Pilot** (`single_run_pilot.py`):
- **Data:** NVIDIA, four AI setups, effect of "test first", in points.
- **Answer key:** the within-bug comparison using all tries.
- **Single-try setting:** we kept one random try per bug and used double machine learning with gradient boosting.
- **Averaging:** each estimate is the mean of 5 random draws.

| AI setup | Answer key | Simple | ML + report text | + size of the human fix | + other AIs' results on the same bug |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6-27B / SWE-agent / scale-swe | −0.4 | +25.7 | +7.7 | +5.2 | +0.6 |
| Qwen3.5-122B / SWE-agent | +1.5 | +21.9 | +13.6 | +8.1 | +2.6 |
| Qwen3.5-122B / OpenHands | −1.0 | +14.1 | +4.2 | −0.4 | −2.2 |
| MiniMax-M2.5 / SWE-agent | +0.3 | +10.2 | +6.2 | +2.3 | +0.3 |
| **Average distance from the answer key** | — | **17.9** | **7.8** | **3.7** | **0.8** |

**Reading:**
- One-try analyses are badly wrong.
- ML on the bug report removes only part of the error.
- Adjusting for **other AIs' results on the same bug** (cross-agent difficulty) recovers the answer almost exactly.

**Novelty (searched 2026-10-20):** no study tests single-run analyses of agents against a repeated-run answer key, and none uses other agents' success on a task to correct behaviour analyses. Closest:
- within-task designs: [Beyond Resolution Rates](https://arxiv.org/abs/2604.02547), [Same Task, Different Fate](https://doi.org/10.1145/3844178.3844642);
- task-difficulty models: [Agent Psychometrics](https://arxiv.org/abs/2604.00594), [2608.18280](https://arxiv.org/abs/2608.18280);
- single-run noise: [Identical Runs, Different Results](https://arxiv.org/abs/2609.33812).

**To confirm before writing:**
- more behaviours, including step-0 choices, which are true experiments;
- the older-AI and Nebius data, which keep every try (10–13 per bug);
- how many tries per bug are enough;
- whether the corrected confidence intervals cover the answer key;
- NVIDIA's filtering, which could inflate the "other AIs" signal.

## Top picks (written before the raw-data check)

### 1. Learn which next action helps, from natural forks (rejected; see above)

- **Idea:** two tries of the same bug make exactly the same moves, then split: one runs the tests, the other opens a file. Everything before the split was the same, so the split is decided by the AI's sampling. Comparing how the branches end tells us which choice was better *at that moment*. A model learns from thousands of such moments and becomes an advisor: "in situations like this, do X".
- **Data:** about 36,600 fork points in the unfiltered datasets (8,363 with mixed outcomes), plus about 152,000 NVIDIA fork pairs as a secondary source (filtering caveat).
- **ML:**
  - situation features (step number, recent actions, error seen, files touched) and an option for text embeddings of the last reply (GPU);
  - gradient-boosted pairwise models;
  - 5-fold cross-fitting by bug.
- **Evaluation on unseen bugs:**
  - does the advisor pick the winning branch more often than chance (AUC with CIs)?
  - the gain from following it, measured exactly at held-out forks;
  - baselines: "always the most common action", simple rules, and a model trained the usual pooled way.
- **Novelty:** published methods get step values by re-running the AI from saved points (150 tasks in the newest paper). Our searches found no work that learns them from forks already in public logs. Off-policy evaluation for agents is considered impractical with standard re-weighting ([2606.05558](https://arxiv.org/abs/2606.05558)); natural forks avoid that problem for these decisions.
- **Risks and go/no-go checks:**
  - Tries can share identical actions while the AI's private thoughts differ, so later forks are close to, but not exactly, the same situation. Checks: compare step-0 forks (exactly identical input) with later forks; a balance test (do earlier thoughts predict the branch?).
  - Mixed-outcome forks are modest in number, so the first check is whether a simple model beats chance on held-out bugs.

### 2. Learn the best first move for each bug (cleanest)

- **Idea:** step 0 is an exact coin flip, because every try sees identical input. A model learns, from the bug report, which first move gives the best chance.
- **Data:** 9,226 step-0 fork points (3,115 older AI, 6,111 Nebius).
- **ML:** offline contextual bandit (causal ML with known, design-based choice probabilities), evaluated exactly on held-out bugs.
- **Fit:** the cleanest special case of idea 1, using the same machinery.

### 3. Smart retry budget (most practical)

- **Idea:** with a fixed number of tries, spend them on bugs where another try is most likely to pay off, using the bug report and what went wrong in earlier failed tries.
- **Data:** about 10,000 bugs with 10–13 tries each (older AI, Nebius).
- **ML:** calibrated success models (gradient boosting, beta-binomial), budget allocation; transfer from one AI to another.
- **Evaluation:** exact, by replaying recorded tries in random order (tries are independent draws).
- **Novelty:** partly open. CodeRescue and Fail-Fast make related "spend more after failure?" decisions, but not allocation across bugs evaluated exactly offline.

## A unifying theme

**"Every repeated try is a randomized experiment."** Public agent logs let machine learning learn three kinds of decisions and evaluate them exactly, without re-running any AI:

1. how to start (idea 2);
2. what to do next (idea 1);
3. where to spend retries (idea 3).

A single paper could lead with idea 1, include idea 2 as its exact special case, and leave idea 3 to a follow-up.
