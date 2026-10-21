# Step 1: Problem Definition (frozen)

*Paper 2, the one-try study. Written 2026-10-20 and tagged `one-try-plan-v1` before any modelling run. Changes after the tag are allowed only as dated amendments at the end of this file, with a reason.*

## In plain words

Most studies of AI coding tools see each bug only **once**. They compare what the AI did with whether it fixed the bug, and conclude things like "testing first helps". This can be wrong, because the AI may do that thing more often on **easy** bugs.

Our data has **many tries of the same bug**, so we can work out the right answer. We then pretend we have one try per bug, apply machine-learning corrections, and measure which correction gets back to the right answer.

## 1. Problem statement

- **Decision supported:** when a researcher has one recorded try per bug, which analysis can they trust to measure whether an AI behaviour helps it fix bugs?
- **Target quantity:** for a behaviour with two choices *a* and *b* (for example "test first" vs "fix first") and one AI setup, the difference in fix rate between *a* and *b* among tries of the same bug (the **many-tries answer**).
- **Success of the study:** we can say, with stated uncertainty:
  - how wrong one-try analyses are;
  - which corrections remove the error;
  - when they fail;
  - how many tries per bug are enough.

## 2. Definitions

| Term | Definition |
| --- | --- |
| Try | One recorded attempt of one AI setup on one bug |
| Bug | One task (issue), identified by its instance ID; the same ID in two datasets is the same bug |
| AI setup | Older AI: one model (`swe-agent-llama-70b` is studied; the 8B and 405B models serve only as other AIs). Nebius: Qwen3-Coder with OpenHands. NVIDIA: one subset (framework × model × task source) |
| Outcome | Bug fixed (`resolved` = 1) or not (0); tries with unknown outcome are removed |
| Behaviour | A two-choice property of a try, from the validated labeler of Paper 1 (κ 0.86–0.90), re-run on this study's data |
| Comparison | One behaviour × one AI setup with at least **100 bugs** where both choices occur (older AI: bug × evaluation-day groups, counted by bug) |
| True experiment | A comparison whose choice is the first action, made on identical input (step 0) |
| Mid-run comparison | Any other comparison; its many-tries answer is the best available, not an experiment |

**Candidate behaviours.** The final list of qualifying comparisons is fixed by the EDA inventory (Step 4) before modelling.

| Behaviour (choice *a* vs *b*) | Kind | Candidate AI setups |
| --- | --- | --- |
| Reproduce first (`create`) vs search first | True experiment | Older AI |
| First move `find_file` / `ls` / `open` / `grep` vs `search_dir` | True experiment | Older AI |
| First move `think` vs `view` | True experiment | Nebius |
| The two most common first tool actions | True experiment | Each NVIDIA setup |
| Test first vs fix first | Mid-run | Each NVIDIA setup |
| Re-ran own test script after the first code edit ("check your fix") | Mid-run | Older AI; each NVIDIA setup |
| Ran the project's test suite after the first code edit | Mid-run | Older AI; each NVIDIA setup |
| Ran any code after the first code edit | Mid-run | Each NVIDIA setup |
| Wrote extra tests after the first code edit | Mid-run | Each NVIDIA setup |
| Cleaned up created files | Mid-run | Each NVIDIA setup |
| Ran the project's test suite before the first code edit | Mid-run | Each NVIDIA setup |

## 3. The many-tries answer (answer key)

- **Estimator:** the weighted average of within-group differences in fix rate, weight n_a·n_b/(n_a+n_b) per group (Paper 1's fair estimator).
  - Groups are the bug; for the older AI, bug × evaluation day (hidden model versions).
  - Only groups with both choices count.
- **Uncertainty:** 95% CI from 2,000 bootstrap resamples of bugs. The CI half-width is reported as the answer key's **noise floor**.
- **Bugs used:** the bugs that carry the answer key, called the **key bugs**.

## 4. One-try copies

- **Primary:** from each key bug, draw **one** try at random from its tries with choice *a* or *b*.
  - Repeat with **20** independent draws (seeds 0–19).
  - Precision check: averages from 40 draws must differ by less than 0.2 points; otherwise use 40.
- **Secondary:** the same, drawing from all bugs of the AI setup.
- **RQ4:** draw *k* = 2, 3, 5, 10 tries per bug (bugs with at least *k* tries) and apply the many-tries estimator to the subsample. Older AI and Nebius only.

## 5. Correction ladder

Each method gets one one-try copy: one row per bug, with outcome *y*, choice *t* (1 = *a*) and bug features *X*. Features are built in Step 5. They use only information that exists without this AI setup's own outcomes.

| Rung | Method | Bug features *X* |
| --- | --- | --- |
| 0 | Simple difference: mean(*y* \| *t*=1) − mean(*y* \| *t*=0) | — |
| 1 | Linear regression *y* ~ *t* + *X*, coefficient of *t* (HC1 errors) | Report facts: text length (log), code block, error trace, names a file, repository (30 most common, else "other"), language, task type |
| 2 | Double machine learning, partialling out | Rung 1 + report text: TF-IDF (1–2-grams, min_df 3, max 50,000 terms, sublinear) → truncated SVD to 100 dimensions, fitted on the training folds only |
| 3 (exploratory) | As rung 2 | + sentence embeddings of the report (model pinned in Step 5; GPU if available). Not part of any hypothesis test |
| 4 | As rung 2 | Rung 2 + human-fix size: log(1 + files changed), log(1 + lines changed), missing flag |
| 5 | As rung 2 | Rung 4 + other AIs: mean fix rate of the *other* AI setups on the same bug (each setup weighted equally), number of other setups, missing flag |
| 6 | As rung 2 | Rung 4 + IRT difficulty: a Rasch model (setup ability + bug difficulty, L2 penalty 1.0) fitted on the other setups' tries only; difficulty score and missing flag |

**Double machine learning, fixed settings:**
- 5 folds split by bug (seed = draw number).
- **Learners:** two scikit-learn `HistGradientBoostingRegressor` models (max_iter 200, learning rate 0.05, other defaults, random_state 0), one for E[*y*|*X*] and one for E[*t*|*X*].
- **Estimate:** θ = Σ r_y r_t / Σ r_t², with r the out-of-fold residuals.
- **Standard error:** from the influence function.

**Checks (reported, not used for decisions):**
- the same rungs with `RandomForestRegressor` (300 trees, min_samples_leaf 20);
- a doubly robust AIPW version of rungs 2, 4 and 5, with a gradient-boosting propensity model clipped to [0.02, 0.98].

## 6. Scores

For comparison *c*, rung *r* and draw *d*, the estimate is θ̂_crd and the answer key is A_c.

| Score | Definition |
| --- | --- |
| Error | e_cr = mean over d of (θ̂_crd − A_c) |
| Absolute error | \|e_cr\| (main score) |
| Spread | standard deviation of θ̂_crd over draws |
| Sign agreement | share of draws with sign(θ̂) = sign(A_c), counted only when A_c's CI excludes 0 |
| Conclusion agreement | share of draws whose conclusion matches the answer key's (95% CI above 0, below 0, or containing 0) |
| Coverage | share of draws whose 95% CI contains A_c |

**Averages:**
- the mean absolute error per rung, over comparisons;
- also by dataset, and for true experiments vs mid-run comparisons.

**CIs of averages:** 2,000 bootstrap resamples of comparisons, clustered by AI setup.

## 7. Research questions and hypotheses

| RQ | Hypothesis | Test and decision rule |
| --- | --- | --- |
| RQ1: How wrong are one-try analyses? | **H1:** the mean absolute error of rung 0 is above 2 points | One-sided: supported if the lower 95% bound is above 2 |
| RQ2: Which correction fixes them? | **H2:** corrections using only the bug report leave a mean absolute error above 2 points | Supported if the lower 95% bounds of **both** rung 1 and rung 2 are above 2 |
| | **H3:** adding other AIs' results makes the error negligible | Equivalence: supported if the upper 90% bound of rung 5's mean absolute error is below 2 |
| RQ3: When does the correction fail? | **H4:** the gain from other-AI information shrinks as that information weakens | Two pre-specified slope tests, both one-sided and both required (see below) |
| RQ4: How many tries per bug are enough? | Descriptive | Mean absolute error of the *k*-try within-bug analysis, *k* = 2, 3, 5, 10, vs the full answer key |

**H4 tests:**
- **(a)** Across comparisons, the gain \|e_c2\| − \|e_c5\| rises with the **agreement** between the studied setup and the other setups. Agreement is the correlation of per-bug fix rates on shared bugs. Test: OLS slope > 0, clustered by setup.
- **(b)** In NVIDIA comparisons, re-run rung 5 using only *k* = 1, 2, 4 or all other setups (random subsets, fixed seeds). The error must fall as *k* rises (Spearman ρ < 0, permutation p).

**Multiple testing:** H1–H4 are Holm-corrected at α = 0.05.

**Margin:** ±2 points, as in Paper 1. It must exceed twice the median noise floor of the answer keys; if not, this is reported, and the margin stays.

## 8. Scope and assumptions

- **Data:** only the three pinned public datasets and the two gold-patch sources. No new agent runs.
- **Older AI:** its hidden model versions are handled by the bug × day groups in the answer key. Its one-try copies ignore the day, as a one-try researcher would.
- **NVIDIA:** NVIDIA removed some failed tries. The study compares methods inside the same data, so the comparison stays internal; the effect on the answer keys is a stated limitation.
- **Mid-run answers:** these are not experiments. All headline claims are reported separately for true experiments.
- **Other AIs:** other-AI information is available only for bugs that other setups tried. The study reports how coverage changes the results (H4) and when the correction cannot be used at all.

## 9. What would change the story

- If rung 2 already reaches the 2-point margin, the message becomes "ML on the bug report is enough". We report that.
- If rung 5 fails on true experiments but works on mid-run comparisons (or the reverse), we report the split.
- If H3 fails, the paper reports how far each correction gets, and why.

## 10. Novelty re-check (2026-10-20)

Searches on 2026-10-20 (details in `docs/project/ML_BRAINSTORM.md`) found no study that does either of the following:
- tests one-try analyses of AI agents against a many-tries answer;
- uses other AIs' results on the same task to correct behaviour analyses.

Closest work:
- within-task designs ([Same Task, Different Fate](https://doi.org/10.1145/3844178.3844642), [Beyond Resolution Rates](https://arxiv.org/abs/2604.02547));
- task difficulty models ([Agent Psychometrics](https://arxiv.org/abs/2604.00594), [What Makes Tasks Difficult](https://arxiv.org/abs/2608.18280));
- the classic within-study comparison design in economics (LaLonde 1986; Smith & Todd 2005).

The novelty check is repeated before writing (Step 8).

## Amendments

*None yet.*
