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

**A1–A6 (2026-10-21, before any modelling run). Reason for all: computing time.**

Timing on the largest comparison (5,514 bugs), per draw: about 30 s for the text model, 13 s per DML rung, and about 50 s for *each* random-forest fit (10 per rung). The frozen design would take well over a day on the study laptop. None of these changes affect a hypothesis test: H1–H4 use rungs 0, 1, 2 and 5, which keep all 20 draws (40 for the precision check).

- **A1. Checks get 5 draws.**
  - The random-forest check runs on rung 5 only (the rung H3 judges).
  - The AIPW check runs on rungs 2, 4 and 5.
  - The exploratory rung 3 is included.

  All of them use draws 0–4 instead of 0–19. They are reported, never used for decisions.
- **A2. Secondary "all bugs" copies get 3 draws:** rungs 0, 2 and 5, draws 0–2.
- **A3. Reproducibility:** instead of a second full run, three comparisons chosen at random (seed 0) are re-run completely, and every result row must match exactly. Every unit of work has a fixed seed, so this checks determinism without doubling the computing time.
- **A4. Precision check:** done on rungs 0 and 5 (draws 20–39), the rungs that carry H1 and H3.
- **A5. Text-dimension robustness (50 / 200 dimensions) is dropped.** The language-model embeddings (rung 3) already give an alternative text representation.
- **A6. Implementation (numbers unchanged):**
  - report text is split into words once per comparison and reused across folds (tested to give identical numbers);
  - comparisons run in parallel (5 workers × 2 threads);
  - random-forest `n_jobs` = 2.

**A7 (2026-10-21, after Step 6, before any Step 7 run). The "all bugs" secondary analysis is strengthened.**
- **What prompted it:** in the Step 6 go/no-go, the simple comparison's error on the pilot's comparisons was 2.4 points, against about 18 in the pilot. The reason is that the primary copies draw only from key bugs (bugs where both choices occur). Most of the easy-bug trap comes from bugs where the AI *always* or *never* makes a choice, and a real one-try study cannot leave those out: it cannot know which bugs are key bugs without many tries.
- **Change:**
  - the secondary analysis (copies drawn from all bugs) now uses draws 0–4 (5 draws) and rungs 0, 1, 2 and 5 (replacing A2);
  - it is reported next to the primary analysis, with the same scores.
- **Unchanged:**
  - H1–H4 and their decision rules (primary, key bugs);
  - the answer keys.
- **Caveat reported with it:** the answer key is defined on key bugs only, so for the all-bugs copies part of the "error" can come from the two groups of bugs having different true effects.
- **Margin check (Step 6):** the median noise floor of the answer keys is 1.7 points, so the ±2-point margin is *not* at least twice the noise floor. As frozen in §7, this is reported and the margin stays.


**A8 (2026-10-21, after an independent expert review, before any Step 7 result was scored). Secondary statistics.**
- **What prompted it:** a review of the design by an outside expert (machine learning, MLOps, data science). The review judged the design sound. It found four weak spots in how the methods are *scored*:
  - the answer key weights bugs differently from the ML methods;
  - the answer key and the one-try copies share tries, so they share noise;
  - coverage and "same conclusion" reward wide intervals;
  - there are only 9 AI setups (clusters).
- **Unchanged:**
  - H1–H4 and their decision rules;
  - the ±2-point margin;
  - the main score (average absolute distance from the frozen answer key);
  - the comparisons, the answer keys and the draws.
- **Wording:** we call the main score the *distance from the many-tries answer*. The answer key is itself an estimate with its own noise.
- **Everything below is secondary.** It is reported to help read the results and never used to accept or reject H1–H4.
  1. **Aligned answer keys.** Two extra keys are computed from the same tries, with 2,000 bootstrap resamples of bugs:
     - *dml*: group weight n_a·n_b/(n_g·N), with N the bug's key tries. This is the frozen weight divided by N, which is what partialling-out targets with one try per bug.
     - *ate*: group weight n_g/N, so each bug counts once. This is what AIPW targets.

     ML rungs are also scored against *dml*, AIPW checks against *ate*, and rungs 0–1 against the frozen key.
  2. **Error floor and noise-corrected bias.**
     - Every rung's estimate is a slope Σ r_y·r_t / Σ r_t² of per-bug leftovers; AIPW is a mean of per-bug scores. These are saved for draws 0–19.
     - A **joint bootstrap** resamples key bugs 1,000 times (fixed seed). It redraws the answer key and all 20 estimates *together*, without refitting.
     - The **error floor** is the standard deviation of (mean estimate − key) over the resamples.
     - Reported per rung:
       - the noise-corrected bias, √max(0, mean(error² − floor²)) over comparisons;
       - the share of comparisons with |error| > 1.96 floors.
  3. **Paired contrasts.** |error A| − |error B| on the same comparisons, for these pairs:
     - 0 vs 2;
     - 2 vs 5;
     - 0 vs 5;
     - 2 vs 4;
     - 4 vs 5;
     - realistic 5 vs 5;
     - 5 vs oracle.
  4. **Few-cluster intervals,** for every average and contrast:
     - per-setup means with a t interval on (setups − 1) degrees of freedom;
     - a Webb six-point wild cluster bootstrap (9,999 draws);
     - the range when one setup is left out.

     NVIDIA setups share bugs and models, so they are not fully independent; this is stated with the results.
  5. **Interval quality,** per rung:
     - mean width;
     - interval score (width + 40 × miss distance);
     - detection: for keys whose CI excludes 0, the share of draws reaching that direction;
     - false alarms: for keys whose CI contains 0, the share of draws claiming a direction.
  6. **Model diagnostics,** per ML rung:
     - R² of the fix model;
     - AUC of the choice model;
     - Var(r_t)/Var(t), the share of the choice left over.
  7. **Two new rungs** (draws 0–19, same settings as rung 5):
     - **Realistic other AIs** (`rung5_real`): the other AIs' rate is computed from **one** random try per other setup and bug (fixed seeds), as a real study would have it.
     - **Oracle** (`rung7`): rung 4's features plus the studied AI's own fix rate on the bug from its *other* tries, leaving out the drawn try (with count and missing flag). This is a best case that no one-try study has.
       - It is run **only for the older AI and Nebius** (10–13 tries per bug).
       - With NVIDIA's 2–3 tries, the "other" try of a key bug mostly made the *opposite* choice. Its outcome would then carry the effect itself, so the oracle is not defined there.
  8. **Sensitivity:** every average is also reported on comparisons whose answer key has a noise floor ≤ 1.5 points.
  9. **All-bugs copies (A7).** Reported in addition:
     - the share of the ML weight (Σ r_t²) that falls on non-key bugs;
     - AIPW rung 5 on all bugs;
     - trimmed versions of rungs 0, 1 and AIPW rung 5 that keep only bugs whose out-of-fold chance of choice *a* (gradient-boosting classifier, rung-5 features) is between 0.1 and 0.9.
- **Disclosure.** scikit-learn's gradient boosting turns on early stopping automatically above 10,000 rows. This happens only in some NVIDIA all-bugs copies, and the count is reported. The setting is deterministic (`random_state` 0), so it is left as frozen.
- **H3 power, stated in advance:**
  - H3's upper bound comes from 9 setup clusters. With an average rung-5 distance of about 1 point (the pilot), the one-sided 95% bound stays below 2 only if the per-setup spread of that distance is below about 1.6 points (1 + 1.86·SD/3 < 2).
  - A failed H3 alone is therefore weak evidence that the correction fails. It is read together with the noise-corrected bias and the paired contrasts.
  - The bootstrap over 9 clusters can also be too narrow; the few-cluster intervals are the check.

**A9 (2026-10-21, same review, before any Step 7 result was scored). Computing and reproducibility.**
- **Unit of work.** One task is one comparison × one draw, with **one thread per task** and 8 parallel tasks on the study laptop. Before this, 5 comparisons each ran with the `threads: 4` setting meant for data loading, which over-booked the 10 cores. This change does not alter any number.
- **Report text, fitted once per dataset.**
  - The TF-IDF → 100-dimension SVD model (frozen settings, seed 0) is fitted **once per dataset on all of its bug reports** and cached (`data/onetry/features/text_svd.parquet`). Before, it was fitted inside each training fold.
  - It uses report text only, never an outcome or a choice, so held-out bugs cannot leak an answer.
  - This removes most of the computing time but changes ML estimates slightly.
  - **Pass rule against the sealed first run** (below):
    - rungs 0 and 1 must be identical draw by draw;
    - for rungs 2, 4, 5 and 6, the mean over draws 0–19 per comparison must agree within **0.3 points**.

    If the rule fails, the per-fold text model is restored and the whole run is repeated with it.
- **Random-forest check:** draws 0–1 instead of 0–4 (A1), with `n_jobs` 1.
- **Sealed first run.**
  - The 15 comparisons finished under the A1–A7 code were moved, unscored, to `outputs/onetry/sealed_v1/`. Their SHA-256 hashes are in `docs/one_try/sealed_units_v1.sha256`.
  - Only their *estimates* are read, for the pass rule above; they are never scored against an answer key.
  - These comparisons run first, and on the computer that made them.
- **Several computers.** The run may be split across the study team's three laptops:
  - an Apple M4 (home: the sealed run, the A3 re-runs);
  - an AMD Ryzen 9 5900HS;
  - an AMD Ryzen 3 3350U.

  How the split works:
  - Whole comparisons are assigned by a fixed plan (`docs/one_try/share_plan.json`, committed before the run).
  - Partners get one zip with the committed code, the study data and the plan, and use the same locked libraries (`uv.lock`).
  - Every result folder records the computer, system and library versions.
  - Results come back as zips with SHA-256 hashes, which are checked on merge.
  - **Cross-computer rule:** for 2 random comparisons per partner computer (seed 0), one draw from 0–4 (every rung and check) is re-run on the home computer:
    - rungs 0 and 1 must be identical;
    - every other estimate must agree within **0.1 point**.

    If the rule fails, that computer's comparisons are re-run on the home computer.
  - The A3 exact-repeat check draws its comparisons from those made on the home computer.
- **No GPU backend.** The models are many small gradient-boosting fits that scikit-learn runs on the CPU. GPU versions (cuML, XGBoost-CUDA) are different algorithms or not deterministic, and would change the numbers. The only GPU step, the rung-3 embeddings, was already done in Step 5.
