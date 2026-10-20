# Step 1: Problem, Hypotheses and Analysis Plan (frozen)

**Version:** `analysis-plan-v1`, frozen 2026-10-20 (the git tag marks the frozen version)
**Rule:** the final analyses (Phase D) must follow this document. Any change after freezing is listed in "Changes after freezing" at the end, with the date and reason.

## In plain words

- **The problem:** AI coding assistants are given tips such as "reproduce the bug first", and nobody has checked fairly whether these tips help. People usually compare successful tries with failed tries, which can give wrong answers.
- **What this document does:** it writes down **exactly** which questions we will answer, how, and what result would count as "yes" or "no", *before* the final analysis is run. This stops us from picking whichever analysis gives the nicest answer.
- **Honesty note:** we already ran quick exploratory analyses (in `analysis/exploratory/`), so this is not a "blind" plan. What it prevents is changing the analysis from here on. The paper will say this openly.

## 1. Problem statement

Prompt tips for coding agents, such as "reproduce the bug first" and "check your fix", are widely used, but whether they help is untested in a causal sense. Published trajectory analyses compare successful and failed runs, which mixes up the effect of a behaviour with the difficulty of the bug and with what happened earlier in the run.

**Goal:** estimate the effect of following common tips on whether the issue gets resolved, using fair comparisons between repeated tries of the same issue. Then measure how far the usual simple comparisons are from those fair estimates.

## 2. Data and units

| Dataset key | Source (pinned revision) | AI(s) | Unit of analysis |
| --- | --- | --- | --- |
| `older_ai` | nebius/SWE-agent-trajectories @ `68195a1` | `swe-agent-llama-70b` only | one try (trajectory) |
| `newer_ai_nebius` | nebius/SWE-rebench-openhands-trajectories @ `3545538` | Qwen3-Coder-480B | one try |
| `newer_ai_nvidia` | nvidia/Open-SWE-Traces @ `1db6741` (7 subsets) | MiniMax-M2.5, Qwen3.5-122B, Qwen3.6-27B, DeepSeek-V4-Flash | one try |
| `gold_extra`, `gold_dev` | nebius/SWE-bench-extra @ `11dcbfb`, princeton-nlp/SWE-bench (dev) @ `e48e2bd` | — | one issue (gold patch) |

**Outcome (all analyses unless stated):** `resolved` = 1 if the hidden tests pass, else 0.
- `older_ai`: field `target`.
- `newer_ai_nebius`: field `resolved`.
- `newer_ai_nvidia`: `resolved` = 1 or 0. Tries with −1 (unknown) are **excluded** and the exclusion is reported.

**Common exclusions:**
- Only tries whose issue text equals the most common text for that issue (identical input).
- `older_ai`: only the 70B model.

Every exclusion is counted in the cleaning report (Step 3).

## 3. Fair-comparison groups ("strata")

| Dataset | Stratum | Why |
| --- | --- | --- |
| `older_ai` | issue × evaluation day | The 70B label mixes model versions; the evaluation day is the version proxy |
| `newer_ai_nebius` | issue | Single model and configuration |
| `newer_ai_nvidia` | subset (scaffold × AI × task source) × issue | One teacher configuration per subset |

**Estimator:**
- Weighted average of within-stratum differences in resolution rate between the two groups, with weight n_a·n_b/(n_a+n_b) per stratum.
- Strata without both groups are dropped.
- **95% CI:** 2,000 bootstrap resamples of whole issues.
- **Permutation p-value:** 1,000 shuffles of group labels within strata.
- **Cross-check:** a linear probability model with stratum fixed effects and standard errors clustered by issue.

## 4. Primary hypotheses

These three are tested without multiple-testing correction, at α = 0.05, two-sided.

### P1: "Reproduce first" for the older AI (RQ2)

- **Comparison:** first action is `create <file>` (writing a reproduction script, group R) vs. first action is a search or navigation command (`search_dir`, `search_file`, `find_file`, `ls`, `grep`, `find`; group S).
- **Data:** `older_ai`, dated tries (with an evaluation timestamp), strata = issue × day.
- **H1:** the R − S difference in resolution ≠ 0 (expected direction from exploration: negative).
- **Decision:** supported if the 95% CI excludes 0.
- **Planned sensitivity analyses** (reported in Step 7, do not change the decision): hour, 6-hour, 60-min-gap and 10-min-gap batches; bounds and imputation for undated tries; placebo `reproduce.py` vs. `reproduce_bug.py`.

### P2: "Test first" for the newer AIs (RQ2)

- **Comparison:** the try creates a test or reproduction file before its first edit of existing project code ("test first") vs. edits existing code first ("fix first"). Labels come from the shared labeler (Step 5).
- **Data:** `newer_ai_nvidia`, all 7 subsets pooled, strata = subset × issue; known outcomes only.
- **H2:** the difference ≠ 0.
- **Equivalence claim:** "no meaningful difference" is claimed only if the 90% CI lies within ±2 points (two one-sided tests at α = 0.05).
- **Decision:** report which of the three applies:
  - "difference" (95% CI excludes 0);
  - "equivalent within ±2 points";
  - "inconclusive".
- **Secondary to P2:** exact forks, i.e. pairs of tries with identical action histories, "test file now" vs. "anything else now".

### P3: Size of the simple-analysis trap (RQ3)

- **Pairs examined:** every tip × AI pair from the list in §6. For each pair, compute the simple pooled difference (no strata) and the fair difference (§3).
- **Gap:** simple − fair.
- **H3:** the mean absolute gap across pairs is greater than 2 points.
- **Decision:** supported if the 95% bootstrap CI of the mean absolute gap (resampling issues within each pair) lies entirely above 2 points.
- **Also reported:**
  - the share of pairs where the simple and fair estimates disagree in sign or in significance;
  - Simpson's paradox across AIs (pooled simple vs. per-AI simple).

## 5. Design checks (RQ1)

These are not hypothesis tests; they are diagnostics with pass/fail rules.

| Check | Rule |
| --- | --- |
| Identical inputs | One system prompt per dataset and AI; modal issue text used |
| Positivity | Report the share of strata containing both groups, for each comparison. A comparison with fewer than 100 such strata is labelled "underpowered" and not interpreted. |
| Balance (older AI) | Within issue × day, the hour does not predict the first action (permutation p > 0.05) |
| Placebo (older AI) | `reproduce.py` vs `reproduce_bug.py`: 95% CI includes 0 |
| Undecided agent | A tip followed in more than 95% or fewer than 5% of tries is reported as "no natural experiment possible" (e.g. the Nebius newer AI) |

## 6. Secondary hypotheses

All secondary tests are corrected together with Benjamini–Hochberg at q = 0.05. Estimates and CIs are reported for all of them, whether significant or not.

| ID | RQ | Comparison | Data |
| --- | --- | --- | --- |
| S1 | RQ4 | Re-ran a repro or test script after the first code edit vs. not | older AI; tries with a code edit |
| S2 | RQ4 | Ran the project's test suite after the first code edit vs. not | older AI; newer AIs (NVIDIA) |
| S3 | RQ4 | Ran code of any kind after fixing vs. not | newer AIs |
| S4 | RQ4 | Wrote extra tests after fixing vs. not | newer AIs |
| S5 | RQ4 | Cleaned up created files vs. not | newer AIs |
| S6 | RQ4 | Ran the project's test suite before fixing vs. not | newer AIs |
| S7 | RQ4 | Cost of test first: number of steps; characters read and written | older AI (steps); newer AIs (steps, characters) |
| S8 | RQ5 | P1 by issue type (code block, traceback, names a file, longer than median): interaction terms | older AI |
| S9 | RQ5 | P2 by programming language: interaction terms | newer AIs |
| S10 | RQ6 | P1 effect on "submitted because the context ran out" and on any context exit | older AI |
| S11 | RQ6 | P1 effect on time to first open a gold file: discrete-time hazard model plus Kaplan–Meier curves; tries that never open one are censored | older AI |
| S12 | RQ2 | First move `think` vs `view` (exact coin flip) | Nebius newer AI |
| S13 | RQ3 | Simple vs. fair estimates for the older AI's other first moves vs. `search_dir` (`find_file`, `ls`, `open`, `grep`) | older AI |

Mid-run comparisons (S1–S6) are **not** coin flips. They are reported as "same-issue associations", not causal effects, and the paper says so.

## 7. What would change the story

| Result | Interpretation we commit to |
| --- | --- |
| P1 CI excludes 0 and is negative | Reproducing first harmed the older AI |
| P1 CI includes 0 | No evidence of harm. The headline becomes "no benefit" for all AIs. |
| P2 equivalent within ±2 points | No meaningful benefit for newer AIs, among finished tries |
| P2 positive and significant | Testing first helps newer AIs; this contradicts the exploration and is reported as such |
| H3 not supported | Simple analyses are not badly biased here; contribution 2 is downgraded to a robustness note |

## 8. Software and reproducibility

- All analyses run through `agent-tips model` and `agent-tips evaluate`, with seed 0.
- Bootstrap: 2,000 reps; permutation: 1,000 reps (`configs/default.yaml`).
- Results are written to `outputs/results/estimates.parquet`, one row per estimate.

## Changes after freezing

None yet.
