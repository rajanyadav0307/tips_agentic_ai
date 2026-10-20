# Do Tips for AI Coding Agents Actually Help? Proposal and Results

**Last updated:** 2026-10-19
**Status:** project chosen; robustness checks passed; checks on newer AIs done. **Next:** paper outline.
**Related files:** [RELATED_WORK.md](RELATED_WORK.md) (first idea search), [PROJECT_OPTIONS.md](PROJECT_OPTIONS.md) (11 project options compared), [DATASET_OPTIONS.md](DATASET_OPTIONS.md) (public datasets)

**Data** (about 400,000 tries by 6 AIs):

- **Older AI:** [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories). 74,792 tries by `swe-agent-llama-70b` (2024) on 3,591 bugs; local copy in `data/`.
- **Newer AI (Nebius):** [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories). 67,074 tries by Qwen3-Coder-480B on 6,306 bugs; local copy in `data/`.
- **Newer AIs (NVIDIA):** [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces). 254,933 tries by MiniMax-M2.5, Qwen3.5-122B, Qwen3.6-27B and DeepSeek-V4-Flash. Streamed from Hugging Face; the per-try summary is in `analysis/data/nvidia_runs.parquet`.

## In plain words

**The project in one sentence:** we test, with data, whether popular tips given to AI coding assistants actually help them fix more bugs. We also show that the usual way of analyzing AI recordings gives wrong answers.

**How it works, with an example.** Suppose the same AI tried Bug #1 ten times. Five tries followed the tip, and 1 of them fixed the bug. Five tries did not, and 2 of them fixed it. On this bug, the tip did slightly worse. We repeat this for thousands of bugs and add up the results. Comparing tries of the *same* bug is fair, because easy and hard bugs can't fool us.

**Our two main points:**

1. **"Reproduce the bug first" doesn't help the way people think.**
   - The older AI sometimes reproduced the bug as its very first step, before reading any code. Those tries fixed about 3 to 4 fewer bugs per 100. At the first step, the choice is made purely by chance, like a coin toss, so this is a fair test.
   - For four newer AIs, writing a test before fixing (after reading the code) made **no difference**.
2. **Simple analysis gives wrong answers.** Mixing all bugs together makes the tip look like it helps by up to 23 points. Fair same-bug comparisons say zero, or harm.

**Supporting findings:**

- **"Check your fix"** (re-run the test after fixing) may help a little: about 3 more fixes per 100 for the older AI. The simple comparison says 10.
- **Other tips made no measurable difference** for the newer AIs: running the project's tests, writing extra tests, and cleaning up afterwards.
- **"Test first" costs extra work:** about 10 more steps for the older AI and about 2 more for the newer AIs.
- **The same in every programming language** we checked (7 languages).
- **AIs differ hugely in how often they follow the tip** (10% to 98%), yet following it doesn't change their success.

**Main caveats:**

- The older AI's dataset mixes several versions of that AI under one name, so we only compare tries checked on the same day.
- NVIDIA removed tries that ran out of steps. For the newer AIs we therefore can't see whether testing first makes them run out of room.
- Only the first move, and pairs of tries with identical histories, are true coin flips. Later behaviours such as "check your fix" are less certain.

**Next steps:** paper outline, a "why" analysis, and automatic validation of the labels (no human effort; see the project plan, task C2).

## Findings at a glance

All numbers are differences in bugs fixed per 100 tries, with 95% confidence intervals in brackets.

| Question | Older AI (2024) | Newer AIs (NVIDIA, 2026 data) |
| --- | --- | --- |
| Does "reproduce / test first" help? Fair comparison | **Hurts: −3.5 [−4.8, −2.2]**. First step, so a coin flip; same bug and same day | **No difference: +0.3 [−0.3, +0.8]** (same bug); near coin flips −0.0 [−1.1, +1.0] |
| What the simple comparison says | −1.7 overall, and −0.1 vs `search_dir` (hides the harm) | +8 to +23 within most AIs |
| Does "check your fix" help? | Maybe a little: +3.0 [+2.2, +3.8] (simple +10.2) | Can't tell: done in 97–100% of tries |
| Other tips (project tests, extra tests, cleanup) | Project tests after fixing: +0.3 [−1.3, +1.7] | All within about ±1 point (simple: ±7 to 11) |
| Cost of "test first" | +10 steps | +1.8 steps; slightly more text read and written |
| Same across bug types or languages? | Harmful for every issue type checked | No clear benefit in any of 7 languages |

The newer Nebius AI (Qwen3-Coder) tests first in 98% of tries, so it can't answer the main question. Its chance choices of first move made no difference (−0.06 [−0.79, +0.64]), while the simple comparison says +3.55.

## Detailed results

### 1. Older AI: the main test and its robustness checks

**In plain words: the result held up.**

- **Same answer with tighter time groups.** We compared only tries checked within the same hour instead of the same day. The answer stayed the same: about 3 to 4 fewer bugs fixed per 100 tries when the AI starts by reproducing.
- **Not luck.** We randomly shuffled which tries counted as "reproduce first" 1,000 times. A gap this big never came up by chance.
- **No sign of mixed AI versions within a day.** The hour of the day does not predict how the AI starts, so comparing within a day looks safe.
- **A test that should show nothing shows nothing.** Two names for the same reproduction script (`reproduce.py` and `reproduce_bug.py`) give the same results, as they should.
- **Missing dates.** About 9,600 tries ended with no fix and no date. Adding them back in a realistic way still gives about 3 fewer fixes per 100. Only a deliberately unfair placement shrinks the gap to about 1 (grouping by day) or erases it (grouping by hour).
- **Why it seems to happen.** Tries that start by reproducing take about 10 more steps. They are also about 11 points more likely to fill up the AI's working memory (its "context window"), which forces it to stop and hand in whatever it has.
- **One correction to the story.** Tries that start by searching almost never write a reproduction script later (about 4 in 100). So the comparison is "start by reproducing" vs. "start by searching", not "reproduce early" vs. "reproduce late".

**Technical details:**

| Check | Result |
| --- | --- |
| Effect by batch definition (R minus S, 95% CI from resampling issues) | day −3.49 [−4.83, −2.18] (1,049 issues, 13,522 runs); 6 h −3.60 [−4.91, −2.28]; hour −3.74 [−5.47, −2.01] (808 issues, 5,342 runs); 60-min gap −3.98 [−5.51, −2.44]; 10-min gap −2.76 [−5.05, −0.36] (571 issues, 2,866 runs) |
| Within-group permutation test (1,000 shuffles) | p < 0.001 (day, hour); p = 0.017 (10-min gap) |
| Does the hour predict the first action within a day? | No. Variation is 1.00× the chance level, p = 0.45 (1,739 groups, 13,166 runs) |
| Placebo: `reproduce.py` vs `reproduce_bug.py` | day −0.78 [−2.29, +0.77]; hour −0.30 [−2.60, +1.91] |
| Undated runs added back (day / hour batches) | like other failures: −3.08 / −3.25; rough worst case: −1.36 / +0.64; best case: −5.39 / −9.66 |
| Label check | R: the first command creates a `reproduce*` file in about 99% of runs, and 83.1% run it within 5 commands. S: 3.9% later create a `repro*` script; 40.4% create some file later; 42.1% run a script of any kind; 7.3% run the project's test suite (R: 2.0%) |
| Mechanism (exploratory) | Within issue × day: R is +11.0 pts [+9.2, +12.7] more often force-submitted at the context limit and uses +10.0 [+8.8, +11.3] more commands. Within issue, all runs: no-patch rate −0.1 [−1.7, +1.4]; any context exit +5.8 [+4.2, +7.4] |

**Simple vs. fair, for every common first move** (each compared with `search_dir`; fair = same issue × same evaluation day):

| First move | Simple (all pooled) | Same issue | Fair (issue × day) |
| --- | ---: | --- | --- |
| `create` (reproduce) | −0.1 | −3.9 [−5.2, −2.6] | −3.4 [−5.1, −1.3] |
| `find_file` | +7.0 | −1.1 [−2.1, −0.0] | −0.5 [−2.0, +0.9] |
| `ls` | −1.0 | −0.9 [−1.7, −0.1] | −1.2 [−2.6, +0.1] |
| `open` | +9.5 | +2.7 [−1.2, +6.1] | +5.0 [−1.5, +11.0] |
| `grep` | −3.3 | −0.6 [−2.4, +1.3] | −1.2 [−3.7, +1.5] |

**By type of issue** (R vs. S, fair): overall −3.5 [−4.8, −2.1].

| Issue... | Yes | No |
| --- | --- | --- |
| has a code block | −2.6 [−4.2, −1.1] | −4.6 [−6.8, −2.6] |
| has a traceback | −4.5 [−7.3, −1.6] | −3.3 [−4.8, −1.9] |
| names a `.py` file | −3.8 [−5.9, −1.7] | −3.3 [−5.0, −1.8] |
| is longer than the median | −3.3 [−5.2, −1.5] | −3.6 [−5.6, −1.8] |

The effect is harmful in every group. It is smaller when the issue includes example code, possibly because a ready-made example makes reproduction cheap.

### 2. Newer AI from Nebius (Qwen3-Coder): why it can't answer the main question

**In plain words:**

- **It almost always tests first.** In 98 of every 100 tries, it writes a test script before changing the code, because its instructions say so and it obeys. With almost no tries going the other way, there is nothing fair to compare.
- **Its chance choices made no difference.** For example, its first move was sometimes "stop and think" and sometimes "look at the files", and results were the same. The simple comparison, however, says "think first" is 3.6 points better.
- **What this means:** this AI shows the limit of the method. A coin flip exists only when the AI is undecided. Evidence on newer AIs therefore comes from NVIDIA's data (section 3).

**Technical details:**

| Check | Result |
| --- | --- |
| Test script created before the first edit of existing code | 97.9% of runs (median step 23); 0.9% edit code first; 1.2% never edit |
| Test first vs. fix first | simple +12.09; within issue (not a coin flip; tiny group) −0.73 [−3.79, +1.99]; hit the 100-step cap −3.09 [−6.69, +0.39]; 216 issues, 2,505 runs |
| Exact fork, step 0: `think` vs `view` | −0.06 [−0.79, +0.64]; 2,067 issues, 22,871 runs (simple +3.55) |
| Exact fork, step 1: `view` vs `think` | −0.46 [−3.69, +2.75]; 192 issues, 659 runs |
| Exact fork, step 2: `view` vs `think` | −1.08 [−5.48, +2.87]; 126 issues, 430 runs |
| Exact fork, step 3: `view` vs shell look-up | +1.18 [−3.92, +7.19]; 42 issues, 141 runs |

"Exact fork" means tries of the same issue whose history so far (every message and tool call) is identical, which then choose different next actions. Natural experiments exist only where the agent is genuinely undecided (the positivity condition). This scope condition belongs in the discussion. The null results at early forks fit He et al.'s finding that outcome uncertainty sits mostly at the task level.

### 3. Newer AIs from NVIDIA: "test first" makes no difference

**In plain words:**

- **The data:** 255,000 tries by four newer AIs (MiniMax-M2.5, Qwen3.5-122B, Qwen3.6-27B, DeepSeek-V4-Flash), with up to 3 tries of the same bug.
- **They really vary.** MiniMax writes a test before fixing about half the time. Qwen almost always does. DeepSeek rarely does. So fair comparisons are possible.
- **Result: writing a test before fixing made no difference.** Comparing tries of the same bug, success was the same, within about 1 point. Near coin flips gave the same answer: tries with identical histories where one wrote a test and the other did something else.
- **The trap is even bigger here.** A simple comparison says testing first helps by 8 to 23 points. The fair comparison says zero.
- **Catch:** NVIDIA removed tries that ran out of steps or produced no fix, which is exactly where the older AI got hurt. So this data shows only that *when the newer AIs finish*, testing first does not change the result.
- **Difference in timing:** the newer AIs test *after* reading the code, because their instructions say so. The older AI reproduced *before* reading any code.

**Technical details** (tries with a known outcome; same-issue = weighted within-issue difference, 95% CI from resampling issues):

| Subset (tool / AI / task source) | Test first | Simple difference | Same-issue difference |
| --- | ---: | ---: | --- |
| OpenHands / DeepSeek-V4-Flash / scale-swe | 10.3% | −3.0 | −0.9 [−2.6, +0.7] (884 issues) |
| OpenHands / MiniMax-M2.5 / swe-rebench-v2 | 50.7% | +12.2 | +0.1 [−0.9, +1.0] (2,701 issues) |
| OpenHands / Qwen3.5-122B / swe-rebench-v2 | 89.8% | +15.5 | −0.7 [−2.8, +1.1] (886 issues) |
| SWE-agent / MiniMax-M2.5 / swe-rebench-v2 | 56.9% | +7.7 | +0.5 [−0.4, +1.5] (3,263 issues) |
| SWE-agent / Qwen3.5-122B / swe-rebench-v2 | 84.3% | +21.5 | +1.7 [−0.6, +4.1] (687 issues) |
| SWE-agent / Qwen3.6-27B / scale-swe | 97.9% | +22.7 | −0.1 [−3.6, +3.5] (192 issues) |
| SWE-agent / Qwen3.6-27B / swe-rebench-v2 | 93.8% | +21.5 | +1.3 [−0.8, +3.4] (617 issues) |
| **All subsets** | — | +0.4 (pooled across AIs) | **+0.3 [−0.3, +0.8]** (9,230 issues, 24,353 tries) |

- **Near coin flips:** 152,058 pairs of tries share an identical action history until they diverge (median step 3). "Write a test file now" vs. "anything else now" gives −0.0 [−1.1, +1.0] (3,594 pairs, 2,525 issues). "Test now" vs. "edit code now" happens in only 11 pairs, too few to use.
- **Ran code before the first edit vs. not:** −0.1 [−0.7, +0.5].
- **Selection:** half of the issue groups have fewer tries than their subset's maximum, consistent with NVIDIA's documented removal of tries that hit the step limit, ended early or had empty patches. Outcomes are unknown (−1) for 16.5% of test-first and 17.7% of fix-first tries.
- **Simpson's paradox:** within each subset the simple comparison favours test-first by up to +23 points, but pooled across subsets it is only +0.4, because the AIs that test most often also differ in overall success.

### 4. Other tips, cost, and languages

**In plain words:**

- **"Check your fix" might help a little.** For the older AI, tries that re-ran their test after fixing fixed about 3 more bugs per 100 than other tries of the same bug. The simple comparison says 10. This is not a pure coin flip, because tries that ran out of room never got to check, so treat it as "may help a little". The newer AIs almost always check (97–100%), so they can't be compared.
- **The other tips do nothing measurable for the newer AIs.** Running the project's tests, writing extra tests and cleaning up all stay within about 1 point in fair comparisons. Simple comparisons swing by 7 to 11 points in either direction.
- **Following "test first" costs extra work.** The older AI used about 10 more steps, roughly a fifth more. The newer AIs used about 2 more steps and read a little more text, with no gain.
- **The same everywhere.** In no programming language did "test first" clearly help: Python, Go, Java, JavaScript, TypeScript, Rust and PHP all came out about zero.
- **AIs differ hugely in how often they follow the tip** (10% to 98%), yet following it doesn't change their success.

**Technical details:**

| Tip / measure | AI | Share who do it | Simple | Same issue |
| --- | --- | ---: | ---: | --- |
| Re-ran a repro/test script after the first fix | Older (issue × day) | 53.9% | +10.2 | **+3.0 [+2.2, +3.8]** (2,410 issues) |
| …only among tries that wrote a repro script | Older | — | +17.2 | +6.2 [+4.3, +8.1] (986 issues) |
| Ran the test suite after fixing | Older | 4.2% | +1.7 | +0.3 [−1.3, +1.7] |
| Ran code after fixing | Newer (NVIDIA) | 99.7% | +9.0 | +1.0 [−2.5, +4.7] (244 issues; too few "no") |
| Ran the project's tests after fixing | Newer | 96.4% | +10.1 | +1.0 [−0.3, +2.2] |
| Wrote extra tests after fixing | Newer | 73.9% | −7.4 | +0.3 [−0.2, +0.8] |
| Cleaned up files (`rm`) | Newer | 75.9% | −0.4 | +0.8 [+0.3, +1.3] |
| Ran the project's tests before fixing | Newer | 38.7% | −10.6 | −0.5 [−0.9, −0.0] |
| Cost of test first: steps | Older / newer | — | — | +10.0 [+8.8, +11.2] / +1.8 [+1.4, +2.3] |
| Cost of test first: text written / read (thousand chars) | Newer | — | — | +1.5 [+0.9, +2.0] / +3.7 [+2.7, +4.6] |

**Test first minus fix first, by language (newer AIs, same issue):** Python −0.2 [−0.9, +0.6]; Go −0.0 [−1.5, +1.5]; TypeScript +0.1 [−1.4, +1.7]; JavaScript +1.1 [−0.6, +2.8]; Rust +0.4 [−1.8, +2.4]; Java +2.4 [−0.6, +5.6]; PHP +2.5 [−0.7, +5.8].

**Caveat:** everything after the first fix is a mid-run behaviour. Comparing tries of the same issue removes the "easy bug" trap, but not every difference between tries. In particular, tries that ran out of room could not check their fix. Only the very first move (older AI) and the exact forks are true coin flips.

The detailed plan below is for writing the paper.

## The paper

**Working title (options):**

- *Do Tips for AI Coding Agents Work? Fair Comparisons from 400,000 Logged Runs*
- *The Reproduction Illusion: Why Simple Log Analyses Overstate What Helps Coding Agents*

**Main contributions:**

1. **A fair way to analyze agent logs.** Repeated tries of the same bug differ only by sampling chance, so they form natural experiments: at the first step exactly, and at forks after identical histories. This needs no new compute. It includes an audit that found a hidden mixture of AI versions inside a public dataset, and a correction for it.
2. **The tip result.** "Reproduce the bug first" lowers success for the older AI when it is done before any code is read (−3.5 per 100), and makes no difference for four newer AIs (+0.3) among finished tries. The older AI's harm runs through running out of context.
3. **The size of the trap.** Across 6 AIs and many tips, simple comparisons report effects of 7 to 23 points where fair comparisons find about 0. Pooling across AIs can even reverse the direction (Simpson's paradox).
4. **Supporting results.** "Check your fix" may help a little; other tips have no measurable effect; following tips costs extra steps; the null holds across 7 languages; compliance varies from 10% to 98% without changing success.

**Novelty statement:** this is a plausible contribution, not a certified novelty claim. The idea that sampling noise makes an action locally random given its history was stated by He et al. ([arXiv 2608.22960](https://arxiv.org/abs/2608.22960)), who use it with environment replay. Our distinctions are: existing logs only; naturally shared histories at large scale; population-level effects of named tips; six AIs; and using fair comparisons to measure the bias of simple ones.

## Data and feasibility

All numbers come from regex-based action labels. Label validation (task C2 in the project plan) is still to be done.

### Older AI: nebius/SWE-agent-trajectories

| Fact | Value | Implication |
| --- | --- | --- |
| Models | 3, all Nebius Llama-3.1 fine-tunes: `swe-agent-llama-70b` (74,792 runs, all 3,591 issues), `-8b` (4,053), `-405b` (1,191) | Model comparison is not viable; the dataset is essentially one agent sampled many times |
| Runs per issue (70B) | median 13, 90th percentile 47, max 91 | Strong within-issue replication |
| Issues with mixed outcomes | 807 (some runs solve, some do not) | Within-issue contrasts are informative |
| Repositories | about 1,276 | Broad heterogeneity for subgroup analysis |
| System prompt | 1 distinct prompt across all 70B runs | Identical inputs, as the design requires |
| Issue text | 162 issues have more than one text variant | Restrict to the modal text per issue |
| **Policy-version mixing** | Solve rate by evaluation day ranges **1.7% to 50.7%**; share of runs that reproduce first ranges **29% to 70%** | The "70B" label very likely mixes checkpoints. This matches Nebius's blog: trajectories came from "instruct models or earlier versions of action generators we trained". Comparisons are stratified by batch. |
| Runs with no eval log | 10,518 runs, all unresolved, so they cannot be dated | Report both with and without them; use bounds |
| First action | 46.9% `create reproduce*.py`; 47.5% search or navigate | The prompt says: "Always start by trying to replicate the bug that the issues discusses." Compliance at step 0 is essentially a coin flip. |

**Estimates by design** (reproduce-first minus search-first, identical issue text):

| Design | Issues | Runs | Difference | 95% CI |
| --- | ---: | ---: | ---: | --- |
| Pooled across everything (confounded) | — | 70,608 | −1.7 points (15.8% vs 17.6%) | — |
| Within issue, runs with eval logs | 1,202 | 23,254 | −4.2 points | [−5.3, −3.0] |
| Within issue, no-log runs counted as failures | 1,349 | 29,020 | −3.7 points | [−4.7, −2.7] |
| **Within issue × evaluation day** (the natural experiment) | 1,049 (1,873 strata) | 13,522 | **−3.5 points** | **[−4.8, −2.2]** |

### Newer AI (Nebius): nebius/SWE-rebench-openhands-trajectories

| Fact | Value |
| --- | --- |
| Runs / issues | 67,074 / 6,306; median 10 runs per issue (10th to 90th percentile: 7 to 15); 1,741 issues with mixed outcomes |
| Configuration | One model and one scaffold (Qwen3-Coder-480B-A35B-Instruct, OpenHands v0.54.0); `config.toml` sets temperature 0.7, top_p 0.8 |
| Instructions | Explore first; then (Phase 4) "before implementing any fix, create a script to reproduce and verify the issue" |
| Compliance | Test script before the first code edit in 97.9% of runs, so there is no comparison group for the main question |

### Newer AIs (NVIDIA): nvidia/Open-SWE-Traces

| Fact | Value |
| --- | --- |
| Subsets used | 7 subsets that use the `str_replace_editor` tool and have repeated tries: SWE-agent (MiniMax-M2.5, Qwen3.5-122B, Qwen3.6-27B ×2) and OpenHands (MiniMax-M2.5, Qwen3.5-122B, DeepSeek-V4-Flash) |
| Tries / issue groups | 254,933 / 117,664; up to 3 tries per issue; one teacher configuration per issue |
| Languages | 9 (Python, TypeScript, Go, JavaScript, Rust, Java, PHP, C++, C) |
| Instructions (SWE-agent) | "1. As a first step, it might be a good idea to find and read code relevant to the <pr_description> 2. Create a script to reproduce the error…" |
| Filtering by NVIDIA ([arXiv 2606.16038](https://arxiv.org/abs/2606.16038)) | Removed tries that hit the step limit, ended early, had empty patches, altered tests, misused tools or used "git hacking". Every comparison is among tries that finished normally. |
| License | CC BY 4.0 (with MIT, Apache 2.0 and BSD code) |

## Closest related work

| Work | What it does | How our project differs |
| --- | --- | --- |
| He et al., "What Process Evaluation of Coding Agents Actually Measures" ([arXiv 2608.22960](https://arxiv.org/abs/2608.22960), preprint, Aug 2026) | Prefix-conditioned identification: decoding noise is exogenous, so an action is locally randomized given its prefix. Estimates step effects by **replaying** 8 continuations per prefix on 499 SWE-bench Verified localization episodes. Finds step-level effects underpowered and uncertainty concentrated at the task level. | Same identification logic, which we must credit. We use **naturally shared histories in existing logs**, estimate **population-level effects of named tips** across six AIs (which addresses their power problem), and measure the bias of simple analyses. |
| Khanna, "Same Task, Different Fate" ([SWGeno '26](https://doi.org/10.1145/3844178.3844642)) | Shows that pooling across tasks inflates behavior–outcome tests (inflation factor ≥1.62 versus 1.02 when stratified by task); one published effect vanishes and another reverses when the task is held fixed. Cites both Nebius datasets. | They address between-task confounding. We add coin-flip comparisons, the AI-version mixture, named tips, newer AIs and cost. |
| "Beyond Resolution Rates" ([arXiv 2604.02547](https://arxiv.org/abs/2604.02547)) | Across 19 agents, gathering context before editing and validating go with success; the length–failure link reverses once difficulty is controlled. | Correlational across agents. We test such claims with fair same-bug comparisons. |
| Validation studies: "Validation Evidence in LLM Repair Agents" ([arXiv 2607.28871](https://arxiv.org/abs/2607.28871)); SWE-Touch ([arXiv 2608.02499](https://arxiv.org/abs/2608.02499)) | Passing validations often don't test the bug; failures are linked to insufficient validation. The evidence is observational. | We estimate how much "check your fix" and related tips change success within the same bug. |
| Instruction-following studies: Harness-IF ([arXiv 2608.11727](https://arxiv.org/abs/2608.11727)); RepoComplianceBench ([arXiv 2607.26819](https://arxiv.org/abs/2607.26819)); CLAUDE.md / AGENTS.md adherence ([arXiv 2605.10039](https://arxiv.org/abs/2605.10039)); OctoBench ([arXiv 2601.10343](https://arxiv.org/abs/2601.10343)); AgentIF ([arXiv 2505.16944](https://arxiv.org/abs/2505.16944)) | Measure *whether* agents follow instructions. | We measure *whether following them changes success*, and find large differences in compliance with no difference in outcomes. |
| "Evaluating AGENTS.md" ([arXiv 2602.11988](https://arxiv.org/abs/2602.11988), ICLR 2026) | Context-file instructions push agents toward more exploration and testing without higher success, at more than 20% extra cost. | Consistent with our results. We isolate single tips with fair comparisons in logged data. |
| Evidence on reproduction: SWE-Doctor ([arXiv 2607.00990](https://arxiv.org/abs/2607.00990)); traceability study ([arXiv 2506.08311](https://arxiv.org/abs/2506.08311)); "Coding Agents Don't Know When to Act" ([arXiv 2605.07769](https://arxiv.org/abs/2605.07769)) | Removing the reproduction stage hurts SWE-Doctor's own pipeline. Only about 40% of reproducers trigger the bug. Reproduction instructions reduce unnecessary patches but cause wrongful abstention. | None isolates whether reproducing first helps, causally, across many real issues and several AIs. |
| Causal Agent Replay ([arXiv 2606.08275](https://arxiv.org/abs/2606.08275)); causal attribution estimators ([arXiv 2609.06445](https://arxiv.org/abs/2609.06445)) | Per-trajectory counterfactual attribution by re-execution; validated on synthetic causal models. | Population-level estimates from real logged data; no re-execution. |
| Forking Paths (Bigelow et al., ICLR 2025); ROME "crucial forks" ([arXiv 2512.24873](https://arxiv.org/abs/2512.24873)); Tree-GRPO ([arXiv 2509.21240](https://arxiv.org/abs/2509.21240), ICLR 2026); "When Agents Commit Too Soon" ([arXiv 2606.22936](https://arxiv.org/abs/2606.22936)) | Resample from prefixes to find decisive steps, branch rollouts during RL training, or describe divergence across runs. | Interventional or training-time machinery, or correlational. We do causal analysis of decisions already in logs. |
| Within-study comparisons (LaLonde 1986; Cook, Shadish & Wong 2008) | Use a randomized benchmark to measure the bias of observational estimators. | Template for RQ3. This is the first application of the design to agent-trajectory analytics that we found. |
| Data source: Open-SWE-Traces ([arXiv 2606.16038](https://arxiv.org/abs/2606.16038)) | NVIDIA's training dataset of agent trajectories. | We use it for analysis, not training, and document how its filtering limits causal questions. |
| Reproduction as a recommended step: CodeR ([arXiv 2406.01304](https://arxiv.org/abs/2406.01304)); Multi-SWE-bench ([arXiv 2504.02605](https://arxiv.org/abs/2504.02605)) | Agent designs that start by reproducing the bug, and an observational conclusion that reproduction is critical for successful repair. | The beliefs we test. Our fair comparisons find that reproducing first hurt the older AI and made no difference for newer AIs. |
| Mündler et al., "Coding Agents Are 'Fixing' Correct Code" ([SRI Lab blog](https://www.sri.inf.ethz.ch/blog/fixedcode), Mar 2026; companion to arXiv 2605.07769) | A prompt asking agents only to "reproduce before patching" left Sonnet 4.6 unchanged and lowered GPT-5.4 mini (60.5% → 47.5%) on already-fixed issues. | An interventional prompt study on a modified task. It agrees with our finding that the instruction alone does not help. We measure natural variation on real issues across seven AIs. |
| Sam-Bodden, "What Context Does a Coding Agent Actually Need to Act?" ([arXiv 2607.09691](https://arxiv.org/abs/2607.09691)) | Temperature-0 API inference flips about 9% of per-instance outcomes. | Independent evidence that repeated tries vary by chance, which our design relies on. |
| Gao et al., ACT*ONOMY ([arXiv 2605.13625](https://arxiv.org/abs/2605.13625)) | A taxonomy and pipeline for describing agent behavior across trajectories. | Descriptive; no effect estimates. |

Ideas considered and dropped during the search (label noise, memorization, run-to-run variance, process quality, behavior correlations, self-verification validity, model comparison) are listed with reasons in [PROJECT_OPTIONS.md](PROJECT_OPTIONS.md) and [RELATED_WORK.md](RELATED_WORK.md). This is a focused search, not a systematic review. **Refreshed on 2026-10-20 (Phase E3):** four searches (reproduce-first effects; within-task and repeated-run analyses; Open-SWE-Traces analyses; issue-stratified behavior effects) found no work that estimates the effect of reproducing or testing first from repeated tries of the same issue; the four rows added above are the relevant new or newly checked items. Re-check again before submission, since new preprints appear weekly.

## Research questions and status

1. **RQ1 – Fair design and data audit.** When do logged repeated tries form a valid natural experiment? This covers identical inputs, the mixture of AI versions, positivity (the AI must be undecided) and placebo checks. **Status:** done for the older AI. Scope shown with the Nebius newer AI, which never hesitates.
2. **RQ2 – Does "reproduce / test first" help?** **Status:** done. Older AI −3.5 (coin flip); newer AIs +0.3 (same bug) and −0.0 (near coin flips).
3. **RQ3 – How wrong are simple analyses?** Compare simple and fair estimates across first moves, tips and six AIs, including Simpson's paradox across AIs. **Status:** results in hand. Needs one summary figure (simple vs. fair estimates).
4. **RQ4 – Other tips and their cost.** "Check your fix", running the project's tests, writing extra tests, cleaning up; extra steps and text. **Status:** done. Needs a multiple-comparison correction (Benjamini–Hochberg).
5. **RQ5 – For which bugs and languages?** **Status:** first pass done (issue types for the older AI; 7 languages for the newer AIs). Optional: causal forests.
6. **RQ6 – Why does it hurt the older AI?** **Status:** partial (context exhaustion +11 points; +10 steps). To do: time until the right file is found, using gold patches from nebius/SWE-bench-extra.

## Methods

- **Data engineering.** [analysis/build_runs.py](../../analysis/exploratory/build_runs.py) builds the older AI's run table. [analysis/build_nvidia_runs.py](../../analysis/exploratory/build_nvidia_runs.py) streams NVIDIA's data and labels each action. Tries are dated from eval-log timestamps (older AI).
- **Fair comparisons.** Stratified differences in means within the same issue (and the same day for the older AI); exact-fork comparisons; 95% intervals from resampling whole issues; within-group permutation tests; Manski-style bounds for undated runs.
- **Heterogeneity.** Subgroup estimates (done); causal forests or DR-learners (optional).
- **Mechanism.** Context exhaustion and step counts (done); localization timing (to do).
- **Validation (no human effort).** Each label is checked against the computer's reply (e.g. a file-created message) to estimate precision. The assistant also labels 600 sampled actions blind, and agreement is measured with Cohen's κ (target ≥ 0.8). This is reported as AI-assisted validation (plan task C2).
- **Multiple comparisons.** Report every tip × AI estimate; apply Benjamini–Hochberg FDR to secondary results.

## Go/no-go checks

1. **Passed.** The estimate is stable when strata are made finer: −3.5 to −4.0 for day, 6-hour, hour and 60-minute-gap batches, and −2.8 for 10-minute gaps. The hour does not predict the first action within a day (p = 0.45).
2. **Passed.** Positivity: at the hour level, 808 issues and 5,342 runs remain in groups that contain both arms.
3. **Passed with day batches.** With undated runs added back realistically the estimate is −3.1; the rough worst case is −1.4. With hour batches the adversarial worst case reaches +0.6, so report the bounds.
4. **Partly done.** The automated label check passed. Automatic label validation (target κ ≥ 0.8; plan task C2) is still open and low-risk for the older AI, because labels come from literal command names. It matters more for NVIDIA's multilingual shell commands.
5. **Done.** Nebius newer AI: the method works (2,070 exact forks at step 0), but it tests first in 97.9% of runs, so the main question cannot be tested on it.
6. **Done.** Modern-AI evidence: in NVIDIA's data, four newer AIs vary in how often they test first (10% to 98%). Same-bug and near-coin-flip estimates are about zero.

## Threats

- **Compound treatment.** The sampled first turn includes both the thought and the command. The estimand is "the agent's own first turn, restricted to category c", which corresponds to rejection-sampling the first turn until it lands in c.
- **Residual version mixing** within an evaluation day (older AI): addressed with finer strata and sensitivity analysis.
- **Survivorship in NVIDIA's data.** Tries that hit the step limit or had no patch were removed, and 16–18% of outcomes are unknown. This can hide harm that works through running out of steps, and makes newer-AI results conditional on finishing.
- **Mid-run behaviours are not coin flips.** "Check your fix" and other later tips are compared within the same bug, but tries can still differ in how the run went. Report them as weaker evidence than first-step effects.
- **Label heuristics.** Actions are detected from command names and shell patterns across 9 languages and are validated automatically: against the computer's replies, plus a blind second labeling by the assistant, disclosed as AI-assisted.
- **Multiple comparisons.** Many tips × AIs; correct for false discoveries.
- **External validity.** Six open-weight AIs in two scaffolds (SWE-agent, OpenHands). Closed frontier models (GPT, Claude, Gemini) are not covered.
- **Licensing.** The Nebius datasets are CC BY 4.0; the SWE-agent card also notes the Llama 3.1 license for model outputs. NVIDIA Open-SWE-Traces is CC BY 4.0.

## Next steps

1. **Paper outline:** sections, figures and where each finding goes.
2. **"Why" analysis:** localization timing (needs gold patches from nebius/SWE-bench-extra).
3. **Automatic label validation:** checks against the computer's replies, plus a blind second labeling of 600 actions (plan task C2).
4. **False-discovery correction** for the secondary results.
5. **Re-check related work** right before submission.

## Code and outputs

From the project root, after `pip install -r analysis/exploratory/requirements.txt`:

```
python -I analysis/exploratory/build_runs.py                     # older AI run table
python -I analysis/exploratory/go_no_go_checks.py > analysis/exploratory/results/go_no_go_checks.txt
python -I analysis/exploratory/options_probes.py  > analysis/exploratory/results/options_probes.txt
python -I analysis/exploratory/older_ai_more.py   > analysis/exploratory/results/older_ai_more.txt
python -I analysis/exploratory/newer_ai_checks.py > analysis/exploratory/results/newer_ai_checks.txt
python -I analysis/exploratory/build_nvidia_runs.py              # streams NVIDIA data (~10 min)
python -I analysis/exploratory/nvidia_checks.py   > analysis/exploratory/results/nvidia_checks.txt
```

Every script uses a fixed random seed, so rerunning reproduces the numbers above.

## Possible venues

Empirical software engineering (MSR, ESEM, FSE/ICSE research tracks), applied or causal data-science tracks (KDD Applied Data Science, CLeaR), or agent and evaluation workshops. Check current deadlines; none were verified here.
