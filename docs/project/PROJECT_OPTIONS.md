# Project Options for the SWE-agent Dataset (compared on 2026-10-19)

**Dataset:** [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories), the older AI with 75,000 recorded tries. The newer AI's dataset was used only as a check.
**Related files:** [RELATED_WORK.md](RELATED_WORK.md) (first idea search), [DS_PROJECT_PROPOSAL.md](DS_PROJECT_PROPOSAL.md) (detailed plan for the recommended project and the results of its checks)

## In plain words

- We compared 11 possible projects. For each one we checked two things: has someone already done it, and can our data answer it? The most promising ones were also tested on the data.
- Most ideas are already taken, or the data cannot answer them.
- **The best project is still the "test drive first" study:** does reproducing the bug first help the AI? It is now stronger than before:
  - **It holds for every type of bug we checked.** Starting by reproducing the bug means fewer fixes. The harm is a little smaller when the bug report already includes example code.
  - **Simple comparisons get fooled.** Just comparing successful tries with failed ones gives wrong answers. Starting with a file search looks 7 points better, but a fair comparison shows no real benefit. Starting by reproducing looks harmless, but a fair comparison shows it hurts. The newer AI shows the same trap.
  - **The likely reason:** reproducing uses up the AI's limited memory, so it more often runs out before finishing the fix.
- **Weak spot:** only the older AI can show the main finding. The newer AI almost always reproduces first, so there is nothing to compare it with.
- **Update (NVIDIA check, 2026-10-19):** four newer AIs in NVIDIA's data vary enough to compare. For them, writing a test before fixing made **no difference** among tries that finished. A simple comparison would claim +8 to +23 points. NVIDIA removed tries that ran out of steps, so their side of the "runs out of room" harm cannot be seen. Details are in DS_PROJECT_PROPOSAL.md, section 3 ("Newer AIs from NVIDIA").

## All options and verdicts

| # | Project idea | Verdict | Why |
| --- | --- | --- | --- |
| 1 | **Does "reproduce the bug first" help?** A fair coin-flip comparison across repeated tries of the same bug | **Best (recommended)** | Our searches found no study that does this. The signal is strong and survived every check (see DS_PROJECT_PROPOSAL.md). |
| 2 | How wrong are simple success-vs-failure comparisons? | Include inside #1 | Partly done: Khanna, "Same Task, Different Fate" ([SWGeno '26](https://doi.org/10.1145/3844178.3844642)) cites both Nebius datasets and shows that pooling across tasks misleads. Our coin-flip estimates add a "true answer" to compare simple methods against. |
| 3 | Which behaviors go with success? | Taken | XEPV ([arXiv 2606.15579](https://arxiv.org/abs/2606.15579), uses this dataset), [Beyond Resolution Rates](https://arxiv.org/abs/2604.02547), [TraceProbe](https://arxiv.org/abs/2607.06184), [AgentLens](https://arxiv.org/abs/2605.12925) |
| 4 | Predict failure early | Taken | [Fail-Fast, Restart-Smart](https://arxiv.org/abs/2608.03222); [EsfandyariDoulabi et al.](https://arxiv.org/abs/2610.05572) |
| 5 | Which model is best? | Data can't answer | Three sizes of one Llama fine-tune; the 70B model produced 93% of runs |
| 6 | Does the AI know when it failed? (its own success claims) | Weak | Its final thoughts follow a fixed template ("changes applied, let's submit"). Only 5.4% claim the fix works, and such claims barely relate to real success. Adjacent work: [self-attribution bias](https://arxiv.org/abs/2603.04582), [experiential confidence](https://arxiv.org/abs/2609.17708). |
| 7 | Find broken or unfair benchmark tasks | Mostly taken; weak here | Audits of SWE-bench Verified and Pro ([Epoch: Verified](https://epoch.ai/benchmarks/swe-bench-verified/review), [Epoch: Pro](https://epoch.ai/benchmarks/swe-bench-pro/review)); IRT-based audits ([arXiv 2605.30504](https://arxiv.org/abs/2605.30504)). With one AI we cannot tell a hard task from a broken one. |
| 8 | How the AI changed across training rounds | Data can't confirm | Versions are not labeled; only evaluation dates hint at them. General self-training work exists ([arXiv 2512.15011](https://arxiv.org/abs/2512.15011)). |
| 9 | Learning from test failures (feedback uptake) | Backup | Still open but crowded, and mainly a measurement study (see RELATED_WORK.md) |
| 10 | Partial credit for near misses | Add-on only | A partial-credit score already exists (SWE-EVO "Fix Rate", [arXiv 2512.18470](https://arxiv.org/abs/2512.18470)) |
| 11 | Time until the AI finds the right file (survival analysis) | Add-on to #1 | We found no published survival model for this; closest are [arXiv 2511.00197](https://arxiv.org/abs/2511.00197) (localization levels) and [arXiv 2609.29508](https://arxiv.org/abs/2609.29508) (a hazard model for a different event). Best used to explain *why* reproducing hurts. |

## Evidence behind the recommendation

**Simple vs. fair comparisons, older AI.** Points of resolution rate for each first move compared with `search_dir`. The fair comparison groups by same issue × same evaluation day; 95% intervals come from resampling issues.

| First move | Simple (all pooled) | Same issue | Fair (issue × day) |
| --- | ---: | --- | --- |
| `create` (reproduce) | −0.1 | −3.9 [−5.2, −2.6] | −3.4 [−5.1, −1.3] |
| `find_file` | +7.0 | −1.1 [−2.1, −0.0] | −0.5 [−2.0, +0.9] |
| `ls` | −1.0 | −0.9 [−1.7, −0.1] | −1.2 [−2.6, +0.1] |
| `open` | +9.5 | +2.7 [−1.2, +6.1] | +5.0 [−1.5, +11.0] |
| `grep` | −3.3 | −0.6 [−2.4, +1.3] | −1.2 [−3.7, +1.5] |

**Newer AI.** `think` vs `view` as the first move: simple +3.55, fair (exact coin flip) −0.06 [−0.79, +0.64]. Test script first vs. fix first: simple +12.09, same issue −0.73 [−3.79, +1.99].

**Does the reproduce-first effect depend on the type of issue?** Older AI, fair comparison. Overall −3.5 [−4.8, −2.1].

| Issue... | Yes | No |
| --- | --- | --- |
| has a code block | −2.6 [−4.2, −1.1] | −4.6 [−6.8, −2.6] |
| has a traceback | −4.5 [−7.3, −1.6] | −3.3 [−4.8, −1.9] |
| names a `.py` file | −3.8 [−5.9, −1.7] | −3.3 [−5.0, −1.8] |
| is longer than the median | −3.3 [−5.2, −1.5] | −3.6 [−5.6, −1.8] |

The effect is consistently negative. It is somewhat smaller when the issue includes code, possibly because a ready-made example makes reproduction cheap.

**Final claims vs. reality.** 47,857 submitted runs; 5.4% claim the fix works. Resolved 26.6% with a claim vs. 23.9% without. Within the same issue: +1.2 [−0.3, +2.7].

## The paper the recommended project becomes

**Working title:** *Should Coding Agents Reproduce the Bug First? Natural-Experiment Evidence from 75,000 Runs*

1. **Design.** Repeated runs of the same issue differ only by sampling chance, so they form natural experiments. We correct for a hidden mixture of model versions in the public dataset.
2. **Finding.** Following the prompt's "reproduce first" tip lowers resolution by about 3.5 points (roughly a fifth) for this agent, across all issue types checked. The likely mechanism is context exhaustion: +11 points more runs are force-submitted at the context limit, with about 10 more steps.
3. **Warning.** Simple comparisons mislead in both directions, on an older and a newer agent. Fair within-issue comparisons fix this. This extends Khanna (2026) with coin-flip benchmarks.
4. **Scope.** Natural experiments exist only where an agent is undecided. The newer agent follows the tip 98% of the time, so it provides no comparison.

**Gaps to close before submission:**

- **Modern-agent evidence for the main finding.** Find another public dataset where an agent sometimes skips reproduction, or run a small randomized experiment (prompt with vs. without the tip).
- **Mechanism through localization timing.** Needs gold patches from nebius/SWE-bench-extra.
- **Validation of the action labels** (automatic, no human effort; plan task C2).

## Code and outputs

- [analysis/options_probes.py](../../analysis/exploratory/options_probes.py) writes [analysis/results/options_probes.txt](../../analysis/exploratory/results/options_probes.txt): simple vs. fair, issue types, and final claims.
- [analysis/newer_ai_checks.py](../../analysis/exploratory/newer_ai_checks.py) writes [analysis/results/newer_ai_checks.txt](../../analysis/exploratory/results/newer_ai_checks.txt): the newer AI.
- [analysis/go_no_go_checks.py](../../analysis/exploratory/go_no_go_checks.py) writes [analysis/results/go_no_go_checks.txt](../../analysis/exploratory/results/go_no_go_checks.txt): robustness checks of the main finding.
