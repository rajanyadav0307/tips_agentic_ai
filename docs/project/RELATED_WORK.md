# Related Work and Novelty Check

**Search date:** 2026-10-19
**Dataset:** [Nebius SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories), 80,036 trajectories

## Bottom line

The three project ideas previously discussed are **not novel enough as currently phrased**. In particular, a 2026 arXiv preprint reports analyzing 2,000 trajectories from this exact dataset for action-pattern/outcome associations and model behavior profiles. Newer work also studies early failure prediction and warns that pooled trajectory correlations can confuse issue difficulty with run-level behavior.

This is a focused search of the closest work, not an exhaustive systematic review. A literature search cannot certify novelty. Treat the extension proposed below as a candidate that needs a dataset feasibility check and a closer full-text comparison before we commit to it.

## Closest work

| Work | What it studies | Relevance to our ideas |
| --- | --- | --- |
| Deng, ["Your Agent Has a Genome: Sequence-Level Behavioral Analysis and Runtime Governance of LLM-Powered Autonomous Agents"](https://arxiv.org/abs/2606.15579), arXiv preprint, submitted 2026-04-05 | Encodes actions as Explore/Execute/Plan/Verify, studies action sequences and success, and describes model-level behavior profiles. Its cross-system section explicitly says it analyzes 2,000 public trajectories from `nebius/SWE-agent-trajectories`. | **Direct overlap** with success-correlate analysis and model behavior comparisons. It is a preprint, not a peer-reviewed publication in the source checked. Its XEPV abstraction is already a baseline, not a new contribution for us. |
| Khanna, ["Same Task, Different Fate: Population Structure in Trajectory-Based Analyses of Coding Agents"](https://doi.org/10.1145/3844178.3844642), SWGeno '26, published 2026-10-08 | Shows how pooled run-level behavior/outcome associations can be distorted when tasks differ; recommends within-task comparisons and permutation calibration. Its abstract reports that, after holding task fixed, one published effect becomes indistinguishable from zero and another reverses sign. | **Direct methodological overlap** with a simple success-prediction study or an unadjusted reanalysis of XEPV-style action patterns. The article page and abstract were checked; its PDF endpoint returned HTTP 403 during this search, so the exact reanalyzed effect is not asserted here. |
| EsfandyariDoulabi et al., ["Disentangling Task Difficulty from Run-Level Failure in Agent Failure Prediction"](https://arxiv.org/abs/2610.05572), arXiv preprint, submitted 2026-10-04 | Separates task-level difficulty prediction from predicting whether a particular run will fail. It reports that more than 99.93% of pooled positive-negative pairs are cross-task and that early within-task failure prediction is much weaker than pooled results suggest. | **Direct overlap and an essential evaluation warning** for both success prediction and early warning. A random row split or pooled AUROC would not establish useful run-level prediction. |
| Wang et al., ["Fail-Fast, Restart-Smart: Early Failure Prediction and Restart for SWE Agentic Tasks"](https://arxiv.org/abs/2608.03222), arXiv preprint, submitted 2026-08-04 | Trains a monitor to detect failure from partial SWE-agent runs, then evaluates stopping and restarting with an optional patch overlay. | **Direct overlap** with the proposed early-failure project. A new classifier on trajectory prefixes alone would duplicate its central question. |
| Al-Haque and Johnson, ["What Makes Software Issue Resolution Tasks Difficult for Agents?"](https://arxiv.org/abs/2608.18280), arXiv preprint; listed as to appear at ESEM 2026 | Predicts task difficulty from static prompt, repository, and patch features in the separate CoderForge-Preview dataset; reports AUC 0.863. | **Adjacent overlap** with predicting success. It studies which tasks are hard, rather than whether one run is heading toward failure, but reinforces the need to control task difficulty. |
| Son et al., ["SWE-Router: Routing in Multi-turn Agentic Software Engineering Tasks"](https://arxiv.org/abs/2607.00053), arXiv preprint; comments list the ICML 2026 Deep Learning for Code workshop | Uses a partial trajectory to decide whether to continue with a cheaper model or escalate to a stronger one. | **Adjacent overlap** with model comparison and model selection. Comparing raw solve rates is not enough to establish a new routing contribution. |
| Ma et al., ["From Patches to Trajectories: Privileged Process Supervision for Software-Engineering Agents"](https://arxiv.org/abs/2605.21996), arXiv preprint, submitted 2026-05-21 | Curates agent trajectories using reference patches and per-step progress/efficiency criteria; evaluates training on SWE-Gym and SWE-Bench. | **Adjacent overlap** with analyzing trajectory quality. It focuses on selecting training supervision, not merely correlating trajectory length with success. |
| Zheng et al., ["SWE-Prime: Fewer Trajectories, Better Performance"](https://arxiv.org/abs/2608.27449), arXiv preprint, submitted 2026-08-27 | Selects successful trajectories and useful segments for fine-tuning, arguing that a successful final result does not make every intermediate step good training data. | **Adjacent overlap** with claims about which trajectory behaviors are desirable. |
| Ding and Zhang, ["SWE-Replay: Efficient Test-Time Scaling for Software Engineering Agents"](https://arxiv.org/abs/2601.22129), arXiv preprint, submitted 2026-01-29 | Reuses archived trajectories and branches from selected intermediate states to improve test-time scaling efficiency. | **Adjacent overlap** with using multiple attempts, trajectory efficiency, or salvaging/replaying partial runs. |
| Bouzenia and Pradel, ["Understanding Software Engineering Agents: A Study of Thought-Action-Result Trajectories"](https://doi.org/10.1109/ase63991.2025.00234), ASE 2025 | Studies thought/action/result trajectories of software engineering agents. | Establishes that process-level trajectory analysis predates the new dataset-specific papers; claims of being the first to analyze agent trajectories would be too broad. |
| Chen et al., ["Beyond Final Code: A Process-Oriented Error Analysis of Software Development Agents in Real-World GitHub Scenarios"](https://arxiv.org/abs/2503.12374), accepted ICSE 2026 | Analyzes thousands of solving-phase trajectories and testing logs across eight agents and 500 SWE-bench issues, including execution errors and debugging effort. | **Close process-analysis overlap.** Our distinction must be the event-level link from a particular failed test observation to the next patch and rerun, not a general analysis of errors or testing logs. |
| Nashid et al., ["Beyond Accuracy: Behavioral Dynamics of Agentic Multi-Hunk Repair"](https://arxiv.org/abs/2511.11012), accepted TOSEM | Analyzes 1,616 repair trajectories, including tool-use sequences, TEST frequency, WRITE-TEST sequences, final regression reduction, and repair outcomes. | **Close overlap.** It counts broad test/edit patterns and final regression effects; our possible distinction is whether an edit responds to the content of the immediately preceding failure. Its data are from PolyHunk, not this Nebius corpus. |
| Wang et al., ["TrajAudit: Automated Failure Diagnosis for Agentic Coding Systems"](https://arxiv.org/abs/2605.26563), arXiv preprint, v3 submitted 2026-07-28 | Uses test failure reports to help identify the earliest decisive error in a trajectory; introduces RootSE, a 102-case benchmark with human-annotated error steps. | **Adjacent, high-overlap failure analysis.** It diagnoses where a run went wrong; it does not evaluate whether the original agent later adapted its patch to the reported test failure. It does, however, reuse some public SWE-agent trajectories, so sample-level overlap must be checked. |
| Zhao et al., ["Debugging the Debuggers: Failure-Anchored Structured Recovery for Software Engineering Agents"](https://doi.org/10.1145/3832783.3834492), ASE 2026, published 2026-10-07 | PROBE converts failed-run telemetry into a diagnosis and bounded guidance for a subsequent recovery attempt; evaluates diagnosis and recovery across repository repair, enterprise workflows, and AIOps. | **Closest intervention overlap.** Do not claim recovery from test failures is new. A narrower distinction is measuring natural, unprompted feedback uptake inside the same trajectory, rather than adding an external diagnosis/guidance side channel for another attempt. |
| Liang et al., ["Test-Driven Approaches to Software Engineering with Large Language Models: A Survey of Phases, Tasks, and Agent Skills"](https://arxiv.org/abs/2609.12012), arXiv scoping survey, submitted 2026-09-10 | Separates test availability, validity, feedback use, and evaluation independence; calls for process measures linking an observed failure to the revised artifact and rerun. | **Direct motivation, not a duplicate empirical study.** The survey identifies the measurement gap; our possible contribution would be applying an explicit uptake measure to the Nebius repository-agent traces. It is a single-workflow scoping review, not an exhaustive systematic review. |
| Liang et al., ["Do Code Language Models Follow Tests? Paired Interventions on Program Behavior"](https://arxiv.org/abs/2607.26244), arXiv preprint, v2 submitted 2026-09-07 | Uses paired, controlled test-rule changes to measure whether code language models switch behavior accordingly on function-level tasks. | **Conceptual overlap on feedback use.** It studies controlled test obedience, not how repository agents consume real test-run failures and modify code over a multi-step trajectory. |
| Zhao et al., ["A Rubric-Supervised Critic from Sparse Real-World Outcomes"](https://arxiv.org/abs/2603.03800), arXiv preprint | Defines a rubric including insufficient testing and insufficient debugging, and trains critics to score traces and support reranking, stopping, and data selection. | **Adjacent overlap.** Broad behavioral labels and trajectory scoring are already covered. A new study must measure the specific failure-to-edit relation, not introduce another general “debugging quality” score. |
| He et al., ["TestGRAD: Evolving Test Suites via Failure Pattern Momentum for SWE-Agent Ensemble"](https://arxiv.org/abs/2610.10242), arXiv preprint, submitted 2026-10-07 | Uses repeated failure patterns to improve test suites for selecting among an ensemble of candidate patches. | **Adjacent overlap.** It optimizes tests and patch selection; it does not measure one agent's spontaneous response to a failed test within its own run. |

## Check of the three proposed ideas

1. **What behaviors predict success?** No-go as a general pooled feature-importance study. The XEPV preprint already analyzes action patterns and outcomes on a 2,000-row subset of our dataset. Task difficulty and task-stratification work also make naive correlations methodologically weak.
2. **Which model is most effective?** No-go as a raw model leaderboard or behavior-profile comparison. XEPV reports model profiles on the dataset, and SWE-Router studies model escalation using partial trajectories. Any comparison would need a clearly different estimand and matched task coverage.
3. **Can failure be predicted early?** No-go as stated. Fail-Fast, Restart-Smart directly studies early failure monitoring and restart; the October 2026 failure-prediction paper also sets a high bar for within-task validation.

## Proposed project

### From Test Failure to Repair: Measuring Feedback Uptake in SWE-Agent Trajectories

**Research question:** When an agent observes a specific test failure during a repository task, does its next code edit respond to that failure, and does that response improve the next test result without introducing other failures?

The unit of analysis is a *failure-to-response episode* within one trajectory:

1. A test or validation command returns an actionable failure (for example, a named failed test, assertion mismatch, traceback, or error location).
2. The agent observes that output and makes a later code edit before terminating.
3. A subsequent test run shows whether the targeted failure disappeared, persisted, or was replaced by a regression.

The proposed contribution is an **audited measurement and characterization**, not a new failure-prediction model or recovery controller. We would define and validate a small coding scheme for responses such as targeted response, partial/ambiguous response, ignored or repeated failure, and regression-inducing response. Then we would report (a) how often each occurs, (b) the relationship between uptake and the next observed test result, and (c) differences by model, test-feedback type, and issue category. Report associations, not causal effects: trajectories are observational and agents may choose when to run tests.

### Why this may be distinct

- *Your Agent Has a Genome* studies the same Nebius dataset subset and broad action-sequence/success relationships, including E-to-Verify transitions. It does not report whether an edit is semantically responsive to the exact preceding failing assertion.
- *Beyond Accuracy* studies action categories and short tool sequences (including WRITE-TEST patterns), final repair accuracy, and regression reduction on a different multi-hunk benchmark. It does not link each specific failure message to the next patch's response.
- *TrajAudit* diagnoses the earliest decisive error and uses test output to guide diagnosis. Its target is the diagnosis model, not whether the original agent later uses a failure report.
- *Debugging the Debuggers* adds structured failure diagnosis and guidance for a subsequent recovery attempt. Our proposed scope is natural within-run behavior without injected guidance; do not claim post-failure recovery itself is novel.
- The 2026 test-driven survey explicitly separates test availability, validity, feedback use, and evaluation independence, and calls for linking failure evidence to revised artifacts and reruns. This project would operationalize that gap on real SWE-agent trajectories.
- *Do Code Language Models Follow Tests?* measures controlled rule switching in function-level code generation. It is complementary evidence, but not the same repository-level, naturally occurring feedback loop.

This is a **plausible contribution, not a certified novelty claim**. The broad idea of test-feedback-guided repair is established. The possible gap is a transparent, failure-event-level measure of unprompted feedback uptake in logged repository-agent trajectories.

### Feasibility and go/no-go checks

The dataset schema stores an ordered `trajectory` of AI and environment/user observations, plus a final `generated_patch` and `eval_logs`. The Hugging Face sample viewer shows command observations, test failures, later edits, and repeated test runs in a trajectory, so the required signal is present in at least some records. The published card reports 80,036 attempts, but does not report how many have a test-fail/edit/retest sequence.

Before scaling, inspect a stratified pilot of 100 trajectories and record:

- How many contain an explicit failed test run rather than setup, import, network, or infrastructure failure?
- How many have a subsequent code edit and a subsequent comparable test run?
- Can annotators reliably connect a failure message to a later patch hunk? Measure agreement before automating the labels.
- Does the final or later test log report enough detail to tell whether the particular failure was fixed, or only whether the whole suite passed?

**Go** only if a useful number of trajectories have complete, interpretable failure-to-edit-to-retest episodes and independent annotation reaches acceptable agreement. Otherwise, do not broaden to generic “debugging quality”; choose another question.

### Threats and safeguards

- Separate agent-caused failing tests from setup/environment failures; report both counts.
- A later test pass may reflect changed execution conditions or a different test set, not uptake. Compare the same test or test identifier when possible.
- Edits may address the failure indirectly or fix it for another reason. Use human-checked semantic labels on a stratified sample; treat automated matching as a measurement aid, not ground truth.
- One trajectory can contain many failure episodes, and one issue can have many model runs. Cluster uncertainty by `instance_id` and repository; do not treat events as independent.
- Report the denominator at each stage (all attempts, attempts with test failures, episodes with edits, episodes with comparable retests). Do not equate correlation with causal benefit.

## Search notes

Primary arXiv abstract/full-text pages, the Hugging Face dataset card, OpenAlex metadata, and the ACM Digital Library page for SWGeno '26 were checked on 2026-10-19. Preprints are labeled as preprints; publication status should be rechecked before a manuscript cites them. The dataset card reports 80,036 rows, while the XEPV paper reports using a 2,000-trajectory subset; the 2,000-row result is not evidence that the full dataset has already been exhaustively analyzed.
