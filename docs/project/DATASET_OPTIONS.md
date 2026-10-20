# Public Datasets of Coding-Agent Recordings (checked on 2026-10-19)

## In plain words

- Several big companies and research groups have published recordings of AI coding agents at work.
- For our project, a dataset needs three things: **many tries of the same bug**, **failed tries kept** (not only successes), and a **recent AI**.
- **Best bet to fill our weak spot: NVIDIA's Open-SWE-Traces.** It has recent AIs, the same tool our older AI used (SWE-agent), failed tries kept, and up to 3 tries per bug.
- **One catch:** in the newer versions of that tool, the instructions say "first read the relevant code, then reproduce the bug". So the "reproduce or not" choice happens in the middle of a run, which makes a pure coin-flip comparison harder.
- **Interesting clue:** SWE-agent's own instructions changed from "Always start by trying to replicate the bug" (the 2024 version our older AI used) to "read code first, then reproduce". That fits our finding, though we don't know why they changed it.

## Recordings of AIs fixing GitHub issues

| Dataset | Made by | What's inside | Tries per bug | Failed tries kept? | Notes |
| --- | --- | --- | --- | --- | --- |
| [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | Nebius (2024) | 80,036 runs, Llama-3.1 fine-tunes in SWE-agent | median 13 (70B) | Yes | **Our main dataset** |
| [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | Nebius (2025) | 67,074 runs, Qwen3-Coder-480B in OpenHands | median 10 | Yes | Already checked: tests first in 98% of runs |
| [togethercomputer/CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview) | Together AI (Feb 2026) | 258k test-verified runs (155k pass, 103k fail), 51k tasks, 1,655 repos; Qwen3-Coder-480B in OpenHands | Run IDs go up to `_run8` (count not verified) | Yes (`reward` field) | Same AI and tool family as the Nebius OpenHands set; 72.8 GB |
| [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | NVIDIA (v1.0 June 2026; later updates) | Up to ~678k rows; MiniMax, Qwen, DeepSeek and GLM families in SWE-agent, OpenHands and mini-SWE-agent; CC BY 4.0 | Up to 3 in some subsets (verified below) | Yes (`resolved` 1/0; −1 = unknown) | **Best bet for our weak spot**; paper [arXiv 2606.16038](https://arxiv.org/abs/2606.16038) |
| [JetBrains-Research/SWE-TaskFlow-SWE-Bench-Pro-Trajectories](https://huggingface.co/datasets/JetBrains-Research/SWE-TaskFlow-SWE-Bench-Pro-Trajectories) | JetBrains (Oct 2026) | 25,200 runs: 700 SWE-Bench Pro tasks × 3 runs × 4 frontier models (GPT-5.6 Luna and Terra, GPT-5.4 mini, Gemini 3.7 Flash) × 3 prompt styles; mini-SWE-agent format | 3 per model and prompt style | Yes (`status`, `resolved`) | License "other" (provider terms); paper [arXiv 2610.09633](https://arxiv.org/abs/2610.09633) |
| [SWE-bench/experiments](https://github.com/swe-bench/experiments) | SWE-bench team (official) | Logs and trajectories of every official leaderboard submission | Usually 1 per system | Yes | Good for comparing many systems; not for coin flips |
| [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | SWE-smith team | 5,017 runs (SWE-agent) | 1 | Training data | Not useful for us |
| [AlienKevin/SWE-ZERO-12M-trajectories](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories) | Community | About 12.3M runs from a 1.7B model, 100 per pull request | 100 | **No pass/fail labels** | Not useful for outcome studies |

**Verified on Open-SWE-Traces** (only the `instance_id` and `resolved` columns were read):

| Subset | Runs | Issues | Resolved | Label missing | Runs per issue (p10, p50, p90) | Issues with mixed outcomes |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `sweagent/minimax_m25/swe-rebench-v2` | 46,819 | 19,522 | 35.8% | 23.2% | 1, 3, 3 | 1,866 |
| `sweagent/qwen36_27b/scale-swe` | 40,802 | 16,997 | 47.6% | 11.3% | 1, 3, 3 | 1,172 |
| `sweagent/qwen38_27b/swe-rebench-v2` | 17,483 | 17,483 | 37.6% | 0% | 1, 1, 1 | 0 |
| `openhands/deepseek_v4_flash_0731/swe-rebench-v2` | 23,842 | 16,895 | 32.4% | 0% | 1, 1, 2 | 336 |

**Inspected in depth on 2026-10-19:** see section 3 ("Newer AIs from NVIDIA") of DS_PROJECT_PROPOSAL.md. Important: NVIDIA's paper says it removed tries that hit the step limit, ended early, produced no patch, altered tests, misused tools or used "git hacking". Every comparison on this data is therefore among tries that finished normally.

In a sample of 600 MiniMax SWE-agent runs, 97.5% start by viewing files. Only 12% create a file within their first 8 actions. The SWE-agent instructions in this data read: "1. As a first step, it might be a good idea to find and read code relevant to the <pr_description> 2. Create a script to reproduce the error…".

## Other kinds of agent tasks

| Dataset | What's inside | Notes |
| --- | --- | --- |
| Terminal-Bench 2.0: official submissions ([harborframework/terminal-bench-2-leaderboard](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard)) and a community extraction ([yoonholee/terminalbench-trajectories](https://huggingface.co/datasets/yoonholee/terminalbench-trajectories)) | 52,104 trials, 89 command-line tasks, 109 agent/model combinations, typically 5 tries each, pass/fail; Apache 2.0 (community set) | Modern frontier agents with repeated tries. Command-line tasks rather than GitHub bugs. |

## Real-world use (not benchmarks)

| Dataset | What's inside | Notes |
| --- | --- | --- |
| AIDev, the official MSR 2026 Mining Challenge dataset ([arXiv 2602.09185](https://arxiv.org/abs/2602.09185)) | 932,791 pull requests by 5 AI coding agents (Codex, Devin, Copilot, Cursor, Claude Code) across 116,211 repositories; a curated subset of 33,596 includes reviews | Outcome is merged or not. No step-by-step recordings. Already used by many challenge papers. |
| SWE-chat ([arXiv 2604.20779](https://arxiv.org/abs/2604.20779)) | Real users' sessions with coding agents | Size not checked |
