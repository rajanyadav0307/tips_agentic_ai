# SWE-rebench OpenHands Trajectories (newer AI)

- **Source:** [`nebius/SWE-rebench-openhands-trajectories`](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories), downloaded 2026-10-19 as a single file, `trajectories.parquet` (about 2.1 GB)
- **What it is:** 67,074 recorded attempts by one AI (Qwen3-Coder-480B-A35B-Instruct, run in the OpenHands v0.54.0 agent framework) at fixing 6,306 GitHub issues from 1,823 Python repositories. Each issue was attempted about 10 times (10th to 90th percentile: 7 to 15).
- **Settings:** a single model and configuration for every run; temperature 0.7 and top_p 0.8 (from the dataset's `config.toml`); at most 100 steps per run.
- **Fields used:** `trajectory_id`, `instance_id`, `trajectory` (system, user, assistant and tool messages; tool-call arguments are JSON strings), `exit_status`, `resolved` (1 = issue fixed).
- **Instructions the agent receives:** explore first, then (Phase 4) "before implementing any fix, create a script to reproduce and verify the issue". The agent follows this in about 98% of runs.
- **License:** CC BY 4.0 (see the dataset card).
