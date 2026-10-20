# Exploratory scripts (frozen record)

These scripts produced the first results (2026-10-19), before the project was rebuilt as the `agent_tips` package. They are kept unchanged so the early numbers can be traced. **Final results come only from the package** (`agent-tips <step>`).

The raw data has since moved:

| Old path used by these scripts | Current location |
| --- | --- |
| `data/SWE-agent-trajectories/data/*.parquet` | `data/raw/older_ai/data/*.parquet` |
| `data/SWE-rebench-openhands-trajectories/trajectories.parquet` | `data/raw/newer_ai_nebius/trajectories.parquet` |
| NVIDIA data streamed from Hugging Face | `data/raw/newer_ai_nvidia/data/...` (after `agent-tips download`) |

To rerun a script, pass the new path as its `--data` or `--runs` argument.
