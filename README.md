# Do Tips for AI Coding Agents Help?

We test, with data, whether popular tips given to AI coding assistants (for example, "reproduce the bug first") actually help them fix more bugs. We use about 400,000 logged tries by six AIs. We also show that the usual way of analyzing these logs gives wrong answers.

The method: the same AI tried each bug many times. Comparing tries **of the same bug** is fair, and at the very first step the AI's choice is decided purely by sampling chance, like a coin toss.

- Project plan (locked 8-step procedure and engineering standards): [`docs/project/PROJECT_PLAN.md`](docs/project/PROJECT_PLAN.md)
- Proposal and results so far: [`docs/project/DS_PROJECT_PROPOSAL.md`](docs/project/DS_PROJECT_PROPOSAL.md)

## Install

Requires [uv](https://docs.astral.sh/uv/). Supported platforms: Linux (x86_64, ARM64), Windows (x86_64) and macOS (ARM64).

```bash
uv sync                    # creates .venv with exact locked versions (includes dev tools)
uv run agent-tips env      # shows platform, paths and compute device
```

Optional GPU support for model training (CUDA on Linux and Windows, Apple MPS on macOS):

```bash
uv sync --extra gpu
```

The statistical analyses run on CPU; a GPU is never required.

## Run the pipeline

Each step of the locked procedure is a command:

| Command | Step |
| --- | --- |
| `agent-tips download` | 2: collect data |
| `agent-tips clean` | 3: clean and prepare |
| `agent-tips eda` | 4: explore |
| `agent-tips features` | 5: create features |
| `agent-tips model` | 6: model and analyze |
| `agent-tips evaluate` | 7: evaluate |
| `agent-tips figures` | 8: communicate |
| `agent-tips run-all` | everything, in order |

Steps are added phase by phase (see the plan). Until a step is built, its command exits with code 2. The exploratory scripts that produced the first results are kept, unchanged, in [`analysis/exploratory/`](analysis/exploratory/).

## Develop

```bash
uv run pre-commit install           # run checks before every commit
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
uv run --group docs mkdocs serve    # documentation site at http://127.0.0.1:8000
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License and citation

Code is released under the [MIT License](LICENSE). To cite this project, use [`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button reads it.

## Data

Raw data is not stored in git. It is downloaded from Hugging Face:
[nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories),
[nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories),
[nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces).
All three are CC BY 4.0. The first also notes the Llama 3.1 license for model outputs.
