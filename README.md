# Do Tips for AI Coding Agents Help?

We test, with data, whether popular tips given to AI coding assistants (for example, "reproduce the bug first") actually help them fix more bugs. We use about 400,000 logged tries by six AIs. We also show that the usual way of analyzing these logs gives wrong answers.

The method: the same AI tried each bug many times. Comparing tries **of the same bug** is fair, and at the very first step the AI's choice is decided purely by sampling chance, like a coin toss.

## Results

| Question | Answer (fair, same-bug comparison) |
| --- | --- |
| Does reproducing the bug first help an older AI (Llama-70B, SWE-agent)? | **No, it hurt:** −3.5 bugs fixed per 100 tries (95% CI −4.8 to −2.2) |
| Does testing first help four newer AIs (seven setups)? | **No meaningful difference:** +0.1 (90% CI −0.4 to +0.5, inside the ±2 margin set in advance) |
| Do simple analyses of the same logs agree? | **No:** they differ by 10.9 points on average and point the other way in 37% of tip × AI pairs |

- **Paper draft:** [`paper/main.pdf`](paper/main.pdf)
- **Step reports** (results, robustness, tables): [`docs/`](docs/), also on the [documentation site](https://rajanyadav0307.github.io/tips_agentic_ai/)
- **Plan** (locked 8-step procedure and engineering standards): [`docs/project/PROJECT_PLAN.md`](docs/project/PROJECT_PLAN.md)

Every number in the paper is generated into [`outputs/tables/numbers.tex`](outputs/tables/numbers.tex) by `agent-tips figures`; nothing is typed by hand.

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

To reproduce everything from scratch (about 25 GB of downloads):

```bash
uv sync
uv run agent-tips run-all          # add --no-repro to skip the two-run reproducibility check
cd paper && latexmk -pdf main.tex  # optional: rebuild the paper (needs a LaTeX install with acmart)
```

The exploratory scripts that produced the first results are kept, unchanged, in [`analysis/exploratory/`](analysis/exploratory/).

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

Raw data is not stored in git. `uv run agent-tips download` fetches about 25 GB into `data/raw/`. Every file is pinned to a dataset revision and checked against the SHA-256 checksum published by Hugging Face. Downloads are recorded in `data/raw/MANIFEST.json`, so reruns skip verified files. See [`docs/02_data_inventory.md`](docs/02_data_inventory.md). The data comes from Hugging Face:
[nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories),
[nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories),
[nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces).
All three are CC BY 4.0. The first also notes the Llama 3.1 license for model outputs.
