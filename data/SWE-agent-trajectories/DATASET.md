# SWE-agent Trajectories Dataset

## At a glance

- **Source:** [`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
- **Local copy:** this directory; Parquet shards are in `data/`
- **Size:** 80,036 trajectory records in one `train` split, stored in 12 Parquet shards (about 1.11 GB)
- **Task:** software-engineering agents attempt to resolve GitHub issues by inspecting and editing repositories, then produce patches and run tests.

The dataset was generated with the SWE-agent framework using multiple action-generating models. Its issue instances come from `nebius/SWE-bench-extra` and the development split of `princeton-nlp/SWE-bench`. The dataset describes trajectories, not necessarily unique issues: `instance_id` identifies the repository and issue, and multiple records can represent different attempts or models.

## Record structure

Each row is one agent trajectory and has these top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `instance_id` | string | Repository and issue identifier. |
| `model_name` | string | Model used to generate the agent's actions. |
| `target` | boolean | Whether the trajectory is marked as having solved the issue. |
| `trajectory` | list of records | Ordered interaction log; see fields below. |
| `exit_status` | string | Agent run's completion status. |
| `generated_patch` | string | Final patch produced by the agent. |
| `eval_logs` | string | Test/evaluation output for the patch. |

The stored Parquet schema represents each `trajectory` as a list with the following fields:

| Field | Type | Notes |
| --- | --- | --- |
| `cutoff_date` | string | Present on each trajectory entry; the card does not define its precise semantics. |
| `mask` | boolean | Present on each entry; the card does not define how it should be interpreted. |
| `role` | string | Speaker or entry role. The card describes system prompt, AI, and user/environment entries. |
| `system_prompt` | string | System-prompt content when present. |
| `text` | string | Entry content, such as an action, reasoning text, or an observation. |

The dataset card describes the log as including model reasoning and actions plus environment observations. Treat these as recorded agent traces; do not assume every field is user-facing text or that the `mask` field has a particular filtering meaning without checking the data-generation implementation.

## Published summary statistics

These are aggregate values reported in the dataset card, grouped by `target`; they are not recalculated here.

| Statistic | Issue resolved (`target=true`, 13,389 records) | Not resolved (`target=false`, 66,647 records) |
| --- | ---: | ---: |
| Average trajectory steps | 31.3 | 58.4 |
| Average context length (Llama 3 tokenizer) | 8,352.4 | 15,241 |
| Average files edited in final patch | 1.33 | 2.17 |
| Average lines edited in final patch | 20.7 | 61.0 |
| Runs ending in a submit | 94.6% | 57.6% |
| Runs exiting due to context | 5.31% | 30.4% |
| At least one correct file opened | 83% | 40% |

The card groups remaining exits as “Other” (0.37% for resolved and 12% for unresolved). These statistics are descriptive; they do not establish that a particular trajectory caused an issue to be resolved.

## Loading the local copy

With the Hugging Face `datasets` package installed, load the Parquet shards directly without downloading the remote repository:

```python
from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files={"train": "data/SWE-agent-trajectories/data/*.parquet"},
)
```

## Licensing and use

The dataset card lists **CC BY 4.0**. It also cautions that the underlying repository instances have their own licenses; check those before reusing repository code or patches. The card further notes that use of model outputs must comply with the **Llama 3.1 License**. Preserve attribution and review those terms for the intended use.
