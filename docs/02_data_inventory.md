# Step 2: Data Inventory

**Downloaded and verified:** 2026-10-20, with `agent-tips download`
**Total:** 121 files, 24.35 GB in `data/raw/` (not in git). Every file's SHA-256 matches the checksum published by Hugging Face, and the record is kept in `data/raw/MANIFEST.json`.

## In plain words

- We use five data sources:
  - three collections of AI recordings (one older AI, five newer AIs);
  - two lists of the "real fixes" (gold patches) that human developers made.
- Each source is **pinned** to an exact version, like a photo taken on a specific date, so the results can always be reproduced even if the source changes later.
- Every file was checked against its official fingerprint (checksum). Re-running the download skips files that are already verified.
- The row counts match our earlier exploration exactly.

## Sources

| Key | Hugging Face dataset | Revision (pinned) | Last changed on Hub | License | Files | Size | Rows | Used for |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `older_ai` | [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | `68195a1450865274106246d0d0296a1d6807b88e` | 2024-12-23 | CC BY 4.0 (model outputs also subject to the Llama 3.1 license) | 12 | 1.11 GB | 80,036 | P1, S1, S2, S7, S8, S10, S11, S13 |
| `newer_ai_nebius` | [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | `35455389ab51bf5e2306bfd436ef72d0f98bf882` | 2025-12-27 | CC BY 4.0 | 1 | 2.08 GB | 67,074 | S12; RQ1 scope check |
| `newer_ai_nvidia` | [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) (7 of its subsets) | `1db674173119d71df45c22007af99e214600711a` | 2026-10-13 | CC BY 4.0 | 106 | 21.13 GB | 254,933 | P2, P3, S2–S7, S9 |
| `gold_extra` | [nebius/SWE-bench-extra](https://huggingface.co/datasets/nebius/SWE-bench-extra) | `11dcbfb30e19552df2a2f8030bd764adc95c92a5` | 2025-05-28 | CC BY 4.0 | 1 | 0.02 GB | 6,376 | Gold patches for the older AI (S11) |
| `gold_dev` | [princeton-nlp/SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench) (dev split) | `e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6` | 2025-03-03 | No license field on the card (SWE-bench is MIT-licensed on GitHub); only the dev split is used | 1 | <0.01 GB | 225 | Gold patches for the older AI (S11) |

## Details per source

### `older_ai`

| Model label | Tries | Issues |
| --- | ---: | ---: |
| `swe-agent-llama-70b` (**used**) | 74,792 | 3,591 |
| `swe-agent-llama-8b` (not used) | 4,053 | 528 |
| `swe-agent-llama-405b` (not used) | 1,191 | 100 |

Gold patches cover **3,585 of the 3,591** issues: 3,386 from SWE-bench-extra and 199 from the SWE-bench dev split.

### `newer_ai_nebius`

67,074 tries on 6,306 issues: one model (Qwen3-Coder-480B), one scaffold (OpenHands v0.54.0), temperature 0.7.

### `newer_ai_nvidia` (7 subsets)

| Scaffold | AI | Task source | Tries | Issues | Outcome unknown (−1) |
| --- | --- | --- | ---: | ---: | ---: |
| OpenHands | DeepSeek-V4-Flash | scale-swe | 26,453 | 15,765 | 0.0% |
| OpenHands | MiniMax-M2.5 | swe-rebench-v2 | 43,603 | 19,022 | 22.3% |
| OpenHands | Qwen3.5-122B | swe-rebench-v2 | 40,463 | 18,451 | 23.5% |
| SWE-agent | MiniMax-M2.5 | swe-rebench-v2 | 46,819 | 19,522 | 23.2% |
| SWE-agent | Qwen3.5-122B | swe-rebench-v2 | 20,334 | 10,511 | 25.7% |
| SWE-agent | Qwen3.6-27B | scale-swe | 40,802 | 16,997 | 11.3% |
| SWE-agent | Qwen3.6-27B | swe-rebench-v2 | 36,459 | 17,396 | 7.6% |
| **Total** | | | **254,933** | | |

**Known filtering by NVIDIA** ([arXiv 2606.16038](https://arxiv.org/abs/2606.16038)): tries that hit the step limit, ended early, had empty patches, altered tests, misused tools or used "git hacking" were removed before release. This limits what this data can show (see the threats section of the proposal).

### Gold patches

| Source | Rows | Created (range) |
| --- | ---: | --- |
| SWE-bench-extra | 6,376 | 2015-10-21 to 2024-04-30 |
| SWE-bench dev | 225 | 2019-01-20 to 2023-09-26 |

The two sources write dates differently (`…T…Z` vs `… +00:00`). Step 3 normalizes them.

## Where the data lives

```
data/raw/
  MANIFEST.json                      repo, revision, path, size, SHA-256 for every file
  older_ai/data/train-*.parquet
  newer_ai_nebius/trajectories.parquet
  newer_ai_nvidia/data/<scaffold>/<ai>/<task source>/train-*.parquet
  gold_extra/data/train-*.parquet
  gold_dev/data/dev-*.parquet
```

The data cards for the first two sources are in [`docs/data/`](data/older_ai.md).

## How to reproduce

```bash
uv run agent-tips download            # download and verify (about 25 GB)
uv run agent-tips download --verify   # re-hash every file
uv run agent-tips download --dry-run  # show what is missing
```

Versions and file patterns live in [`configs/default.yaml`](../configs/default.yaml).
