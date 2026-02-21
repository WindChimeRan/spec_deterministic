# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repo is an experiment workspace for verifying the **algorithmic correctness of batch speculative decoding** in vLLM. The vLLM source lives at `/scratch3/hzz5361/vllm` (added as an additional working directory).

### Model Pairs Under Test

| Target Model | Draft Model | Reference |
|---|---|---|
| Qwen/Qwen3-4B | Qwen/Qwen3-0.6B | Yang et al., 2025 |
| lmsys/vicuna-7b-v1.3 | double7/vicuna-68m | Zheng et al., 2023 |
| THUDM/glm-4-9b | THUDM/glm-4-0.6b | GLM et al., 2024 |

**Proof-of-concept phase**: Run Qwen3-4B/0.6B only. All pairs run at the end.

## Environment Setup

```bash
# Use vllm's uv venv (from the vllm repo root)
cd /scratch3/hzz5361/vllm
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Required Environment Variable

Always enable batch invariance for all experiments:
```bash
export VLLM_BATCH_INVARIANT=1
```
This is read at import time in `vllm/model_executor/layers/batch_invariant.py`. Requires NVIDIA GPUs with compute capability >= 9.0 (H100, H200, B100, B200).

## vLLM Codebase Key Paths

### Speculative Decoding

- **Core logic**: `vllm/v1/spec_decode/` — `eagle.py`, `draft_model.py`, `ngram_proposer.py`, `medusa.py`, `metadata.py`, `metrics.py`, `utils.py`
- **GPU worker impl**: `vllm/v1/worker/gpu/spec_decode/` — `eagle.py`, `rejection_sample.py`, `utils.py`
- **Config**: `vllm/config/speculative.py` — `SpeculativeConfig` class; methods: `ngram`, `medusa`, `mlp_speculator`, `draft_model`, `suffix`, `eagle`, `eagle3`, `mtp`
- **Example script**: `examples/offline_inference/spec_decode.py`

### Batch Invariance

- **Implementation**: `vllm/model_executor/layers/batch_invariant.py` — Triton kernels for deterministic matmul, softmax, rms_norm
- **Docs**: `docs/features/batch_invariance.md`
- **Benchmark**: `benchmarks/benchmark_batch_invariance.py`

### Tests

- **Spec decode unit tests**: `tests/v1/spec_decode/` — test_eagle.py, test_ngram.py, test_mtp.py, test_tree_attention.py
- **Spec decode e2e**: `tests/v1/e2e/test_spec_decode.py` — includes `test_draft_model_correctness` using Qwen3-1.7B/0.6B pairs
- **Batch invariance tests**: `tests/v1/determinism/test_batch_invariance.py`
- **Basic correctness**: `tests/basic_correctness/test_basic_correctness.py` — compares vLLM greedy vs HuggingFace

## Running Experiments

### Offline Inference with Draft Model Speculative Decoding

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-4B",
    speculative_config={
        "method": "draft_model",
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 3,
    },
    trust_remote_code=True,
)
```

### Running Tests

```bash
# Single test file
pytest tests/v1/e2e/test_spec_decode.py::test_draft_model_correctness -v

# Spec decode unit tests
pytest tests/v1/spec_decode/ -v

# Batch invariance
pytest tests/v1/determinism/test_batch_invariance.py -v

# Basic correctness (vLLM vs HuggingFace)
pytest tests/basic_correctness/test_basic_correctness.py -v
```

### Lint / Type Check (vllm repo)

```bash
# Ruff lint
ruff check vllm/

# Mypy (pydantic plugin enabled)
mypy vllm/
```

## Dataset

Use **`philschmid/mt-bench`** (80 prompts, 8 categories: writing, roleplay, reasoning, math, coding, extraction, STEM, humanities).

- Already used by vLLM's own spec decode tests (`tests/v1/spec_decode/test_acceptance_length.py`, `examples/offline_inference/spec_decode.py`)
- Right size for correctness testing — diverse enough without being slow
- Single-turn prompts, simple format

```python
from datasets import load_dataset

ds = load_dataset("philschmid/mt-bench", split="train")
prompts = [sample["turns"][0] for sample in ds]  # first turn, 80 prompts
```

## Correctness Verification Strategy

The experiment verifies that batch speculative decoding produces outputs identical to non-speculative decoding under `VLLM_BATCH_INVARIANT=1`. Key metrics from `llm.get_metrics()`:

- `vllm:spec_decode_num_drafts` — total speculation rounds
- `vllm:spec_decode_num_accepted_tokens` — accepted tokens
- `vllm:spec_decode_num_accepted_tokens_per_pos` — per-position acceptance
- **Mean acceptance length** = 1 + (accepted / drafts)

When target == draft model with greedy sampling, expected acceptance rate is 1.0 and acceptance length is K+1 (where K = `num_speculative_tokens`).
