# Speculative Decoding Correctness Tests

## Quick Start

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install datasets
uv pip install vllm --torch-backend=auto

CUDA_VISIBLE_DEVICES=5 bash run_all.sh
```

Reports are saved to `reports/`.

## Motivation

vLLM's speculative decoding uses an MQA scorer, which is essentially chunked prefilling — the same path as normal batch inference. This means enabling batch invariance (`VLLM_BATCH_INVARIANT=1`) makes speculative decoding deterministic across batch sizes as well.

This test suite verifies algorithmic correctness by checking that outputs are identical across four conditions:

1. **Non-speculative BS=1** (anchor ground truth)
2. **Non-speculative BS=N**
3. **Speculative BS=1** (Qwen3-4B target + Qwen3-0.6B draft)
4. **Speculative BS=N**

Each condition is run with batch invariance both on and off, producing 4 experiment reports total.
