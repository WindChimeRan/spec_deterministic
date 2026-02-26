# Speculative Decoding Correctness Tests

## Conclusion

vLLM's speculative decoding is **algorithmically correct**. On H100 with batch invariance enabled, all four conditions (non-spec BS=1, non-spec BS=N, spec BS=1, spec BS=N) produce **identical outputs** (80/80 match). Mismatches on A100 and H100-without-invariance are purely from hardware floating-point non-determinism — batch invariance requires compute capability >= 9.0 (H100+), so it is a no-op on A100.

### Results (anchor = non-speculative BS=1)

| GPU | Batch Invariant | Batch Inv Test | non-spec BS=N | spec BS=1 | spec BS=N |
|-----|----------------|----------------|---------------|-----------|-----------|
| **H100** | **ON** | **MATCH** | **MATCH** | **MATCH** | **MATCH** |
| H100 | OFF | NO MATCH (66) | NO MATCH (66) | MATCH | NO MATCH (62) |
| A100 | ON | NO MATCH (62) | NO MATCH (62) | NO MATCH (58) | NO MATCH (56) |
| A100 | OFF | NO MATCH (58) | NO MATCH (58) | NO MATCH (61) | NO MATCH (63) |

Model pair: Qwen3-4B (target) + Qwen3-0.6B (draft), K=3, 80 prompts from mt-bench, max 256 tokens.

## Quick Start

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install datasets
uv pip install vllm --torch-backend=auto

CUDA_VISIBLE_DEVICES=5 bash run_all.sh
```

Reports are saved to `A100_reports/` and `H100_reports/`.

## Motivation

vLLM's speculative decoding uses an MQA scorer, which is essentially chunked prefilling — the same path as normal batch inference. This means enabling batch invariance (`VLLM_BATCH_INVARIANT=1`) makes speculative decoding deterministic across batch sizes as well.

This test suite verifies algorithmic correctness by checking that outputs are identical across four conditions:

1. **Non-speculative BS=1** (anchor ground truth)
2. **Non-speculative BS=N**
3. **Speculative BS=1** (Qwen3-4B target + Qwen3-0.6B draft)
4. **Speculative BS=N**

Each condition is run with batch invariance both on and off, on both A100 and H100 GPUs.
