"""Batch invariance test using MT-Bench dataset.

Verifies that VLLM_BATCH_INVARIANT=1 produces identical outputs
for the same prompt regardless of batch composition (BS=1 vs BS=N).

Usage:
    python test_batch_invariance.py --batch-invariant
    python test_batch_invariance.py --no-batch-invariant
"""

import argparse
import os

# ── Parse args BEFORE vllm import (env var is read at import time) ─────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Test batch invariance on MT-Bench using vLLM"
    )
    parser.add_argument(
        "--batch-invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable VLLM_BATCH_INVARIANT (default: on)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Model name (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per prompt (default: 256)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max model context length (default: 4096)",
    )
    return parser.parse_args()


args = parse_args()

os.environ["VLLM_BATCH_INVARIANT"] = "1" if args.batch_invariant else "0"

# ── Now safe to import vllm ────────────────────────────────────────────
import time  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

from utils import (  # noqa: E402
    compare_results,
    format_comparison_line,
    load_mt_bench_prompts,
    run_batch,
    run_bs1,
    save_report,
)


def create_engine(
    model: str,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> LLM:
    """Create a vLLM engine configured for batch invariance testing."""
    return LLM(
        model=model,
        max_num_seqs=128,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=False,
        attention_config={"backend": "FLASH_ATTN"},
    )


def main():
    batch_inv = os.environ["VLLM_BATCH_INVARIANT"]

    print("=" * 80)
    print("MT-Bench Batch Invariance Test")
    print("=" * 80)
    print(f"  Model:                {args.model}")
    print(f"  Max tokens:           {args.max_tokens}")
    print(f"  GPU mem util:         {args.gpu_memory_utilization}")
    print(f"  Max model len:        {args.max_model_len}")
    print(f"  VLLM_BATCH_INVARIANT: {batch_inv}")

    prompts = load_mt_bench_prompts()

    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    print("\nCreating engine...")
    llm = create_engine(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    t0 = time.time()
    bs1_results = run_bs1(llm, prompts, sampling_params, label="non-spec BS=1")
    t1 = time.time()
    print(f"  BS=1 phase: {t1 - t0:.1f}s")

    t2 = time.time()
    bsN_results = run_batch(llm, prompts, sampling_params, label="non-spec BS=N")
    t3 = time.time()
    print(f"  BS=N phase: {t3 - t2:.1f}s")

    mismatches = compare_results(prompts, bs1_results, bsN_results)

    # Report
    total = len(prompts)
    wall = t3 - t0
    report = "\n".join([
        "",
        "=" * 60,
        f"Batch Invariance Report  [VLLM_BATCH_INVARIANT={batch_inv}]",
        "-" * 60,
        format_comparison_line("BS=N vs BS=1 (anchor):", mismatches, total),
        "-" * 60,
        f"  Wall time: {wall:.1f}s",
        "=" * 60,
    ])
    tag = "on" if batch_inv == "1" else "off"
    save_report(report, f"batch_invariance_inv_{tag}.txt")


if __name__ == "__main__":
    main()
