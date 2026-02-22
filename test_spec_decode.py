"""Speculative decoding correctness test using MT-Bench dataset.

Verifies that draft-model speculative decoding produces outputs identical
to non-speculative decoding.

Tests the 4-way invariant:
    non-spec BS=1 (anchor) == non-spec BS=N == spec BS=1 == spec BS=N

Usage:
    python test_spec_decode.py --batch-invariant
    python test_spec_decode.py --no-batch-invariant
"""

import argparse
import os

# ── Parse args BEFORE vllm import (env var is read at import time) ─────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Test speculative decoding correctness on MT-Bench"
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
        help="Target model (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Draft model (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=3,
        help="Number of speculative tokens K (default: 3)",
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
import gc  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
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
    speculative_config: dict | None = None,
) -> LLM:
    """Create a vLLM engine, optionally with speculative decoding."""
    kwargs = dict(
        model=model,
        max_num_seqs=128,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=False,
        attention_config={"backend": "FLASH_ATTN"},
    )
    if speculative_config is not None:
        kwargs["speculative_config"] = speculative_config
        kwargs["disable_log_stats"] = False  # enable metrics collection
    return LLM(**kwargs)


def destroy_engine(llm: LLM):
    """Shut down engine subprocesses and free GPU memory."""
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def collect_spec_metrics(llm: LLM) -> dict:
    """Extract spec decode metrics from engine."""
    metrics = llm.get_metrics()
    name_to_value = {}
    for metric in metrics:
        name_to_value[metric.name] = metric

    result = {}
    drafts_metric = name_to_value.get("vllm:spec_decode_num_drafts")
    accepted_metric = name_to_value.get("vllm:spec_decode_num_accepted_tokens")
    draft_tokens_metric = name_to_value.get("vllm:spec_decode_num_draft_tokens")

    if drafts_metric and accepted_metric and draft_tokens_metric:
        num_drafts = drafts_metric.value
        num_accepted = accepted_metric.value
        num_draft_tokens = draft_tokens_metric.value

        result["num_drafts"] = num_drafts
        result["num_accepted"] = num_accepted
        result["num_draft_tokens"] = num_draft_tokens

        if num_drafts > 0:
            result["mean_accepted_len"] = 1 + (num_accepted / num_drafts)
        if num_draft_tokens > 0:
            result["acceptance_rate"] = num_accepted / num_draft_tokens

    return result


def main():
    batch_inv = os.environ["VLLM_BATCH_INVARIANT"]
    K = args.num_speculative_tokens

    print("=" * 80)
    print("Speculative Decoding Correctness Test")
    print("=" * 80)
    print(f"  Target model:         {args.model}")
    print(f"  Draft model:          {args.draft_model}")
    print(f"  K (spec tokens):      {K}")
    print(f"  Max tokens:           {args.max_tokens}")
    print(f"  GPU mem util:         {args.gpu_memory_utilization}")
    print(f"  Max model len:        {args.max_model_len}")
    print(f"  VLLM_BATCH_INVARIANT: {batch_inv}")

    prompts = load_mt_bench_prompts()
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    # ── Phase 1: Non-speculative runs ──────────────────────────────────
    print("\n>>> Creating non-speculative engine...")
    llm = create_engine(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    t0 = time.time()
    nonspec_bs1 = run_bs1(llm, prompts, sampling_params, label="non-spec BS=1")
    t1 = time.time()
    print(f"  non-spec BS=1: {t1 - t0:.1f}s")

    t2 = time.time()
    nonspec_bsN = run_batch(llm, prompts, sampling_params, label="non-spec BS=N")
    t3 = time.time()
    print(f"  non-spec BS=N: {t3 - t2:.1f}s")

    print("\n>>> Destroying non-speculative engine...")
    destroy_engine(llm)

    # ── Phase 2: Speculative runs ──────────────────────────────────────
    print("\n>>> Creating speculative engine...")
    spec_config = {
        "method": "draft_model",
        "model": args.draft_model,
        "num_speculative_tokens": K,
    }
    llm = create_engine(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        speculative_config=spec_config,
    )

    t4 = time.time()
    spec_bs1 = run_bs1(llm, prompts, sampling_params, label="spec BS=1")
    t5 = time.time()
    print(f"  spec BS=1: {t5 - t4:.1f}s")

    t6 = time.time()
    spec_bsN = run_batch(llm, prompts, sampling_params, label="spec BS=N")
    t7 = time.time()
    print(f"  spec BS=N: {t7 - t6:.1f}s")

    spec_metrics = collect_spec_metrics(llm)

    print("\n>>> Destroying speculative engine...")
    destroy_engine(llm)

    # ── Phase 3: Compare all against anchor (non-spec BS=1) ───────────
    anchor = nonspec_bs1
    total = len(prompts)

    mm_nonspec_bsN = compare_results(prompts, anchor, nonspec_bsN)
    mm_spec_bs1 = compare_results(prompts, anchor, spec_bs1)
    mm_spec_bsN = compare_results(prompts, anchor, spec_bsN)

    # ── Report ─────────────────────────────────────────────────────────
    wall = (t3 - t0) + (t7 - t4)
    lines = [
        "",
        "=" * 60,
        f"Spec Decode Report  [VLLM_BATCH_INVARIANT={batch_inv}]",
        f"  Target: {args.model}  Draft: {args.draft_model}  K={K}",
        "-" * 60,
        format_comparison_line("non-spec BS=N vs anchor:", mm_nonspec_bsN, total),
        format_comparison_line("spec BS=1 vs anchor:", mm_spec_bs1, total),
        format_comparison_line("spec BS=N vs anchor:", mm_spec_bsN, total),
    ]
    if spec_metrics:
        rate = spec_metrics.get("acceptance_rate", 0)
        mlen = spec_metrics.get("mean_accepted_len", 0)
        lines.append("-" * 60)
        lines.append(f"  Spec decode metrics: acceptance_rate={rate:.2f}, mean_accepted_len={mlen:.2f}")
    lines.extend([
        "-" * 60,
        f"  Wall time: {wall:.1f}s",
        "=" * 60,
    ])
    report = "\n".join(lines)
    tag = "on" if batch_inv == "1" else "off"
    save_report(report, f"spec_decode_inv_{tag}.txt")


if __name__ == "__main__":
    main()
