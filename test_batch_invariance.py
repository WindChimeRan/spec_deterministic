"""Batch invariance test using MT-Bench dataset.

Verifies that VLLM_BATCH_INVARIANT=1 produces identical outputs
for the same prompt regardless of batch composition (BS=1 vs BS=N).

Usage:
    python test_batch_invariance.py
    python test_batch_invariance.py --model Qwen/Qwen3-4B --max-tokens 128
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# MUST be set before any vllm imports (read at module load time)
os.environ["VLLM_BATCH_INVARIANT"] = "1"

import argparse
import time

from datasets import load_dataset
from vllm import LLM, SamplingParams


def load_mt_bench_prompts() -> list[str]:
    """Load first-turn prompts from philschmid/mt-bench."""
    ds = load_dataset("philschmid/mt-bench", split="train")
    prompts = [sample["turns"][0] for sample in ds]
    print(f"Loaded {len(prompts)} prompts from mt-bench")
    return prompts


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


def run_bs1(llm: LLM, prompts: list[str], sampling_params: SamplingParams):
    """Run each prompt individually (BS=1) and collect results.

    Returns list of (text, token_ids) tuples.
    """
    results = []
    total = len(prompts)

    print("\n" + "=" * 80)
    print(f"PHASE 1: Running {total} prompts individually (BS=1)")
    print("=" * 80)

    for i, prompt in enumerate(prompts):
        outputs = llm.generate([prompt], sampling_params)
        assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
        assert len(outputs[0].outputs) >= 1, f"No completions for prompt {i}"

        text = outputs[0].outputs[0].text
        token_ids = tuple(outputs[0].outputs[0].token_ids)
        results.append((text, token_ids))

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{total}] tokens={len(token_ids)}, "
                f"preview: {text[:60]}..."
            )

    print(f"  Completed all {total} BS=1 runs")
    return results


def run_batch(llm: LLM, prompts: list[str], sampling_params: SamplingParams):
    """Run all prompts as a single batch (BS=N) and collect results.

    Returns list of (text, token_ids) tuples.
    """
    total = len(prompts)

    print("\n" + "=" * 80)
    print(f"PHASE 2: Running {total} prompts as batch (BS={total})")
    print("=" * 80)

    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == total, f"Expected {total} outputs, got {len(outputs)}"

    results = []
    for i, output in enumerate(outputs):
        assert len(output.outputs) >= 1, f"No completions for prompt {i}"
        text = output.outputs[0].text
        token_ids = tuple(output.outputs[0].token_ids)
        results.append((text, token_ids))

    print(f"  Completed BS={total} batch run")
    return results


def compare_results(
    prompts: list[str],
    bs1_results: list[tuple[str, tuple]],
    bsN_results: list[tuple[str, tuple]],
) -> list[dict]:
    """Compare BS=1 vs BS=N results for each prompt.

    Returns list of mismatch dicts. Empty list means all passed.
    """
    assert len(bs1_results) == len(bsN_results) == len(prompts)

    mismatches = []

    print("\n" + "=" * 80)
    print("PHASE 3: Comparing BS=1 vs BS=N outputs")
    print("=" * 80)

    for i, (prompt, (text_1, tids_1), (text_N, tids_N)) in enumerate(
        zip(prompts, bs1_results, bsN_results)
    ):
        text_match = text_1 == text_N
        tids_match = tids_1 == tids_N

        if text_match and tids_match:
            print(f"  [PASS] Prompt {i:3d}: {len(tids_1)} tokens match")
        else:
            # Find first divergence position
            first_diff_pos = None
            min_len = min(len(tids_1), len(tids_N))
            for pos in range(min_len):
                if tids_1[pos] != tids_N[pos]:
                    first_diff_pos = pos
                    break
            if first_diff_pos is None and len(tids_1) != len(tids_N):
                first_diff_pos = min_len

            mismatches.append({
                "prompt_idx": i,
                "prompt_preview": prompt[:120],
                "bs1_text": text_1,
                "bsN_text": text_N,
                "bs1_token_ids": tids_1,
                "bsN_token_ids": tids_N,
                "first_diff_pos": first_diff_pos,
            })

            print(
                f"  [FAIL] Prompt {i:3d}: "
                f"text_match={text_match}, tids_match={tids_match}, "
                f"first_diff@{first_diff_pos}, "
                f"lengths=({len(tids_1)}, {len(tids_N)})"
            )

    return mismatches


def print_summary_and_assert(prompts: list[str], mismatches: list[dict]):
    """Print final summary and assert no mismatches."""
    total = len(prompts)
    num_fail = len(mismatches)
    num_pass = total - num_fail

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total prompts:  {total}")
    print(f"  Passed:         {num_pass}")
    print(f"  Failed:         {num_fail}")

    if mismatches:
        print("\n" + "-" * 80)
        print("MISMATCH DETAILS (first 5)")
        print("-" * 80)

        for m in mismatches[:5]:
            idx = m["prompt_idx"]
            print(f"\n  Prompt {idx}: {m['prompt_preview']}...")
            print(
                f"    BS=1 tokens: {len(m['bs1_token_ids'])}, "
                f"BS=N tokens: {len(m['bsN_token_ids'])}"
            )
            print(f"    First divergence at position: {m['first_diff_pos']}")

            if m["first_diff_pos"] is not None:
                pos = m["first_diff_pos"]
                tids_1 = m["bs1_token_ids"]
                tids_N = m["bsN_token_ids"]
                start = max(0, pos - 2)
                end_1 = min(len(tids_1), pos + 3)
                end_N = min(len(tids_N), pos + 3)
                print(f"    BS=1 tokens[{start}:{end_1}]: {tids_1[start:end_1]}")
                print(f"    BS=N tokens[{start}:{end_N}]: {tids_N[start:end_N]}")

            print(f"    BS=1 text[:200]: {m['bs1_text'][:200]}")
            print(f"    BS=N text[:200]: {m['bsN_text'][:200]}")

        if num_fail > 5:
            print(f"\n  ... and {num_fail - 5} more mismatches")

    print("\n" + "=" * 80)

    assert num_fail == 0, (
        f"Batch invariance VIOLATED: {num_fail}/{total} prompts produced "
        f"different outputs between BS=1 and BS=N"
    )

    print("Batch invariance VERIFIED: all prompts match between BS=1 and BS=N")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test batch invariance on MT-Bench using vLLM"
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


def main():
    args = parse_args()

    assert os.environ.get("VLLM_BATCH_INVARIANT") == "1", (
        "VLLM_BATCH_INVARIANT must be set to 1"
    )

    print("=" * 80)
    print("MT-Bench Batch Invariance Test")
    print("=" * 80)
    print(f"  Model:          {args.model}")
    print(f"  Max tokens:     {args.max_tokens}")
    print(f"  GPU mem util:   {args.gpu_memory_utilization}")
    print(f"  Max model len:  {args.max_model_len}")
    print(f"  VLLM_BATCH_INVARIANT={os.environ['VLLM_BATCH_INVARIANT']}")

    prompts = load_mt_bench_prompts()

    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    print("\nCreating engine...")
    llm = create_engine(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    t0 = time.time()
    bs1_results = run_bs1(llm, prompts, sampling_params)
    t1 = time.time()
    print(f"  BS=1 phase: {t1 - t0:.1f}s")

    t2 = time.time()
    bsN_results = run_batch(llm, prompts, sampling_params)
    t3 = time.time()
    print(f"  BS=N phase: {t3 - t2:.1f}s")

    mismatches = compare_results(prompts, bs1_results, bsN_results)
    print_summary_and_assert(prompts, mismatches)

    print(f"\nTotal wall time: {t3 - t0:.1f}s")


if __name__ == "__main__":
    main()
