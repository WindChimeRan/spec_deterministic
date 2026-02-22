"""Shared utilities for batch invariance and speculative decoding tests."""

from datasets import load_dataset
from vllm import LLM, SamplingParams


def load_mt_bench_prompts() -> list[str]:
    """Load first-turn prompts from philschmid/mt-bench."""
    ds = load_dataset("philschmid/mt-bench", split="train")
    prompts = [sample["turns"][0] for sample in ds]
    print(f"Loaded {len(prompts)} prompts from mt-bench")
    return prompts


def run_bs1(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    label: str = "BS=1",
) -> list[tuple[str, tuple]]:
    """Run each prompt individually (BS=1) and collect results.

    Returns list of (text, token_ids) tuples.
    """
    results = []
    total = len(prompts)

    print(f"\n{'=' * 80}")
    print(f"Running {total} prompts individually ({label})")
    print("=" * 80)

    for i, prompt in enumerate(prompts):
        outputs = llm.generate([prompt], sampling_params)
        if len(outputs) != 1:
            raise RuntimeError(f"Expected 1 output, got {len(outputs)}")
        if len(outputs[0].outputs) < 1:
            raise RuntimeError(f"No completions for prompt {i}")

        text = outputs[0].outputs[0].text
        token_ids = tuple(outputs[0].outputs[0].token_ids)
        results.append((text, token_ids))

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{total}] tokens={len(token_ids)}, "
                f"preview: {text[:60]}..."
            )

    print(f"  Completed all {total} {label} runs")
    return results


def run_batch(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    label: str = "BS=N",
) -> list[tuple[str, tuple]]:
    """Run all prompts as a single batch and collect results.

    Returns list of (text, token_ids) tuples.
    """
    total = len(prompts)

    print(f"\n{'=' * 80}")
    print(f"Running {total} prompts as batch ({label})")
    print("=" * 80)

    outputs = llm.generate(prompts, sampling_params)
    if len(outputs) != total:
        raise RuntimeError(f"Expected {total} outputs, got {len(outputs)}")

    results = []
    for i, output in enumerate(outputs):
        if len(output.outputs) < 1:
            raise RuntimeError(f"No completions for prompt {i}")
        text = output.outputs[0].text
        token_ids = tuple(output.outputs[0].token_ids)
        results.append((text, token_ids))

    print(f"  Completed {label} batch run")
    return results


def find_first_divergence(tids_a: tuple, tids_b: tuple) -> int | None:
    """Find the first position where two token ID sequences differ."""
    min_len = min(len(tids_a), len(tids_b))
    for pos in range(min_len):
        if tids_a[pos] != tids_b[pos]:
            return pos
    if len(tids_a) != len(tids_b):
        return min_len
    return None


def compare_results(
    prompts: list[str],
    anchor: list[tuple[str, tuple]],
    candidate: list[tuple[str, tuple]],
) -> list[dict]:
    """Compare candidate results against anchor results for each prompt.

    Returns list of mismatch dicts. Empty list means all matched.
    """
    if len(anchor) != len(candidate) or len(anchor) != len(prompts):
        raise RuntimeError(
            f"Result count mismatch: prompts={len(prompts)}, "
            f"anchor={len(anchor)}, candidate={len(candidate)}"
        )

    mismatches = []
    for i, (prompt, (text_a, tids_a), (text_c, tids_c)) in enumerate(
        zip(prompts, anchor, candidate)
    ):
        if tids_a == tids_c:
            continue

        mismatches.append({
            "prompt_idx": i,
            "prompt_preview": prompt[:120],
            "anchor_token_ids": tids_a,
            "candidate_token_ids": tids_c,
            "first_diff_pos": find_first_divergence(tids_a, tids_c),
        })

    return mismatches


def format_comparison_line(
    label: str, mismatches: list[dict], total: int, indent: int = 2
) -> str:
    """Format a single comparison result line with optional mismatch details."""
    prefix = " " * indent
    num_fail = len(mismatches)
    if num_fail == 0:
        return f"{prefix}{label:<30s} MATCH ({total}/{total})"

    lines = [f"{prefix}{label:<30s} NO MATCH ({num_fail}/{total} differ)"]
    for m in mismatches[:3]:
        lines.append(
            f"{prefix}  prompt {m['prompt_idx']}: "
            f"diverge@pos {m['first_diff_pos']}, "
            f"len anchor={len(m['anchor_token_ids'])} "
            f"vs candidate={len(m['candidate_token_ids'])}"
        )
    if num_fail > 3:
        lines.append(f"{prefix}  ... and {num_fail - 3} more")
    return "\n".join(lines)
