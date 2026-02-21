"""Minimal repro: VLLM_BATCH_INVARIANT triggers unknown env var warning."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len=128,
    attention_config={"backend": "FLASH_ATTN"},
)
print(llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=5))[0].outputs[0].text)
