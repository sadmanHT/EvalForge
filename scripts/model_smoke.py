#!/usr/bin/env python3
"""Connected-hardware Phase 01 smoke for the exact frozen 7B model revision.

This is intentionally not replaced by a tiny model or mocked response: Phase 01 requires
proof that the selected primary model itself can load and produce a structured response.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = json.loads((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install torch + transformers on a machine with enough RAM/VRAM before running this hard gate."
        ) from exc

    model_id = MODEL["base_model_id"]
    revision = MODEL["base_model_revision"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    messages = [
        {
            "role": "user",
            "content": (
                "Classify the incident. Return only JSON with keys root_cause_code and reasoning. "
                "Incident: checkout latency rose sharply and database query count increased 18x "
                "after an application change; CPU and memory stayed normal."
            ),
        }
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("frozen model produced empty output")

    print(f"MODEL_ID={model_id}")
    print(f"REVISION={revision}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
