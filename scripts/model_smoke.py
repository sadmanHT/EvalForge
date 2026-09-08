#!/usr/bin/env python3
"""Connected-hardware Phase 01 smoke for the exact frozen 7B model revision.

This hard gate intentionally uses the real frozen model and rejects merely non-empty text:
the model must load at the exact revision and emit valid JSON containing a canonical
root-cause code for a deliberately obvious fixture incident.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.phase1 import load_json_yaml, validate_root_cause_code  # noqa: E402

MODEL = load_json_yaml(ROOT / "configs/model.yaml")
TAXONOMY = load_json_yaml(ROOT / "configs/label-taxonomy.yaml")


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install torch, transformers, and accelerate on a machine with enough RAM/VRAM "
            "before running this hard gate."
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
                "Classify the incident. Return only a JSON object with exactly the keys "
                "root_cause_code and reasoning. root_cause_code must be one of: "
                "n_plus_one_query, database_connection_leak, disk_exhaustion, "
                "broken_payment_configuration, memory_leak, no_fault. "
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

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"frozen model did not produce strict JSON: {text!r}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("structured output must be a JSON object")
    if set(payload) != {"root_cause_code", "reasoning"}:
        raise RuntimeError(f"unexpected structured-output keys: {sorted(payload)}")
    if not isinstance(payload["reasoning"], str) or not payload["reasoning"].strip():
        raise RuntimeError("structured output reasoning must be a non-empty string")

    code = payload["root_cause_code"]
    if not isinstance(code, str):
        raise RuntimeError("root_cause_code must be a string")
    validate_root_cause_code(code, TAXONOMY)
    if code != "n_plus_one_query":
        raise RuntimeError(
            f"frozen model failed obvious compatibility fixture: expected n_plus_one_query, got {code}"
        )

    print(f"MODEL_ID={model_id}")
    print(f"REVISION={revision}")
    print(json.dumps(payload, ensure_ascii=False))
    print("Phase 01 connected model smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
