#!/usr/bin/env python3
"""Hard Phase 01 smoke for the exact frozen primary model revision.

This script must be run on the intended connected GPU development/training environment.
It deliberately does not substitute a mock/tiny model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
MODEL = json.loads((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))
TAXONOMY = json.loads((ROOT / "configs/label-taxonomy.yaml").read_text(encoding="utf-8"))
CANONICAL_LABELS = {row["id"] for row in TAXONOMY["labels"]}


def extract_json(text: str) -> dict:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"model did not return a JSON object: {text!r}")
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"model output was not valid JSON: {text!r}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("model JSON output must be an object")
    return payload


def validate_smoke_payload(payload: dict) -> None:
    root_cause_code = payload.get("root_cause_code")
    reasoning = payload.get("reasoning")
    if root_cause_code not in CANONICAL_LABELS:
        raise RuntimeError(f"smoke output uses non-canonical root_cause_code: {root_cause_code!r}")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise RuntimeError("smoke output requires non-empty string reasoning")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-in-4bit", action="store_true", help="Use bitsandbytes 4-bit loading on a compatible CUDA GPU.")
    args = parser.parse_args()

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install torch and transformers in the intended GPU environment before running this hard gate.") from exc

    model_id = MODEL["base_model_id"]
    revision = MODEL["base_model_revision"]

    load_kwargs = {
        "revision": revision,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except Exception as exc:
            raise RuntimeError("4-bit smoke requested but bitsandbytes/quantization support is unavailable") from exc
    else:
        load_kwargs["torch_dtype"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    allowed = ", ".join(sorted(CANONICAL_LABELS))
    messages = [
        {
            "role": "user",
            "content": (
                "Classify this production incident. Return ONLY one JSON object with string keys "
                "root_cause_code and reasoning. root_cause_code MUST be one of: "
                f"{allowed}. Incident: checkout latency rose sharply and database query count increased "
                "18x after an application change; CPU and memory stayed normal."
            ),
        }
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("frozen model produced empty output")

    payload = extract_json(text)
    validate_smoke_payload(payload)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA_GPU"
    vram = None
    if torch.cuda.is_available():
        vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

    evidence = {
        "status": "PASS",
        "model_id": model_id,
        "revision": revision,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
        "vram_gib": vram,
        "load_in_4bit": args.load_in_4bit,
        "raw_output": text,
        "parsed_output": payload,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
