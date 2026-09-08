#!/usr/bin/env python3
"""Machine-checkable Phase 01 hard exit gate.

Run only after the Phase 01 test suite and static validator are green.
This script intentionally fails until real GPU smoke evidence exists.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.phase1 import canonical_label_ids, load_json_yaml, validate_model_config, validate_study_config


def fail(message: str) -> None:
    raise SystemExit(f"PHASE01_EXIT_GATE=FAIL: {message}")


def main() -> int:
    model = load_json_yaml(ROOT / "configs/model.yaml")
    study = load_json_yaml(ROOT / "configs/study.yaml")
    taxonomy = load_json_yaml(ROOT / "configs/label-taxonomy.yaml")
    validate_model_config(model)
    validate_study_config(study, model, taxonomy)

    evidence_path = ROOT / "evidence/phase-01/model-smoke.json"
    if not evidence_path.is_file():
        fail("missing real model smoke evidence at evidence/phase-01/model-smoke.json")

    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"model-smoke.json is not valid JSON: {exc}")

    if evidence.get("status") != "PASS":
        fail("model smoke evidence does not report PASS")
    if evidence.get("model_id") != model["base_model_id"]:
        fail("model smoke evidence uses a different base_model_id")
    if evidence.get("revision") != model["base_model_revision"]:
        fail("model smoke evidence uses a different frozen revision")
    if evidence.get("cuda_available") is not True:
        fail("smoke was not executed in the intended CUDA GPU environment")
    if not isinstance(evidence.get("gpu"), str) or not evidence["gpu"].strip() or evidence["gpu"] == "NO_CUDA_GPU":
        fail("GPU model is missing from smoke evidence")
    if not isinstance(evidence.get("vram_gib"), (int, float)) or evidence["vram_gib"] <= 0:
        fail("GPU VRAM is missing/invalid in smoke evidence")

    payload = evidence.get("parsed_output")
    if not isinstance(payload, dict):
        fail("parsed_output is missing from model smoke evidence")
    code = payload.get("root_cause_code")
    if code not in canonical_label_ids(taxonomy):
        fail(f"smoke root_cause_code is non-canonical: {code!r}")
    if not isinstance(payload.get("reasoning"), str) or not payload["reasoning"].strip():
        fail("smoke reasoning is missing/blank")

    print("PHASE01_EXIT_GATE=PASS")
    print(f"MODEL={model['base_model_id']}@{model['base_model_revision']}")
    print(f"GPU={evidence['gpu']} ({evidence['vram_gib']} GiB)")
    print(f"SMOKE_LABEL={code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
