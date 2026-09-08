#!/usr/bin/env python3
"""Machine-checkable Phase 01 hard exit gate.

The gate validates the exact frozen model/revision, the structured-output contract,
GPU/runtime evidence, internal evidence integrity, and the preserved evidence-bundle
checksums. It accepts the current nested Kaggle evidence schema and retains read
compatibility with the earlier flat schema so historical evidence remains inspectable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.phase1 import canonical_label_ids, load_json_yaml, validate_model_config, validate_study_config


class GateError(ValueError):
    """Raised when preserved Phase 01 evidence violates a hard-gate invariant."""


def fail(message: str) -> None:
    raise SystemExit(f"PHASE01_EXIT_GATE=FAIL: {message}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"{field} is missing/blank")
    return value


def _canonical_evidence_hash(evidence: dict[str, Any]) -> str:
    unhashed = dict(evidence)
    unhashed.pop("evidence_sha256", None)
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(canonical.encode("utf-8"))


def _validate_bundle_checksums(evidence_dir: Path) -> None:
    sums_path = evidence_dir / "SHA256SUMS.txt"
    if not sums_path.is_file():
        raise GateError("missing SHA256SUMS.txt for model smoke evidence bundle")

    expected_files = {"model-smoke.json", "model-smoke.txt", "environment.txt"}
    declared: dict[str, str] = {}
    for raw_line in sums_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            digest, filename = line.split(maxsplit=1)
        except ValueError as exc:
            raise GateError(f"invalid SHA256SUMS.txt line: {raw_line!r}") from exc
        filename = filename.strip()
        if filename.startswith("*"):
            filename = filename[1:]
        if Path(filename).name != filename or filename in declared:
            raise GateError(f"invalid/duplicate checksum filename: {filename!r}")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise GateError(f"invalid SHA-256 digest for {filename}")
        declared[filename] = digest.lower()

    missing = expected_files - set(declared)
    if missing:
        raise GateError(f"SHA256SUMS.txt missing required files: {sorted(missing)}")

    for filename in expected_files:
        path = evidence_dir / filename
        if not path.is_file():
            raise GateError(f"missing evidence bundle file: {filename}")
        if _sha256_file(path) != declared[filename]:
            raise GateError(f"checksum mismatch for {filename}")


def validate_smoke_evidence(
    evidence: dict[str, Any],
    model: dict[str, Any],
    taxonomy: dict[str, Any],
    evidence_dir: Path,
) -> tuple[str, float, str, str]:
    """Validate preserved smoke evidence and return gpu, vram, label, load mode."""
    if evidence.get("status") != "PASS":
        raise GateError("model smoke evidence does not report PASS")
    if evidence.get("smoke_passed") is False:
        raise GateError("model smoke evidence explicitly reports smoke_passed=false")

    nested_model = evidence.get("model") if isinstance(evidence.get("model"), dict) else None
    if nested_model is not None:
        model_id = nested_model.get("id")
        requested_revision = nested_model.get("requested_revision")
        resolved_revision = nested_model.get("resolved_revision")
        loaded_revision = nested_model.get("loaded_config_commit")
        load_mode = nested_model.get("load_mode")
    else:
        # Historical flat-schema compatibility.
        model_id = evidence.get("model_id")
        requested_revision = evidence.get("revision")
        resolved_revision = evidence.get("revision")
        loaded_revision = evidence.get("revision")
        load_mode = evidence.get("load_mode", "unspecified_historical")

    if model_id != model["base_model_id"]:
        raise GateError("model smoke evidence uses a different base_model_id")
    frozen_revision = model["base_model_revision"]
    for field, revision in (
        ("requested_revision", requested_revision),
        ("resolved_revision", resolved_revision),
        ("loaded_config_commit", loaded_revision),
    ):
        if revision != frozen_revision:
            raise GateError(f"model smoke evidence {field} does not match the frozen revision")
    load_mode = _require_nonempty_string(load_mode, "model.load_mode")

    nested_hardware = evidence.get("hardware") if isinstance(evidence.get("hardware"), dict) else None
    if nested_hardware is not None:
        gpu = nested_hardware.get("gpu_name")
        vram_gib = nested_hardware.get("gpu_vram_gib")
        cuda_version = nested_hardware.get("cuda_version")
        python_version = nested_hardware.get("python")
    else:
        if evidence.get("cuda_available") is not True:
            raise GateError("smoke was not executed in the intended CUDA GPU environment")
        gpu = evidence.get("gpu")
        vram_gib = evidence.get("vram_gib")
        cuda_version = evidence.get("cuda_version", "historical_cuda")
        python_version = evidence.get("python", "historical_python")

    gpu = _require_nonempty_string(gpu, "hardware.gpu_name")
    if gpu == "NO_CUDA_GPU":
        raise GateError("GPU model is missing from smoke evidence")
    if not isinstance(vram_gib, (int, float)) or isinstance(vram_gib, bool) or vram_gib <= 0:
        raise GateError("GPU VRAM is missing/invalid in smoke evidence")
    _require_nonempty_string(cuda_version, "hardware.cuda_version")
    _require_nonempty_string(python_version, "hardware.python")

    packages = evidence.get("packages")
    if nested_model is not None:
        if not isinstance(packages, dict):
            raise GateError("packages section is missing from current smoke evidence")
        _require_nonempty_string(packages.get("torch"), "packages.torch")
        _require_nonempty_string(packages.get("transformers"), "packages.transformers")

    payload = evidence.get("parsed_output")
    if not isinstance(payload, dict):
        raise GateError("parsed_output is missing from model smoke evidence")
    if set(payload) != {"root_cause_code", "reasoning"}:
        raise GateError("parsed_output must contain exactly root_cause_code and reasoning")
    code = payload.get("root_cause_code")
    canonical_codes = canonical_label_ids(taxonomy)
    if code not in canonical_codes:
        raise GateError(f"smoke root_cause_code is non-canonical: {code!r}")
    if not isinstance(payload.get("reasoning"), str) or not payload["reasoning"].strip():
        raise GateError("smoke reasoning is missing/blank")

    raw_output = evidence.get("raw_output")
    if nested_model is not None:
        raw_output = _require_nonempty_string(raw_output, "raw_output")
        try:
            raw_payload = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise GateError("raw_output is not strict JSON") from exc
        if raw_payload != payload:
            raise GateError("raw_output JSON does not exactly match parsed_output")

        declared_codes = evidence.get("canonical_root_cause_codes")
        if not isinstance(declared_codes, list) or set(declared_codes) != canonical_codes:
            raise GateError("smoke evidence canonical_root_cause_codes does not match taxonomy")

        claimed_evidence_hash = evidence.get("evidence_sha256")
        if not isinstance(claimed_evidence_hash, str) or claimed_evidence_hash != _canonical_evidence_hash(evidence):
            raise GateError("model-smoke.json internal evidence_sha256 is missing or invalid")

        if evidence.get("gate") != "exact_frozen_model_structured_output_smoke":
            raise GateError("unexpected smoke gate identifier")
        if evidence.get("repository") != "sadmanHT/EvalForge":
            raise GateError("smoke evidence repository identity mismatch")

        _validate_bundle_checksums(evidence_dir)

    return gpu, float(vram_gib), str(code), load_mode


def main() -> int:
    model = load_json_yaml(ROOT / "configs/model.yaml")
    study = load_json_yaml(ROOT / "configs/study.yaml")
    taxonomy = load_json_yaml(ROOT / "configs/label-taxonomy.yaml")
    validate_model_config(model)
    validate_study_config(study, model, taxonomy)

    evidence_dir = ROOT / "evidence/phase-01"
    evidence_path = evidence_dir / "model-smoke.json"
    if not evidence_path.is_file():
        fail("missing real model smoke evidence at evidence/phase-01/model-smoke.json")

    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"model-smoke.json is not valid JSON: {exc}")
    if not isinstance(evidence, dict):
        fail("model-smoke.json must contain an object")

    try:
        gpu, vram_gib, code, load_mode = validate_smoke_evidence(evidence, model, taxonomy, evidence_dir)
    except GateError as exc:
        fail(str(exc))

    print("PHASE01_EXIT_GATE=PASS")
    print(f"MODEL={model['base_model_id']}@{model['base_model_revision']}")
    print(f"GPU={gpu} ({vram_gib} GiB)")
    print(f"LOAD_MODE={load_mode}")
    print(f"SMOKE_LABEL={code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
