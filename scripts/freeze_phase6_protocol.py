#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.inference.evidence import (  # noqa: E402
    ValidationReview,
    validate_run_evidence,
    validate_validation_review,
)
from app.inference.gpu_host import (  # noqa: E402
    hardware_runtime_descriptor,
    load_gpu_host_contract,
    validate_gpu_host_evidence,
    verify_reference_smoke,
)
from app.inference.protocol import (  # noqa: E402
    BASELINE_PROTOCOL_PATH,
    BaselineProtocol,
    ProtocolState,
    load_baseline_protocol,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify real Phase 06 validation evidence and, only when explicitly requested, "
            "freeze/authorize the locked-test protocol."
        )
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--gpu-host-evidence", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--freeze-record",
        type=Path,
        default=Path("evidence/phase-06/protocol-freeze.json"),
    )
    args = parser.parse_args()

    evidence_path = _resolve(args.evidence)
    review_path = _resolve(args.review)
    host_evidence_path = _resolve(args.gpu_host_evidence)
    protocol_path = ROOT / BASELINE_PROTOCOL_PATH
    freeze_record_path = _resolve(args.freeze_record)

    protocol = load_baseline_protocol(ROOT)
    if protocol.state is not ProtocolState.VALIDATION or protocol.locked_test_authorized:
        raise SystemExit("PHASE06_PROTOCOL_FREEZE=FAIL protocol must still be validation/unlocked")
    evidence = _load_json(evidence_path)
    review = ValidationReview.model_validate(_load_json(review_path))
    validate_run_evidence(
        evidence,
        root=ROOT,
        protocol=protocol,
        expected_split="validation",
    )
    validate_validation_review(review, validation_evidence=evidence)

    host_contract = load_gpu_host_contract(ROOT)
    verify_reference_smoke(ROOT, host_contract)
    host_evidence = validate_gpu_host_evidence(
        _load_json(host_evidence_path),
        contract=host_contract,
    )
    expected_descriptor = hardware_runtime_descriptor(host_evidence)
    if evidence.get("hardware_runtime_descriptor") != expected_descriptor:
        raise ValueError("validation run evidence is not bound to the supplied GPU host evidence")

    scientific_hash = protocol.scientific_config_hash()
    print("PHASE06_VALIDATION_FREEZE_READINESS=PASS")
    print(f"VALIDATION_RUN_ID={evidence['run_id']}")
    print(f"SCIENTIFIC_CONFIG_HASH={scientific_hash}")
    print(f"VALIDATION_EVIDENCE_SHA256={evidence['evidence_sha256']}")
    print(f"GPU_ENVIRONMENT_FINGERPRINT={host_evidence.environment_fingerprint_sha256}")
    print(f"MANUAL_REVIEW_SHA256={_file_sha256(review_path)}")
    if not args.apply:
        print("PROTOCOL_MUTATED=false")
        return 0

    raw_protocol = _load_json(protocol_path)
    raw_protocol["state"] = ProtocolState.FROZEN.value
    raw_protocol["locked_test_authorized"] = True
    frozen = BaselineProtocol.model_validate(raw_protocol)
    if frozen.scientific_config_hash() != scientific_hash:
        raise RuntimeError("freezing Phase 06 unexpectedly changed the scientific config hash")

    protocol_path.write_text(
        json.dumps(raw_protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze_record = {
        "freeze_record_version": "phase6-protocol-freeze-v1",
        "protocol_version": frozen.protocol_version,
        "scientific_config_hash": scientific_hash,
        "validation_run_id": evidence["run_id"],
        "validation_evidence_sha256": evidence["evidence_sha256"],
        "validation_evidence_file_sha256": _file_sha256(evidence_path),
        "gpu_host_evidence_sha256": host_evidence.evidence_sha256,
        "gpu_host_evidence_file_sha256": _file_sha256(host_evidence_path),
        "gpu_environment_fingerprint_sha256": (
            host_evidence.environment_fingerprint_sha256
        ),
        "manual_review_file_sha256": _file_sha256(review_path),
        "manual_reviewed_at": review.reviewed_at.isoformat(),
        "manual_reviewer": review.reviewer,
        "frozen_at": datetime.now(UTC).isoformat(),
        "locked_test_authorized": True,
    }
    freeze_record_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_record_path.write_text(
        json.dumps(freeze_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE06_PROTOCOL_FREEZE=PASS")
    print("PROTOCOL_MUTATED=true")
    print(f"FREEZE_RECORD={freeze_record_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
