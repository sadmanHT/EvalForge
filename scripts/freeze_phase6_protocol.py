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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify real Phase 06 validation evidence and, only when explicitly requested, "
            "freeze/authorize the locked-test protocol."
        )
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--freeze-record",
        type=Path,
        default=Path("evidence/phase-06/protocol-freeze.json"),
    )
    args = parser.parse_args()

    evidence_path = args.evidence if args.evidence.is_absolute() else ROOT / args.evidence
    review_path = args.review if args.review.is_absolute() else ROOT / args.review
    protocol_path = ROOT / BASELINE_PROTOCOL_PATH
    freeze_record_path = (
        args.freeze_record
        if args.freeze_record.is_absolute()
        else ROOT / args.freeze_record
    )

    protocol = load_baseline_protocol(ROOT)
    if protocol.state is not ProtocolState.VALIDATION or protocol.locked_test_authorized:
        raise SystemExit(
            "PHASE06_PROTOCOL_FREEZE=FAIL protocol must still be validation/unlocked"
        )
    evidence = _load_json(evidence_path)
    review = ValidationReview.model_validate(_load_json(review_path))
    validate_run_evidence(
        evidence,
        root=ROOT,
        protocol=protocol,
        expected_split="validation",
    )
    validate_validation_review(review, validation_evidence=evidence)

    scientific_hash = protocol.scientific_config_hash()
    print("PHASE06_VALIDATION_FREEZE_READINESS=PASS")
    print(f"VALIDATION_RUN_ID={evidence['run_id']}")
    print(f"SCIENTIFIC_CONFIG_HASH={scientific_hash}")
    print(f"VALIDATION_EVIDENCE_SHA256={evidence['evidence_sha256']}")
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
