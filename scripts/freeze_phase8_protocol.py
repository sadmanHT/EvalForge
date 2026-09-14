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

from app.inference.gpu_host import (  # noqa: E402
    hardware_runtime_descriptor,
    load_gpu_host_contract,
    validate_gpu_host_evidence,
    verify_reference_smoke,
)
from app.inference.rag_ablation_report import validate_rag_ablation_report  # noqa: E402
from app.inference.rag_protocol import (  # noqa: E402
    RAG_PROTOCOL_PATH,
    RAGProtocol,
    RAGProtocolState,
    load_rag_protocol,
)

DEFAULT_EVIDENCE_DIR = Path("evidence/phase-08/validation")
DEFAULT_REPORT_PATH = Path("evidence/phase-08/ablation-report.json")
DEFAULT_FREEZE_PATH = Path("evidence/phase-08/protocol-freeze.json")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return {str(key): value for key, value in payload.items()}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the sealed Phase 08 validation ablation report and, only when explicitly "
            "requested, freeze the validation-selected RAG variant for one locked test."
        )
    )
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--ablation-report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--gpu-host-evidence", required=True, type=Path)
    parser.add_argument("--freeze-record", type=Path, default=DEFAULT_FREEZE_PATH)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    protocol, suite = load_rag_protocol(ROOT)
    if protocol.state is not RAGProtocolState.VALIDATION or protocol.locked_test_authorized:
        raise SystemExit("PHASE08_PROTOCOL_FREEZE=FAIL protocol must be validation/unlocked")

    evidence_dir = _resolve(args.evidence_dir)
    evidence_by_variant = {
        variant.variant_id: _load_json(evidence_dir / f"{variant.variant_id}.json")
        for variant in suite.variants
    }
    report_path = _resolve(args.ablation_report)
    report = _load_json(report_path)
    validate_rag_ablation_report(
        report,
        root=ROOT,
        protocol=protocol,
        suite=suite,
        evidence_by_variant=evidence_by_variant,
    )

    host_path = _resolve(args.gpu_host_evidence)
    host_contract = load_gpu_host_contract(ROOT)
    verify_reference_smoke(ROOT, host_contract)
    host = validate_gpu_host_evidence(_load_json(host_path), contract=host_contract)
    expected_descriptor = hardware_runtime_descriptor(host)
    for variant_id, evidence in evidence_by_variant.items():
        if evidence.get("hardware_runtime_descriptor") != expected_descriptor:
            raise ValueError(f"Phase 08 validation host binding disagrees for variant {variant_id}")

    selected_variant_id = report.get("selected_variant_id")
    if not isinstance(selected_variant_id, str) or selected_variant_id not in {
        variant.variant_id for variant in suite.variants
    }:
        raise ValueError("Phase 08 ablation report has an invalid selected variant")

    scientific_hash = protocol.scientific_config_hash()
    print("PHASE08_VALIDATION_FREEZE_READINESS=PASS")
    print(f"SCIENTIFIC_CONFIG_HASH={scientific_hash}")
    print(f"SELECTED_VARIANT_ID={selected_variant_id}")
    print(f"ABLATION_REPORT_SHA256={report['report_sha256']}")
    print(f"GPU_ENVIRONMENT_FINGERPRINT={host.environment_fingerprint_sha256}")
    if not args.apply:
        print("PROTOCOL_MUTATED=false")
        return 0

    protocol_path = ROOT / RAG_PROTOCOL_PATH
    raw_protocol = _load_json(protocol_path)
    raw_protocol["state"] = RAGProtocolState.FROZEN.value
    raw_protocol["selected_variant_id"] = selected_variant_id
    raw_protocol["locked_test_authorized"] = True
    frozen = RAGProtocol.model_validate(raw_protocol)
    if frozen.scientific_config_hash() != scientific_hash:
        raise RuntimeError("freezing Phase 08 unexpectedly changed the scientific config hash")

    protocol_path.write_text(
        json.dumps(raw_protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation_evidence = {
        variant_id: {
            "run_id": evidence["run_id"],
            "evidence_sha256": evidence["evidence_sha256"],
            "file_sha256": _file_sha256(evidence_dir / f"{variant_id}.json"),
        }
        for variant_id, evidence in sorted(evidence_by_variant.items())
    }
    freeze_record = {
        "freeze_record_version": "phase8-rag-protocol-freeze-v1",
        "protocol_version": frozen.protocol_version,
        "scientific_config_hash": scientific_hash,
        "ablation_suite_hash": frozen.ablation_suite_hash,
        "selection_policy_version": report["selection_policy_version"],
        "selected_variant_id": selected_variant_id,
        "ablation_report_sha256": report["report_sha256"],
        "ablation_report_file_sha256": _file_sha256(report_path),
        "validation_evidence": validation_evidence,
        "gpu_host_evidence_sha256": host.evidence_sha256,
        "gpu_host_evidence_file_sha256": _file_sha256(host_path),
        "gpu_environment_fingerprint_sha256": host.environment_fingerprint_sha256,
        "frozen_at": datetime.now(UTC).isoformat(),
        "locked_test_authorized": True,
    }
    freeze_path = _resolve(args.freeze_record)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(
        json.dumps(freeze_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE08_PROTOCOL_FREEZE=PASS")
    print("PROTOCOL_MUTATED=true")
    print(f"FREEZE_RECORD={freeze_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
