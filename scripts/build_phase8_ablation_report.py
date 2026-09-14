#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
from app.inference.rag_ablation_report import (  # noqa: E402
    build_rag_ablation_report,
    validate_rag_ablation_report,
)
from app.inference.rag_protocol import RAGProtocolState, load_rag_protocol  # noqa: E402

DEFAULT_EVIDENCE_DIR = Path("evidence/phase-08/validation")
DEFAULT_REPORT_PATH = Path("evidence/phase-08/ablation-report.json")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return {str(key): value for key, value in payload.items()}


def _require_tracking(evidence: dict[str, Any]) -> None:
    tracking = evidence.get("tracking")
    if not isinstance(tracking, dict) or tracking.get("provider") != "wandb":
        raise ValueError("Phase 08 validation evidence has an invalid W&B tracking payload")
    if tracking.get("configured") is not True:
        raise ValueError("Phase 08 validation evidence requires configured W&B tracking")
    for field in ("run_reference", "artifact_reference"):
        value = tracking.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Phase 08 validation W&B tracking is missing {field}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate every registered Phase 08 validation run and build the deterministic "
            "validation-only ablation/selection report."
        )
    )
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--gpu-host-evidence", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    protocol, suite = load_rag_protocol(ROOT)
    if protocol.state is not RAGProtocolState.VALIDATION or protocol.locked_test_authorized:
        raise SystemExit("PHASE08_ABLATION_REPORT=FAIL protocol must be validation/unlocked")

    host_contract = load_gpu_host_contract(ROOT)
    verify_reference_smoke(ROOT, host_contract)
    host_evidence = validate_gpu_host_evidence(
        _load_json(_resolve(args.gpu_host_evidence)),
        contract=host_contract,
    )
    expected_descriptor = hardware_runtime_descriptor(host_evidence)

    evidence_dir = _resolve(args.evidence_dir)
    evidence_by_variant: dict[str, dict[str, Any]] = {}
    for variant in suite.variants:
        path = evidence_dir / f"{variant.variant_id}.json"
        if not path.is_file():
            raise ValueError(f"missing Phase 08 validation evidence: {path}")
        evidence = _load_json(path)
        if evidence.get("protocol_state_at_export") != RAGProtocolState.VALIDATION.value:
            raise ValueError("Phase 08 validation evidence was exported after protocol freeze")
        if evidence.get("locked_test_authorized_at_export") is not False:
            raise ValueError("Phase 08 validation evidence was exported after test authorization")
        if evidence.get("hardware_runtime_descriptor") != expected_descriptor:
            raise ValueError("Phase 08 validation run is not bound to the supplied GPU host")
        _require_tracking(evidence)
        evidence_by_variant[variant.variant_id] = evidence

    report = build_rag_ablation_report(
        root=ROOT,
        protocol=protocol,
        suite=suite,
        evidence_by_variant=evidence_by_variant,
    )
    validate_rag_ablation_report(
        report,
        root=ROOT,
        protocol=protocol,
        suite=suite,
        evidence_by_variant=evidence_by_variant,
    )

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PHASE08_ABLATION_REPORT=PASS")
    print(f"SELECTED_VARIANT_ID={report['selected_variant_id']}")
    print(f"REPORT_SHA256={report['report_sha256']}")
    print(f"GPU_ENVIRONMENT_FINGERPRINT={host_evidence.environment_fingerprint_sha256}")
    print(f"REPORT={output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
