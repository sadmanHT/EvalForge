from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.inference.gpu_host import (
    GPUHostEvidence,
    hardware_runtime_descriptor,
    load_gpu_host_contract,
    validate_gpu_host_evidence,
    verify_reference_smoke,
)
from app.inference.protocol import load_baseline_protocol
from app.inference.rag_ablation_report import validate_rag_ablation_report
from app.inference.rag_comparison import validate_paired_baseline_rag_comparison
from app.inference.rag_evidence import validate_phase8_rag_run_evidence
from app.inference.rag_protocol import RAGProtocol, RAGProtocolState, load_rag_protocol

PHASE8_EVIDENCE_DIR = Path("evidence/phase-08")
VALIDATION_DIR = PHASE8_EVIDENCE_DIR / "validation"
VALIDATION_GPU_HOST_PATH = PHASE8_EVIDENCE_DIR / "validation-gpu-host.json"
ABLATION_REPORT_PATH = PHASE8_EVIDENCE_DIR / "ablation-report.json"
VALIDATION_COMPARISON_PATH = PHASE8_EVIDENCE_DIR / "validation-comparison.json"
FREEZE_RECORD_PATH = PHASE8_EVIDENCE_DIR / "protocol-freeze.json"
TEST_RUN_PATH = PHASE8_EVIDENCE_DIR / "test-run.json"
TEST_GPU_HOST_PATH = PHASE8_EVIDENCE_DIR / "test-gpu-host.json"
TEST_COMPARISON_PATH = PHASE8_EVIDENCE_DIR / "baseline-vs-rag-test-comparison.json"


class Phase8Stage(StrEnum):
    PRE_VALIDATION = "pre_validation"
    VALIDATION_READY_TO_FREEZE = "validation_ready_to_freeze"
    FROZEN_WAITING_TEST = "frozen_waiting_test"
    COMPLETE = "complete"


@dataclass(frozen=True)
class Phase8RepositoryStatus:
    stage: Phase8Stage
    blocker: str | None
    scientific_config_hash: str
    selected_variant_id: str | None = None
    ablation_report_sha256: str | None = None
    test_run_id: str | None = None
    test_evidence_sha256: str | None = None
    comparison_sha256: str | None = None
    gpu_environment_fingerprint_sha256: str | None = None

    @property
    def complete(self) -> bool:
        return self.stage is Phase8Stage.COMPLETE


def _path(root: Path, relative: Path) -> Path:
    return root / relative


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Phase 08 JSON artifact must contain an object: {path}")
    return {str(key): value for key, value in payload.items()}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_host(root: Path, relative: Path) -> GPUHostEvidence:
    contract = load_gpu_host_contract(root)
    verify_reference_smoke(root, contract)
    return validate_gpu_host_evidence(
        _load_json(_path(root, relative)),
        contract=contract,
    )


def _require_tracking(evidence: dict[str, Any], *, context: str) -> None:
    tracking = evidence.get("tracking")
    if not isinstance(tracking, dict) or tracking.get("provider") != "wandb":
        raise ValueError(f"{context} has an invalid W&B tracking payload")
    if tracking.get("configured") is not True:
        raise ValueError(f"{context} requires configured W&B tracking")
    for field in ("run_reference", "artifact_reference"):
        value = tracking.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context} W&B tracking is missing {field}")


def _validation_bundle(
    root: Path,
    protocol: RAGProtocol,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], GPUHostEvidence, dict[str, Any]]:
    _loaded_protocol, suite = load_rag_protocol(root)
    host = _load_host(root, VALIDATION_GPU_HOST_PATH)
    descriptor = hardware_runtime_descriptor(host)
    evidence_by_variant: dict[str, dict[str, Any]] = {}
    for variant in suite.variants:
        path = _path(root, VALIDATION_DIR / f"{variant.variant_id}.json")
        evidence = _load_json(path)
        validate_phase8_rag_run_evidence(
            evidence,
            root=root,
            protocol=protocol,
            expected_split="validation",
            expected_variant=variant,
        )
        if evidence.get("protocol_state_at_export") != RAGProtocolState.VALIDATION.value:
            raise ValueError("Phase 08 validation evidence was not exported before freeze")
        if evidence.get("locked_test_authorized_at_export") is not False:
            raise ValueError("Phase 08 validation evidence was exported after test authorization")
        if evidence.get("hardware_runtime_descriptor") != descriptor:
            raise ValueError("Phase 08 validation evidence is not bound to the validation GPU host")
        _require_tracking(evidence, context=f"Phase 08 validation variant {variant.variant_id}")
        evidence_by_variant[variant.variant_id] = evidence

    report = _load_json(_path(root, ABLATION_REPORT_PATH))
    validate_rag_ablation_report(
        report,
        root=root,
        protocol=protocol,
        suite=suite,
        evidence_by_variant=evidence_by_variant,
    )
    selected_variant_id = report.get("selected_variant_id")
    if not isinstance(selected_variant_id, str):
        raise ValueError("Phase 08 ablation report is missing selected_variant_id")
    selected_variant = next(
        variant for variant in suite.variants if variant.variant_id == selected_variant_id
    )

    baseline_protocol = load_baseline_protocol(root)
    baseline_validation = _load_json(root / "evidence/phase-06/validation-run.json")
    validation_comparison = _load_json(_path(root, VALIDATION_COMPARISON_PATH))
    validate_paired_baseline_rag_comparison(
        validation_comparison,
        root=root,
        split="validation",
        baseline_protocol=baseline_protocol,
        rag_protocol=protocol,
        rag_variant=selected_variant,
        baseline_evidence=baseline_validation,
        rag_evidence=evidence_by_variant[selected_variant_id],
    )
    return evidence_by_variant, report, host, validation_comparison


def _validate_freeze_record(
    root: Path,
    *,
    protocol: RAGProtocol,
    validation_evidence: dict[str, dict[str, Any]],
    report: dict[str, Any],
    host: GPUHostEvidence,
) -> dict[str, Any]:
    freeze = _load_json(_path(root, FREEZE_RECORD_PATH))
    expected_validation = {
        variant_id: {
            "run_id": evidence["run_id"],
            "evidence_sha256": evidence["evidence_sha256"],
            "file_sha256": _file_sha256(_path(root, VALIDATION_DIR / f"{variant_id}.json")),
        }
        for variant_id, evidence in sorted(validation_evidence.items())
    }
    expected = {
        "freeze_record_version": "phase8-rag-protocol-freeze-v1",
        "protocol_version": protocol.protocol_version,
        "scientific_config_hash": protocol.scientific_config_hash(),
        "ablation_suite_hash": protocol.ablation_suite_hash,
        "selection_policy_version": report["selection_policy_version"],
        "selected_variant_id": report["selected_variant_id"],
        "ablation_report_sha256": report["report_sha256"],
        "ablation_report_file_sha256": _file_sha256(_path(root, ABLATION_REPORT_PATH)),
        "validation_evidence": expected_validation,
        "gpu_host_evidence_sha256": host.evidence_sha256,
        "gpu_host_evidence_file_sha256": _file_sha256(_path(root, VALIDATION_GPU_HOST_PATH)),
        "gpu_environment_fingerprint_sha256": host.environment_fingerprint_sha256,
        "locked_test_authorized": True,
    }
    for key, value in expected.items():
        if freeze.get(key) != value:
            raise ValueError(f"Phase 08 protocol-freeze record mismatch: {key}")
    frozen_at = freeze.get("frozen_at")
    if not isinstance(frozen_at, str) or not frozen_at.strip():
        raise ValueError("Phase 08 protocol-freeze record is missing frozen_at")
    return freeze


def _test_bundle(
    root: Path,
    *,
    protocol: RAGProtocol,
    freeze: dict[str, Any],
) -> tuple[dict[str, Any], GPUHostEvidence, dict[str, Any]]:
    _loaded_protocol, suite = load_rag_protocol(root)
    if protocol.selected_variant_id is None:
        raise ValueError("frozen Phase 08 protocol is missing selected_variant_id")
    variant = next(
        item for item in suite.variants if item.variant_id == protocol.selected_variant_id
    )
    evidence = _load_json(_path(root, TEST_RUN_PATH))
    validate_phase8_rag_run_evidence(
        evidence,
        root=root,
        protocol=protocol,
        expected_split="test",
        expected_variant=variant,
    )
    if evidence.get("protocol_state_at_export") != RAGProtocolState.FROZEN.value:
        raise ValueError("Phase 08 test evidence was not exported under the frozen protocol")
    if evidence.get("locked_test_authorized_at_export") is not True:
        raise ValueError("Phase 08 test evidence was exported without authorization")
    _require_tracking(evidence, context="Phase 08 locked test")

    host = _load_host(root, TEST_GPU_HOST_PATH)
    if evidence.get("hardware_runtime_descriptor") != hardware_runtime_descriptor(host):
        raise ValueError("Phase 08 locked-test evidence is not bound to its GPU host")
    if host.environment_fingerprint_sha256 != freeze.get("gpu_environment_fingerprint_sha256"):
        raise ValueError("Phase 08 locked-test GPU/software fingerprint differs from freeze")

    baseline_protocol = load_baseline_protocol(root)
    baseline_test = _load_json(root / "evidence/phase-06/test-run.json")
    comparison = _load_json(_path(root, TEST_COMPARISON_PATH))
    validate_paired_baseline_rag_comparison(
        comparison,
        root=root,
        split="test",
        baseline_protocol=baseline_protocol,
        rag_protocol=protocol,
        rag_variant=variant,
        baseline_evidence=baseline_test,
        rag_evidence=evidence,
    )
    return evidence, host, comparison


def evaluate_phase8_repository_state(root: Path) -> Phase8RepositoryStatus:
    protocol, suite = load_rag_protocol(root)
    scientific_hash = protocol.scientific_config_hash()
    validation_paths = [VALIDATION_DIR / f"{variant.variant_id}.json" for variant in suite.variants]
    validation_paths.extend(
        [VALIDATION_GPU_HOST_PATH, ABLATION_REPORT_PATH, VALIDATION_COMPARISON_PATH]
    )
    validation_present = [_path(root, path).is_file() for path in validation_paths]
    freeze_present = _path(root, FREEZE_RECORD_PATH).is_file()
    test_paths = [TEST_RUN_PATH, TEST_GPU_HOST_PATH, TEST_COMPARISON_PATH]
    test_present = [_path(root, path).is_file() for path in test_paths]

    if protocol.state is RAGProtocolState.VALIDATION:
        if protocol.locked_test_authorized or protocol.selected_variant_id is not None:
            raise ValueError("validation Phase 08 protocol may not select or authorize test")
        if freeze_present or any(test_present):
            raise ValueError("Phase 08 freeze/test evidence exists before protocol freeze")
        if not any(validation_present):
            return Phase8RepositoryStatus(
                stage=Phase8Stage.PRE_VALIDATION,
                blocker="real RAG validation ablations and selection evidence are pending",
                scientific_config_hash=scientific_hash,
            )
        if not all(validation_present):
            raise ValueError("Phase 08 validation evidence bundle is incomplete")
        _evidence, report, host, comparison = _validation_bundle(root, protocol)
        return Phase8RepositoryStatus(
            stage=Phase8Stage.VALIDATION_READY_TO_FREEZE,
            blocker="validation selection is sealed; protocol freeze is pending",
            scientific_config_hash=scientific_hash,
            selected_variant_id=str(report["selected_variant_id"]),
            ablation_report_sha256=str(report["report_sha256"]),
            comparison_sha256=str(comparison["comparison_sha256"]),
            gpu_environment_fingerprint_sha256=host.environment_fingerprint_sha256,
        )

    if protocol.state is not RAGProtocolState.FROZEN or not protocol.locked_test_authorized:
        raise ValueError("frozen Phase 08 protocol must explicitly authorize the locked test")
    if not all(validation_present):
        raise ValueError("frozen Phase 08 protocol is missing its validation evidence bundle")
    if not freeze_present:
        raise ValueError("frozen Phase 08 protocol is missing protocol-freeze evidence")

    validation, report, validation_host, validation_comparison = _validation_bundle(root, protocol)
    if protocol.selected_variant_id != report.get("selected_variant_id"):
        raise ValueError("frozen Phase 08 selected variant disagrees with validation report")
    freeze = _validate_freeze_record(
        root,
        protocol=protocol,
        validation_evidence=validation,
        report=report,
        host=validation_host,
    )
    if not any(test_present):
        return Phase8RepositoryStatus(
            stage=Phase8Stage.FROZEN_WAITING_TEST,
            blocker="single authorized real RAG locked test is pending",
            scientific_config_hash=scientific_hash,
            selected_variant_id=protocol.selected_variant_id,
            ablation_report_sha256=str(report["report_sha256"]),
            comparison_sha256=str(validation_comparison["comparison_sha256"]),
            gpu_environment_fingerprint_sha256=(validation_host.environment_fingerprint_sha256),
        )
    if not all(test_present):
        raise ValueError("Phase 08 locked-test evidence bundle is incomplete")

    test, test_host, comparison = _test_bundle(root, protocol=protocol, freeze=freeze)
    return Phase8RepositoryStatus(
        stage=Phase8Stage.COMPLETE,
        blocker=None,
        scientific_config_hash=scientific_hash,
        selected_variant_id=protocol.selected_variant_id,
        ablation_report_sha256=str(report["report_sha256"]),
        test_run_id=str(test["run_id"]),
        test_evidence_sha256=str(test["evidence_sha256"]),
        comparison_sha256=str(comparison["comparison_sha256"]),
        gpu_environment_fingerprint_sha256=test_host.environment_fingerprint_sha256,
    )
