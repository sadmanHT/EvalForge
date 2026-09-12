from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.inference.evidence import (
    ValidationReview,
    validate_real_run_operational_evidence,
    validate_run_evidence,
    validate_validation_review,
)
from app.inference.gpu_host import (
    GPUHostEvidence,
    hardware_runtime_descriptor,
    load_gpu_host_contract,
    validate_gpu_host_evidence,
    verify_reference_smoke,
)
from app.inference.protocol import BaselineProtocol, ProtocolState, load_baseline_protocol

PHASE6_EVIDENCE_DIR = Path("evidence/phase-06")
VALIDATION_RUN_PATH = PHASE6_EVIDENCE_DIR / "validation-run.json"
VALIDATION_REVIEW_PATH = PHASE6_EVIDENCE_DIR / "validation-review.json"
VALIDATION_GPU_HOST_PATH = PHASE6_EVIDENCE_DIR / "validation-gpu-host.json"
FREEZE_RECORD_PATH = PHASE6_EVIDENCE_DIR / "protocol-freeze.json"
TEST_RUN_PATH = PHASE6_EVIDENCE_DIR / "test-run.json"
TEST_GPU_HOST_PATH = PHASE6_EVIDENCE_DIR / "test-gpu-host.json"


class Phase6Stage(StrEnum):
    PRE_VALIDATION = "pre_validation"
    VALIDATION_READY_TO_FREEZE = "validation_ready_to_freeze"
    FROZEN_WAITING_TEST = "frozen_waiting_test"
    COMPLETE = "complete"


@dataclass(frozen=True)
class Phase6RepositoryStatus:
    stage: Phase6Stage
    blocker: str | None
    scientific_config_hash: str
    validation_run_id: str | None = None
    test_run_id: str | None = None
    validation_evidence_sha256: str | None = None
    test_evidence_sha256: str | None = None
    gpu_environment_fingerprint_sha256: str | None = None
    test_metrics: dict[str, float] | None = None
    tracking_run_reference: str | None = None
    tracking_artifact_reference: str | None = None

    @property
    def complete(self) -> bool:
        return self.stage is Phase6Stage.COMPLETE


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Phase 06 JSON artifact must contain an object: {path}")
    return {str(key): value for key, value in payload.items()}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(root: Path, relative: Path) -> Path:
    return root / relative


def _present(root: Path, relative: Path) -> bool:
    return _path(root, relative).is_file()


def _load_host_evidence(root: Path, relative: Path) -> GPUHostEvidence:
    contract = load_gpu_host_contract(root)
    verify_reference_smoke(root, contract)
    return validate_gpu_host_evidence(
        _load_json_object(_path(root, relative)),
        contract=contract,
    )


def _validate_run_host_binding(
    run_evidence: dict[str, Any],
    host_evidence: GPUHostEvidence,
) -> None:
    expected_descriptor = hardware_runtime_descriptor(host_evidence)
    if run_evidence.get("hardware_runtime_descriptor") != expected_descriptor:
        raise ValueError("Phase 06 run evidence is not bound to its sealed GPU host evidence")


def _validation_bundle(
    root: Path,
    protocol: BaselineProtocol,
) -> tuple[dict[str, Any], ValidationReview, GPUHostEvidence]:
    evidence = _load_json_object(_path(root, VALIDATION_RUN_PATH))
    validate_run_evidence(
        evidence,
        root=root,
        protocol=protocol,
        expected_split="validation",
    )
    validate_real_run_operational_evidence(evidence, require_tracking=False)
    if evidence.get("protocol_state_at_export") != ProtocolState.VALIDATION.value:
        raise ValueError(
            "validation evidence was not exported while the protocol was in validation"
        )
    if evidence.get("locked_test_authorized_at_export") is not False:
        raise ValueError("validation evidence was exported after locked-test authorization")

    review = ValidationReview.model_validate(_load_json_object(_path(root, VALIDATION_REVIEW_PATH)))
    validate_validation_review(review, validation_evidence=evidence)
    host = _load_host_evidence(root, VALIDATION_GPU_HOST_PATH)
    _validate_run_host_binding(evidence, host)
    return evidence, review, host


def _validate_freeze_record(
    root: Path,
    *,
    protocol: BaselineProtocol,
    validation_evidence: dict[str, Any],
    review: ValidationReview,
    validation_host: GPUHostEvidence,
) -> dict[str, Any]:
    path = _path(root, FREEZE_RECORD_PATH)
    freeze = _load_json_object(path)
    expected = {
        "freeze_record_version": "phase6-protocol-freeze-v1",
        "protocol_version": protocol.protocol_version,
        "scientific_config_hash": protocol.scientific_config_hash(),
        "validation_run_id": validation_evidence.get("run_id"),
        "validation_evidence_sha256": validation_evidence.get("evidence_sha256"),
        "validation_evidence_file_sha256": _file_sha256(_path(root, VALIDATION_RUN_PATH)),
        "gpu_host_evidence_sha256": validation_host.evidence_sha256,
        "gpu_host_evidence_file_sha256": _file_sha256(_path(root, VALIDATION_GPU_HOST_PATH)),
        "gpu_environment_fingerprint_sha256": validation_host.environment_fingerprint_sha256,
        "manual_review_file_sha256": _file_sha256(_path(root, VALIDATION_REVIEW_PATH)),
        "manual_reviewed_at": review.reviewed_at.isoformat(),
        "manual_reviewer": review.reviewer,
        "locked_test_authorized": True,
    }
    for key, value in expected.items():
        if freeze.get(key) != value:
            raise ValueError(f"Phase 06 protocol-freeze record mismatch: {key}")
    frozen_at = freeze.get("frozen_at")
    if not isinstance(frozen_at, str) or not frozen_at.strip():
        raise ValueError("Phase 06 protocol-freeze record is missing frozen_at")
    return freeze


def _test_bundle(
    root: Path,
    protocol: BaselineProtocol,
    *,
    freeze: dict[str, Any],
    validation_evidence: dict[str, Any],
) -> tuple[dict[str, Any], GPUHostEvidence]:
    evidence = _load_json_object(_path(root, TEST_RUN_PATH))
    validate_run_evidence(
        evidence,
        root=root,
        protocol=protocol,
        expected_split="test",
    )
    validate_real_run_operational_evidence(evidence, require_tracking=True)
    if evidence.get("protocol_state_at_export") != ProtocolState.FROZEN.value:
        raise ValueError("locked-test evidence was not exported under the frozen protocol")
    if evidence.get("locked_test_authorized_at_export") is not True:
        raise ValueError("locked-test evidence was exported without locked-test authorization")
    if evidence.get("run_id") == validation_evidence.get("run_id"):
        raise ValueError("validation and locked-test runs must have distinct run IDs")
    if evidence.get("experiment_id") == validation_evidence.get("experiment_id"):
        raise ValueError("validation and locked-test runs must have distinct experiment IDs")

    host = _load_host_evidence(root, TEST_GPU_HOST_PATH)
    _validate_run_host_binding(evidence, host)
    if host.environment_fingerprint_sha256 != freeze.get("gpu_environment_fingerprint_sha256"):
        raise ValueError("locked-test GPU/software fingerprint differs from the validation freeze")
    return evidence, host


def _bundle_presence(root: Path, paths: tuple[Path, ...]) -> tuple[bool, ...]:
    return tuple(_present(root, path) for path in paths)


def evaluate_phase6_repository_state(root: Path) -> Phase6RepositoryStatus:
    protocol = load_baseline_protocol(root)
    scientific_hash = protocol.scientific_config_hash()
    validation_paths = (
        VALIDATION_RUN_PATH,
        VALIDATION_REVIEW_PATH,
        VALIDATION_GPU_HOST_PATH,
    )
    test_paths = (TEST_RUN_PATH, TEST_GPU_HOST_PATH)
    validation_present = _bundle_presence(root, validation_paths)
    test_present = _bundle_presence(root, test_paths)
    freeze_present = _present(root, FREEZE_RECORD_PATH)

    if protocol.state is ProtocolState.VALIDATION:
        if protocol.locked_test_authorized:
            raise ValueError("validation-state protocol cannot authorize the locked test")
        if freeze_present or any(test_present):
            raise ValueError("locked-test/freeze evidence exists before Phase 06 protocol freeze")
        if not any(validation_present):
            return Phase6RepositoryStatus(
                stage=Phase6Stage.PRE_VALIDATION,
                blocker="real validation run, manual review, and GPU-host evidence are pending",
                scientific_config_hash=scientific_hash,
            )
        if not all(validation_present):
            raise ValueError("Phase 06 validation evidence bundle is incomplete")
        validation, _review, host = _validation_bundle(root, protocol)
        return Phase6RepositoryStatus(
            stage=Phase6Stage.VALIDATION_READY_TO_FREEZE,
            blocker="validation is accepted; protocol freeze/locked-test authorization is pending",
            scientific_config_hash=scientific_hash,
            validation_run_id=str(validation["run_id"]),
            validation_evidence_sha256=str(validation["evidence_sha256"]),
            gpu_environment_fingerprint_sha256=host.environment_fingerprint_sha256,
        )

    if protocol.state is not ProtocolState.FROZEN or not protocol.locked_test_authorized:
        raise ValueError("Phase 06 frozen protocol must explicitly authorize the locked test")
    if not all(validation_present):
        raise ValueError("frozen Phase 06 protocol is missing its validation evidence bundle")
    if not freeze_present:
        raise ValueError("frozen Phase 06 protocol is missing protocol-freeze evidence")

    validation, review, validation_host = _validation_bundle(root, protocol)
    freeze = _validate_freeze_record(
        root,
        protocol=protocol,
        validation_evidence=validation,
        review=review,
        validation_host=validation_host,
    )
    if not any(test_present):
        return Phase6RepositoryStatus(
            stage=Phase6Stage.FROZEN_WAITING_TEST,
            blocker="single authorized real locked-test baseline is pending",
            scientific_config_hash=scientific_hash,
            validation_run_id=str(validation["run_id"]),
            validation_evidence_sha256=str(validation["evidence_sha256"]),
            gpu_environment_fingerprint_sha256=validation_host.environment_fingerprint_sha256,
        )
    if not all(test_present):
        raise ValueError("Phase 06 locked-test evidence bundle is incomplete")

    test, test_host = _test_bundle(
        root,
        protocol,
        freeze=freeze,
        validation_evidence=validation,
    )
    metrics = test.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Phase 06 locked-test metrics are malformed")
    test_metrics = {str(name): float(value) for name, value in metrics.items()}
    tracking = test.get("tracking")
    if not isinstance(tracking, dict):
        raise ValueError("Phase 06 locked-test tracking evidence is malformed")
    return Phase6RepositoryStatus(
        stage=Phase6Stage.COMPLETE,
        blocker=None,
        scientific_config_hash=scientific_hash,
        validation_run_id=str(validation["run_id"]),
        test_run_id=str(test["run_id"]),
        validation_evidence_sha256=str(validation["evidence_sha256"]),
        test_evidence_sha256=str(test["evidence_sha256"]),
        gpu_environment_fingerprint_sha256=test_host.environment_fingerprint_sha256,
        test_metrics=test_metrics,
        tracking_run_reference=str(tracking["run_reference"]),
        tracking_artifact_reference=str(tracking["artifact_reference"]),
    )
