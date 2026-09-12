from __future__ import annotations

from pathlib import Path

import pytest

from app.inference.gpu_host import (
    hardware_runtime_descriptor,
    load_gpu_host_contract,
    seal_gpu_host_evidence,
    validate_gpu_host_evidence,
    verify_reference_smoke,
)

ROOT = Path(__file__).resolve().parents[3]


def _valid_payload() -> dict[str, object]:
    contract = load_gpu_host_contract(ROOT)
    return seal_gpu_host_evidence(
        {
            "evidence_version": "phase6-gpu-host-evidence-v1",
            "contract_version": contract.contract_version,
            "reference_smoke_evidence_sha256": contract.reference_smoke_evidence_sha256,
            "captured_at_utc": "2026-09-12T00:00:00+00:00",
            "python_version": "3.12.13 (fixture)",
            "platform": "Linux-fixture-x86_64",
            "packages": dict(contract.required_packages),
            "torch_cuda_version": "12.8",
            "cuda_available": True,
            "gpus": [
                {
                    "index": 0,
                    "name": "Tesla T4",
                    "total_memory_bytes": 15 * 1024**3,
                    "compute_capability": [7, 5],
                }
            ],
            "nvidia_smi": "fixture nvidia-smi output",
        }
    )


def test_gpu_host_evidence_is_sealed_and_descriptor_is_stable() -> None:
    contract = load_gpu_host_contract(ROOT)
    verify_reference_smoke(ROOT, contract)
    evidence = validate_gpu_host_evidence(_valid_payload(), contract=contract)

    descriptor = hardware_runtime_descriptor(evidence)
    assert evidence.environment_fingerprint_sha256 in descriptor
    assert "torch=2.10.0+cu128" in descriptor
    assert "gpus=Tesla T4" in descriptor


def test_gpu_host_evidence_rejects_package_drift() -> None:
    contract = load_gpu_host_contract(ROOT)
    payload = _valid_payload()
    packages = dict(payload["packages"])
    packages["transformers"] = "4.58.0"
    payload["packages"] = packages

    with pytest.raises(ValueError, match="package versions"):
        validate_gpu_host_evidence(payload, contract=contract)


def test_gpu_host_evidence_rejects_undersized_gpu() -> None:
    contract = load_gpu_host_contract(ROOT)
    payload = seal_gpu_host_evidence(
        {
            **_valid_payload(),
            "gpus": [
                {
                    "index": 0,
                    "name": "fixture-small-gpu",
                    "total_memory_bytes": 8 * 1024**3,
                    "compute_capability": [7, 5],
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="compute-capability/VRAM"):
        validate_gpu_host_evidence(payload, contract=contract)


def test_gpu_host_evidence_rejects_checksum_tampering() -> None:
    contract = load_gpu_host_contract(ROOT)
    payload = _valid_payload()
    payload["nvidia_smi"] = "tampered after sealing"

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_gpu_host_evidence(payload, contract=contract)
