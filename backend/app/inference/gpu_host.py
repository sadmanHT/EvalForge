from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

GPU_HOST_CONTRACT_PATH = Path("configs/phase6-gpu-host.json")
GPU_HOST_EVIDENCE_VERSION = "phase6-gpu-host-evidence-v1"


class GPURuntimeExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    quantization: str = Field(min_length=1)
    compute_dtype: str = Field(min_length=1)
    bnb_4bit_use_double_quant: bool


class GPUHostContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: str = Field(min_length=1)
    reference_smoke_path: str = Field(min_length=1)
    reference_smoke_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    python_version_prefix: str = Field(min_length=1)
    require_cuda: bool
    minimum_compute_capability: tuple[int, int]
    minimum_vram_gib: float = Field(gt=0)
    required_packages: dict[str, str]
    runtime_expectations: GPURuntimeExpectations


class GPUDeviceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    name: str = Field(min_length=1)
    total_memory_bytes: int = Field(gt=0)
    compute_capability: tuple[int, int]


class GPUHostEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_version: str = Field(min_length=1)
    contract_version: str = Field(min_length=1)
    reference_smoke_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    captured_at_utc: datetime
    python_version: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    packages: dict[str, str]
    torch_cuda_version: str | None
    cuda_available: bool
    gpus: tuple[GPUDeviceEvidence, ...]
    nvidia_smi: str = Field(min_length=1)
    environment_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_gpu_host_contract(root: Path) -> GPUHostContract:
    path = root / GPU_HOST_CONTRACT_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GPUHostContract.model_validate(payload)


def verify_reference_smoke(root: Path, contract: GPUHostContract) -> None:
    path = root / contract.reference_smoke_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 01 reference smoke evidence must be a JSON object")
    evidence_sha = payload.get("evidence_sha256")
    if evidence_sha != contract.reference_smoke_evidence_sha256:
        raise ValueError("GPU host contract disagrees with Phase 01 smoke evidence identity")
    unsealed = dict(payload)
    unsealed.pop("evidence_sha256", None)
    if _canonical_json_sha256(unsealed) != evidence_sha:
        raise ValueError("Phase 01 reference smoke evidence checksum is invalid")


def _stable_environment_payload(evidence: GPUHostEvidence) -> dict[str, object]:
    return {
        "contract_version": evidence.contract_version,
        "python_version": evidence.python_version,
        "packages": dict(sorted(evidence.packages.items())),
        "torch_cuda_version": evidence.torch_cuda_version,
        "cuda_available": evidence.cuda_available,
        "gpus": [gpu.model_dump(mode="json") for gpu in evidence.gpus],
    }


def environment_fingerprint(evidence: GPUHostEvidence) -> str:
    return _canonical_json_sha256(_stable_environment_payload(evidence))


def seal_gpu_host_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    draft = dict(payload)
    draft.pop("environment_fingerprint_sha256", None)
    draft.pop("evidence_sha256", None)
    provisional = GPUHostEvidence.model_validate(
        {
            **draft,
            "environment_fingerprint_sha256": "0" * 64,
            "evidence_sha256": "0" * 64,
        }
    )
    normalized = provisional.model_dump(mode="json")
    normalized.pop("environment_fingerprint_sha256", None)
    normalized.pop("evidence_sha256", None)
    normalized["environment_fingerprint_sha256"] = environment_fingerprint(provisional)
    normalized["evidence_sha256"] = _canonical_json_sha256(normalized)
    return normalized


def validate_gpu_host_evidence(
    payload: Mapping[str, Any],
    *,
    contract: GPUHostContract,
) -> GPUHostEvidence:
    evidence = GPUHostEvidence.model_validate(payload)
    if evidence.evidence_version != GPU_HOST_EVIDENCE_VERSION:
        raise ValueError("unsupported Phase 06 GPU host evidence version")
    if evidence.contract_version != contract.contract_version:
        raise ValueError("GPU host evidence contract version mismatch")
    if evidence.reference_smoke_evidence_sha256 != contract.reference_smoke_evidence_sha256:
        raise ValueError("GPU host evidence reference smoke identity mismatch")
    if not evidence.python_version.startswith(contract.python_version_prefix):
        raise ValueError("GPU host Python version is outside the pinned Phase 06 contract")
    if evidence.packages != contract.required_packages:
        raise ValueError("GPU host package versions do not match the pinned Phase 06 contract")
    if contract.require_cuda and not evidence.cuda_available:
        raise ValueError("Phase 06 real baseline requires CUDA")
    if not evidence.gpus:
        raise ValueError("Phase 06 GPU host evidence contains no GPUs")

    minimum_vram_bytes = int(contract.minimum_vram_gib * 1024**3)
    capable = [
        gpu
        for gpu in evidence.gpus
        if gpu.compute_capability >= contract.minimum_compute_capability
        and gpu.total_memory_bytes >= minimum_vram_bytes
    ]
    if not capable:
        raise ValueError("GPU host does not satisfy the minimum compute-capability/VRAM contract")
    if environment_fingerprint(evidence) != evidence.environment_fingerprint_sha256:
        raise ValueError("GPU host environment fingerprint mismatch")
    unsealed = evidence.model_dump(mode="json")
    unsealed.pop("evidence_sha256", None)
    if _canonical_json_sha256(unsealed) != evidence.evidence_sha256:
        raise ValueError("GPU host evidence checksum mismatch")
    return evidence


def capture_gpu_host_evidence(
    *,
    root: Path,
    contract: GPUHostContract,
) -> GPUHostEvidence:
    verify_reference_smoke(root, contract)
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("Phase 06 GPU host preflight requires torch") from exc

    package_versions = {
        package: importlib.metadata.version(package)
        for package in sorted(contract.required_packages)
    }
    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, object]] = []
    if cuda_available:
        for index in range(int(torch.cuda.device_count())):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": str(torch.cuda.get_device_name(index)),
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    nvidia_smi = subprocess.run(
        ["nvidia-smi"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = seal_gpu_host_evidence(
        {
            "evidence_version": GPU_HOST_EVIDENCE_VERSION,
            "contract_version": contract.contract_version,
            "reference_smoke_evidence_sha256": contract.reference_smoke_evidence_sha256,
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": package_versions,
            "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
            "cuda_available": cuda_available,
            "gpus": devices,
            "nvidia_smi": nvidia_smi,
        }
    )
    return validate_gpu_host_evidence(payload, contract=contract)


def hardware_runtime_descriptor(evidence: GPUHostEvidence) -> str:
    gpu_names = ",".join(gpu.name for gpu in evidence.gpus)
    return (
        "phase6-gpu-host-v1 "
        f"environment={evidence.environment_fingerprint_sha256} "
        f"torch={evidence.packages['torch']} "
        f"cuda={evidence.torch_cuda_version} "
        f"gpus={gpu_names}"
    )
