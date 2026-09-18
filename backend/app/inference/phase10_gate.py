from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from app.inference.efficiency_study import aggregate_efficiency_conditions
from app.inference.finetuned_comparison import (
    validate_paired_baseline_finetuned_comparison,
)
from app.inference.finetuned_evidence import validate_portable_finetuned_evidence
from app.inference.finetuned_protocol import load_phase10_protocol
from app.inference.protocol import load_baseline_protocol
from app.training.evidence import sha256_file
from app.training.hub_release import (
    LOCKED_TEST_SOURCE_COMMIT,
    PHASE10_HUB_RELEASE_EVIDENCE_VERSION,
    PHASE10_HUB_SMOKE_EVIDENCE_VERSION,
    validate_release_revision,
)


class Phase10Stage(StrEnum):
    LOCKED_TEST_PENDING = "locked_test_pending"
    EXTERNAL_EVIDENCE_PENDING = "external_evidence_pending"
    COMPLETE = "complete"


@dataclass(frozen=True)
class Phase10RepositoryStatus:
    stage: Phase10Stage
    scientific_config_hash: str
    adapter_sha256: str
    locked_test_evidence_sha256: str | None
    comparison_sha256: str | None
    efficiency_condition_count: int
    hub_repo_id: str | None
    hub_revision: str | None
    complete: bool
    blocker: str | None


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Phase 10 evidence must contain a JSON object: {path}")
    return payload


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Phase 10 evidence requires non-empty {key}")
    return value.strip()


def _validate_locked_test(root: Path) -> tuple[dict[str, Any], str]:
    protocol = load_phase10_protocol(root)
    evidence_dir = root / "evidence/phase-10/locked-test"
    test_path = evidence_dir / "phase10-finetuned-test.json"
    if not test_path.is_file():
        raise FileNotFoundError("one-time fine-tuned locked-test evidence is not preserved")

    payload = _load(test_path)
    validate_portable_finetuned_evidence(
        payload,
        root=root,
        protocol=protocol,
        expected_split="test",
        expected_run_id="phase10-finetuned-test-v1",
        expected_source_commit=LOCKED_TEST_SOURCE_COMMIT,
    )
    evidence_sha = sha256_file(test_path)

    completion = _load(evidence_dir / "completion-summary.json")
    if completion.get("status") != "pass":
        raise ValueError("locked-test completion summary did not record pass")
    if completion.get("locked_test_consumed") is not True:
        raise ValueError("locked-test completion summary does not mark consumption")
    if completion.get("selection_or_retuning_after_test") is not False:
        raise ValueError("locked-test evidence records prohibited post-test selection")
    if completion.get("evidence_sha256") != evidence_sha:
        raise ValueError("locked-test completion summary is not linked to raw evidence")

    sums = (evidence_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    declared: dict[str, str] = {}
    for line in sums:
        digest, separator, filename = line.partition("  ")
        if separator:
            declared[filename] = digest
    if declared.get("phase10-finetuned-test.json") != evidence_sha:
        raise ValueError("locked-test SHA256SUMS does not match raw evidence")
    return payload, evidence_sha


def _validate_comparison(
    root: Path,
    *,
    locked: Mapping[str, Any],
    locked_sha256: str,
) -> str:
    comparison_path = root / "evidence/phase-10/baseline-finetuned-comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError("paired zero-shot versus fine-tuned comparison is missing")
    comparison = _load(comparison_path)
    baseline = _load(root / "evidence/phase-06/test-run.json")
    protocol = load_phase10_protocol(root)
    validate_paired_baseline_finetuned_comparison(
        comparison,
        root=root,
        baseline_protocol=load_baseline_protocol(root),
        finetuned_protocol=protocol,
        baseline_evidence=baseline,
        finetuned_evidence=locked,
        finetuned_evidence_file_sha256=locked_sha256,
        expected_finetuned_source_commit=LOCKED_TEST_SOURCE_COMMIT,
    )
    return _required_text(comparison, "comparison_sha256")


def _validate_efficiency(
    root: Path,
    *,
    locked_sha256: str,
) -> int:
    protocol = load_phase10_protocol(root)
    evidence_dir = root / "evidence/phase-10/data-efficiency"
    aggregate_path = evidence_dir / "aggregate.json"
    conditions_dir = evidence_dir / "conditions"
    if not aggregate_path.is_file():
        raise FileNotFoundError("Phase 10 data-efficiency aggregate evidence is missing")
    if not conditions_dir.is_dir():
        raise FileNotFoundError("Phase 10 data-efficiency condition evidence is missing")

    paths = sorted(conditions_dir.glob("*.json"))
    payloads = [_load(path) for path in paths]
    expected_count = len(protocol.data_efficiency.fractions) * len(
        protocol.data_efficiency.seeds
    )
    if len(payloads) != expected_count:
        raise ValueError(
            f"Phase 10 data-efficiency requires {expected_count} condition files"
        )
    for path, payload in zip(paths, payloads, strict=True):
        if payload.get("locked_test_evidence_sha256") != locked_sha256:
            raise ValueError(f"{path.name} is not linked to the sealed locked test")
        if payload.get("base_phase09_training_config_hash") != (
            protocol.source_training_config_hash
        ):
            raise ValueError(f"{path.name} changed the Phase 09 training hyperparameters")
        tracking = payload.get("tracking")
        if not isinstance(tracking, Mapping) or tracking.get("configured") is not True:
            raise ValueError(f"{path.name} is missing configured W&B tracking")
        if tracking.get("provider") != "wandb":
            raise ValueError(f"{path.name} tracking provider is not W&B")
        _required_text(tracking, "run_reference")
        _required_text(tracking, "artifact_reference")
        environment = payload.get("gpu_environment")
        if not isinstance(environment, Mapping):
            raise ValueError(f"{path.name} is missing GPU environment")
        if (
            environment.get("cuda_available") is not True
            or environment.get("visible_gpu_count") != 1
        ):
            raise ValueError(f"{path.name} is not a single-GPU CUDA run")
        subset = payload.get("subset_manifest")
        if not isinstance(subset, Mapping):
            raise ValueError(f"{path.name} is missing subset lineage")
        if subset.get("locked_test_split_used") is not False:
            raise ValueError(f"{path.name} subset used the locked test")

    rebuilt = aggregate_efficiency_conditions(payloads, protocol=protocol)
    stored = _load(aggregate_path)
    for key, value in rebuilt.items():
        if stored.get(key) != value:
            raise ValueError(f"data-efficiency aggregate disagrees with conditions: {key}")

    recorded = stored.get("conditions")
    if not isinstance(recorded, list) or len(recorded) != expected_count:
        raise ValueError("data-efficiency aggregate condition index is incomplete")
    index = {
        str(item.get("condition_id")): item
        for item in recorded
        if isinstance(item, Mapping)
    }
    for path, payload in zip(paths, payloads, strict=True):
        condition_id = _required_text(payload, "condition_id")
        item = index.get(condition_id)
        if not isinstance(item, Mapping):
            raise ValueError(f"aggregate is missing condition index: {condition_id}")
        if item.get("evidence_sha256") != sha256_file(path):
            raise ValueError(f"aggregate condition checksum is stale: {condition_id}")
        if item.get("adapter_sha256") != payload.get("adapter_sha256"):
            raise ValueError(f"aggregate adapter checksum is stale: {condition_id}")
    return expected_count


def _validate_hub_evidence(
    root: Path,
    *,
    locked_sha256: str,
) -> tuple[str, str]:
    protocol = load_phase10_protocol(root)
    release_path = root / "evidence/phase-10/huggingface-release.json"
    smoke_path = root / "evidence/phase-10/huggingface-smoke.json"
    if not release_path.is_file():
        raise FileNotFoundError("Hugging Face release evidence is missing")
    if not smoke_path.is_file():
        raise FileNotFoundError("Hugging Face clean-smoke evidence is missing")
    release = _load(release_path)
    smoke = _load(smoke_path)

    if release.get("evidence_version") != PHASE10_HUB_RELEASE_EVIDENCE_VERSION:
        raise ValueError("unexpected Hugging Face release evidence version")
    if release.get("status") != "completed" or release.get("public") is not True:
        raise ValueError("Hugging Face release is not a completed public artifact")
    if release.get("phase10_scientific_config_hash") != protocol.scientific_config_hash():
        raise ValueError("Hugging Face release changed the Phase 10 scientific identity")
    if release.get("adapter_sha256") != protocol.candidate_adapter_sha256:
        raise ValueError("Hugging Face release changed the frozen adapter")
    if release.get("base_model_id") != protocol.base_model_id:
        raise ValueError("Hugging Face release changed the frozen base model")
    if release.get("base_model_revision") != protocol.base_model_revision:
        raise ValueError("Hugging Face release changed the frozen base revision")
    if release.get("locked_test_evidence_sha256") != locked_sha256:
        raise ValueError("Hugging Face release is not linked to sealed test evidence")
    if release.get("post_test_retuning") is not False:
        raise ValueError("Hugging Face release records prohibited post-test retuning")
    if release.get("primary_adapter_selection_use") is not False:
        raise ValueError("Hugging Face release changed primary selection")
    repo_id = _required_text(release, "repo_id")
    revision = validate_release_revision(_required_text(release, "revision"))
    _required_text(release, "commit_url")
    model_card_sha = _required_text(release, "model_card_sha256")

    if smoke.get("evidence_version") != PHASE10_HUB_SMOKE_EVIDENCE_VERSION:
        raise ValueError("unexpected Hugging Face smoke evidence version")
    if smoke.get("status") != "completed" or smoke.get("public") is not True:
        raise ValueError("Hugging Face smoke did not verify a public completed artifact")
    if smoke.get("repo_id") != repo_id or smoke.get("revision") != revision:
        raise ValueError("Hugging Face smoke did not load the released immutable revision")
    if smoke.get("phase10_scientific_config_hash") != protocol.scientific_config_hash():
        raise ValueError("Hugging Face smoke changed the scientific identity")
    if smoke.get("adapter_sha256") != protocol.candidate_adapter_sha256:
        raise ValueError("Hugging Face clean download changed the adapter bytes")
    if smoke.get("readme_sha256") != model_card_sha:
        raise ValueError("downloaded Hugging Face model card differs from release")
    if smoke.get("smoke_split") != "validation":
        raise ValueError("Hugging Face smoke must use validation only")
    if smoke.get("parse_status") != "OK" or smoke.get("pipeline_type") != "FINETUNED":
        raise ValueError("Hugging Face clean inference smoke did not satisfy the contract")
    if smoke.get("retrieved_chunk_ids") != []:
        raise ValueError("Hugging Face FINETUNED smoke unexpectedly used retrieval")
    if smoke.get("locked_test_split_used") is not False:
        raise ValueError("Hugging Face smoke reused the locked test")
    environment = smoke.get("gpu_environment")
    if not isinstance(environment, Mapping) or environment.get("visible_gpu_count") != 1:
        raise ValueError("Hugging Face smoke did not use exactly one visible GPU")
    return repo_id, revision


def evaluate_phase10_repository_state(root: Path) -> Phase10RepositoryStatus:
    protocol = load_phase10_protocol(root)
    scientific_hash = protocol.scientific_config_hash()
    try:
        locked, locked_sha = _validate_locked_test(root)
        comparison_sha = _validate_comparison(
            root,
            locked=locked,
            locked_sha256=locked_sha,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        return Phase10RepositoryStatus(
            stage=Phase10Stage.LOCKED_TEST_PENDING,
            scientific_config_hash=scientific_hash,
            adapter_sha256=protocol.candidate_adapter_sha256,
            locked_test_evidence_sha256=None,
            comparison_sha256=None,
            efficiency_condition_count=0,
            hub_repo_id=None,
            hub_revision=None,
            complete=False,
            blocker=str(exc),
        )

    missing: list[str] = []
    if not (root / "evidence/phase-10/data-efficiency/aggregate.json").is_file():
        missing.append("data-efficiency aggregate")
    if not (root / "evidence/phase-10/huggingface-release.json").is_file():
        missing.append("Hugging Face release")
    if not (root / "evidence/phase-10/huggingface-smoke.json").is_file():
        missing.append("Hugging Face clean smoke")
    if missing:
        return Phase10RepositoryStatus(
            stage=Phase10Stage.EXTERNAL_EVIDENCE_PENDING,
            scientific_config_hash=scientific_hash,
            adapter_sha256=protocol.candidate_adapter_sha256,
            locked_test_evidence_sha256=locked_sha,
            comparison_sha256=comparison_sha,
            efficiency_condition_count=0,
            hub_repo_id=None,
            hub_revision=None,
            complete=False,
            blocker="missing external completion evidence: " + ", ".join(missing),
        )

    try:
        condition_count = _validate_efficiency(root, locked_sha256=locked_sha)
        repo_id, revision = _validate_hub_evidence(
            root,
            locked_sha256=locked_sha,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        return Phase10RepositoryStatus(
            stage=Phase10Stage.EXTERNAL_EVIDENCE_PENDING,
            scientific_config_hash=scientific_hash,
            adapter_sha256=protocol.candidate_adapter_sha256,
            locked_test_evidence_sha256=locked_sha,
            comparison_sha256=comparison_sha,
            efficiency_condition_count=0,
            hub_repo_id=None,
            hub_revision=None,
            complete=False,
            blocker=f"invalid external completion evidence: {exc}",
        )

    return Phase10RepositoryStatus(
        stage=Phase10Stage.COMPLETE,
        scientific_config_hash=scientific_hash,
        adapter_sha256=protocol.candidate_adapter_sha256,
        locked_test_evidence_sha256=locked_sha,
        comparison_sha256=comparison_sha,
        efficiency_condition_count=condition_count,
        hub_repo_id=repo_id,
        hub_revision=revision,
        complete=True,
        blocker=None,
    )
