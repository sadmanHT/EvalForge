# ruff: noqa: I001
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

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


EFFICIENCY_PACKAGE_VALIDATION_VERSION = "phase10-data-efficiency-package-validation-v1"
EFFICIENCY_PACKAGE_SHA256 = "c1616b7fffc58e78f107de91725e706c4af410632e105513add4d700fb149139"
EFFICIENCY_SOURCE_AGGREGATE_SHA256 = (
    "7ae6c09672e3e12992908032379dd30b431c3d2aac90b39a9d2f2ca7c463b30e"
)
EFFICIENCY_PRESERVED_AGGREGATE_SHA256 = (
    "ea3159ed0a962c50295d44bfabf40eae92014dc878f263b7d755d4a6ee66a261"
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


def _required_sha256(payload: Mapping[str, Any], key: str) -> str:
    value = _required_text(payload, key).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"Phase 10 evidence requires 64-hex {key}")
    return value


def _parse_sha256sums(path: Path) -> dict[str, str]:
    declared: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, filename = line.partition("  ")
        if separator:
            declared[filename] = digest
    return declared


def _validate_compact_efficiency(
    root: Path,
    *,
    locked_sha256: str,
    aggregate_path: Path,
    evidence_dir: Path,
) -> int:
    """Validate the sealed connected-run package without exploding every raw JSON into git.

    The full Kaggle ZIP is content-addressed by package-validation.json.  The repository
    preserves its aggregate, original SHA256SUMS roster, per-condition W&B identities and
    independent package-validation manifest.  Raw condition files remain recoverable from the
    sealed ZIP / W&B artifacts and their exact SHA-256 values are checked here.
    """

    protocol = load_phase10_protocol(root)
    validation_path = evidence_dir / "package-validation.json"
    sums_path = evidence_dir / "SHA256SUMS"
    if not validation_path.is_file() or not sums_path.is_file():
        raise FileNotFoundError("compact Phase 10 data-efficiency package evidence is incomplete")

    aggregate = _load(aggregate_path)
    validation = _load(validation_path)
    expected_count = len(protocol.data_efficiency.fractions) * len(protocol.data_efficiency.seeds)

    if validation.get("evidence_version") != EFFICIENCY_PACKAGE_VALIDATION_VERSION:
        raise ValueError("unexpected data-efficiency package-validation version")
    if validation.get("status") != "completed":
        raise ValueError("data-efficiency package validation is not completed")
    if _required_sha256(validation, "package_sha256") != EFFICIENCY_PACKAGE_SHA256:
        raise ValueError("data-efficiency package SHA-256 changed")
    if _required_sha256(validation, "aggregate_sha256") != EFFICIENCY_SOURCE_AGGREGATE_SHA256:
        raise ValueError("data-efficiency source aggregate SHA-256 changed")
    if sha256_file(aggregate_path) != EFFICIENCY_PRESERVED_AGGREGATE_SHA256:
        raise ValueError("preserved data-efficiency aggregate changed")
    if sha256_file(sums_path) != _required_sha256(validation, "sha256sums_sha256"):
        raise ValueError("preserved data-efficiency SHA256SUMS changed")
    if validation.get("phase10_scientific_config_hash") != protocol.scientific_config_hash():
        raise ValueError("data-efficiency package changed the Phase 10 scientific identity")
    if validation.get("source_git_commit") != "193969c73b0bb278222f8ef902f05639981f1a82":
        raise ValueError("data-efficiency package source commit changed")
    if validation.get("locked_test_evidence_sha256") != locked_sha256:
        raise ValueError("data-efficiency package is not linked to the sealed locked test")
    if validation.get("comparison_sha256") != (
        "01409f6e1a04a970f451ed580b8b0f01e72b4fa09b51ae52071e562aa05e4105"
    ):
        raise ValueError("data-efficiency package comparison linkage changed")
    for key in (
        "secure_archive_paths_verified",
        "sha256sums_verified",
        "canonical_validation_result_hashes_verified",
        "validation_prediction_ids_verified",
    ):
        if validation.get(key) is not True:
            raise ValueError(f"data-efficiency package validation did not verify {key}")
    if validation.get("test_ids_touched") != 0 or validation.get("locked_test_usage") != "none":
        raise ValueError("data-efficiency package touched the locked test")
    if validation.get("test_split_used") is not False:
        raise ValueError("data-efficiency package used the test split")
    if validation.get("primary_adapter_selection_use") is not False:
        raise ValueError("data-efficiency package changed primary adapter selection")

    if aggregate.get("evidence_version") != "phase10-data-efficiency-aggregate-v1":
        raise ValueError("unexpected data-efficiency aggregate version")
    if aggregate.get("status") != "completed":
        raise ValueError("data-efficiency aggregate is not completed")
    if aggregate.get("dataset_version") != protocol.dataset_version:
        raise ValueError("data-efficiency aggregate dataset changed")
    if aggregate.get("phase10_scientific_config_hash") != protocol.scientific_config_hash():
        raise ValueError("data-efficiency aggregate scientific identity changed")
    if aggregate.get("study_role") != "secondary_descriptive_no_primary_selection":
        raise ValueError("data-efficiency aggregate study role changed")
    if aggregate.get("test_split_used") is not False:
        raise ValueError("data-efficiency aggregate used the test split")
    if aggregate.get("primary_adapter_selection_use") is not False:
        raise ValueError("data-efficiency aggregate changed primary adapter selection")
    if aggregate.get("condition_count") != expected_count:
        raise ValueError(f"data-efficiency aggregate requires {expected_count} conditions")
    if validation.get("condition_count") != expected_count:
        raise ValueError(f"data-efficiency package validation requires {expected_count} conditions")

    aggregate_conditions = aggregate.get("conditions")
    validated_conditions = validation.get("conditions")
    if not isinstance(aggregate_conditions, list) or not isinstance(validated_conditions, list):
        raise ValueError("data-efficiency condition indexes are malformed")
    if len(aggregate_conditions) != expected_count or len(validated_conditions) != expected_count:
        raise ValueError("data-efficiency condition indexes are incomplete")

    expected_pairs = {
        (float(fraction), int(seed))
        for fraction in protocol.data_efficiency.fractions
        for seed in protocol.data_efficiency.seeds
    }
    aggregate_index = {
        str(item.get("condition_id")): item
        for item in aggregate_conditions
        if isinstance(item, Mapping)
    }
    validation_index = {
        str(item.get("condition_id")): item
        for item in validated_conditions
        if isinstance(item, Mapping)
    }
    if len(aggregate_index) != expected_count or set(aggregate_index) != set(validation_index):
        raise ValueError("data-efficiency condition IDs are incomplete or duplicated")

    declared = _parse_sha256sums(sums_path)
    if declared.get("evidence/phase-10/data-efficiency/aggregate.json") != (
        EFFICIENCY_SOURCE_AGGREGATE_SHA256
    ):
        raise ValueError("source SHA256SUMS aggregate checksum changed")

    observed_pairs: set[tuple[float, int]] = set()
    for condition_id, item in validation_index.items():
        fraction = item.get("fraction")
        seed = item.get("seed")
        if isinstance(fraction, bool) or not isinstance(fraction, int | float):
            raise ValueError(f"{condition_id} has invalid fraction")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{condition_id} has invalid seed")
        observed_pairs.add((float(fraction), seed))
        if item.get("base_phase09_training_config_hash") != protocol.source_training_config_hash:
            raise ValueError(f"{condition_id} changed Phase 09 training hyperparameters")
        if item.get("locked_test_evidence_sha256") != locked_sha256:
            raise ValueError(f"{condition_id} is not linked to the sealed locked test")
        if item.get("test_split_used") is not False:
            raise ValueError(f"{condition_id} used the test split")
        if item.get("primary_adapter_selection_use") is not False:
            raise ValueError(f"{condition_id} changed primary adapter selection")
        if item.get("visible_gpu_count") != 1:
            raise ValueError(f"{condition_id} did not use exactly one visible GPU")
        evidence_sha = _required_sha256(item, "evidence_sha256")
        _required_sha256(item, "adapter_sha256")
        _required_sha256(item, "validation_result_hash")
        _required_text(item, "run_reference")
        _required_text(item, "artifact_reference")

        condition_filename = f"evidence/phase-10/data-efficiency/conditions/{condition_id}.json"
        if declared.get(condition_filename) != evidence_sha:
            raise ValueError(f"{condition_id} checksum is not sealed by source SHA256SUMS")

        aggregate_item = aggregate_index[condition_id]
        for key in (
            "fraction",
            "seed",
            "evidence_sha256",
            "adapter_sha256",
            "validation_result_hash",
        ):
            if aggregate_item.get(key) != item.get(key):
                raise ValueError(f"{condition_id} aggregate index disagrees on {key}")
        tracking = aggregate_item.get("tracking")
        if not isinstance(tracking, Mapping) or tracking.get("configured") is not True:
            raise ValueError(f"{condition_id} aggregate tracking is not configured")
        if tracking.get("provider") != "wandb":
            raise ValueError(f"{condition_id} aggregate tracking provider is not W&B")
        if tracking.get("run_reference") != item.get("run_reference"):
            raise ValueError(f"{condition_id} W&B run reference changed")
        if tracking.get("artifact_reference") != item.get("artifact_reference"):
            raise ValueError(f"{condition_id} W&B artifact reference changed")

    if observed_pairs != expected_pairs:
        raise ValueError("data-efficiency fraction/seed matrix is incomplete")

    fraction_rows = aggregate.get("fractions")
    if not isinstance(fraction_rows, list) or len(fraction_rows) != len(
        protocol.data_efficiency.fractions
    ):
        raise ValueError("data-efficiency fraction aggregates are incomplete")
    by_fraction = {
        float(row.get("fraction")): row
        for row in fraction_rows
        if isinstance(row, Mapping) and isinstance(row.get("fraction"), int | float)
    }
    if set(by_fraction) != {float(value) for value in protocol.data_efficiency.fractions}:
        raise ValueError("data-efficiency fraction aggregates changed")
    for fraction in protocol.data_efficiency.fractions:
        row = by_fraction[float(fraction)]
        if row.get("seeds") != list(protocol.data_efficiency.seeds):
            raise ValueError(f"fraction {fraction} seed index changed")
        condition_ids = row.get("condition_ids")
        if not isinstance(condition_ids, list) or len(condition_ids) != len(
            protocol.data_efficiency.seeds
        ):
            raise ValueError(f"fraction {fraction} condition index is incomplete")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"fraction {fraction} metrics are missing")
        for metric_name in ("primary.exact_accuracy", "quality.parse_failure_rate"):
            metric = metrics.get(metric_name)
            if not isinstance(metric, Mapping) or metric.get("count") != len(
                protocol.data_efficiency.seeds
            ):
                raise ValueError(f"fraction {fraction} metric {metric_name} is incomplete")

    return expected_count


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
        return _validate_compact_efficiency(
            root,
            locked_sha256=locked_sha256,
            aggregate_path=aggregate_path,
            evidence_dir=evidence_dir,
        )

    paths = sorted(conditions_dir.glob("*.json"))
    payloads = [_load(path) for path in paths]
    expected_count = len(protocol.data_efficiency.fractions) * len(protocol.data_efficiency.seeds)
    if len(payloads) != expected_count:
        raise ValueError(f"Phase 10 data-efficiency requires {expected_count} condition files")
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
    index = {str(item.get("condition_id")): item for item in recorded if isinstance(item, Mapping)}
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
