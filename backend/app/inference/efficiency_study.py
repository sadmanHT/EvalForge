from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.data.io import read_jsonl
from app.data.schemas import IncidentRecord, Split
from app.evaluation.contracts import EvaluationExample, Prediction
from app.evaluation.harness import EvaluationHarness
from app.inference.base_model import IncidentInput, InferenceError
from app.inference.costing import HourlyRateCostBackend
from app.inference.data_efficiency import (
    DataEfficiencySubset,
    aggregate_efficiency_metric,
    select_data_efficiency_subset,
)
from app.inference.finetuned_protocol import Phase10Protocol
from app.inference.protocol import load_taxonomy
from app.training.config import TrainingConfigBundle, load_training_config
from app.training.evidence import tree_sha256
from app.training.formatter import TrainingExample, format_incident
from app.training.inference import FineTunedAdapterPipeline, PeftTransformersBackend

EFFICIENCY_PREPARED_MANIFEST_VERSION: Literal["phase10-efficiency-prepared-v1"] = (
    "phase10-efficiency-prepared-v1"
)
EFFICIENCY_CONDITION_EVIDENCE_VERSION = "phase10-data-efficiency-condition-v1"
EFFICIENCY_AGGREGATE_VERSION = "phase10-data-efficiency-aggregate-v1"
EFFICIENCY_NUMERIC_RECOVERY_POLICY = "fp16_grad_scaler_recover_transient_nonfinite_grad_norm_v1"


class EfficiencyPreparedManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: Literal["phase10-efficiency-prepared-v1"] = (
        EFFICIENCY_PREPARED_MANIFEST_VERSION
    )
    dataset_version: str = Field(min_length=1)
    dataset_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_content_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase10_scientific_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fraction: float
    seed: int
    subset_policy: Literal["nested_seeded_family_prefix_v1"]
    population_train_family_ids: tuple[str, ...]
    selected_family_ids: tuple[str, ...]
    selected_incident_ids: tuple[str, ...]
    selected_root_cause_codes: tuple[str, ...]
    train_record_count: int = Field(gt=0)
    validation_record_count: int = Field(gt=0)
    prompt_version: str = Field(min_length=1)
    output_schema_version: str = Field(min_length=1)
    train_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_adapter_selection_use: Literal[False] = False
    locked_test_split_used: Literal[False] = False


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def efficiency_condition_id(fraction: float, seed: int) -> str:
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("fraction must be within (0, 1]")
    basis_points = round(fraction * 1000)
    if abs((basis_points / 1000.0) - fraction) > 1e-12:
        raise ValueError("fraction must be representable to three decimal places")
    return f"phase10-efficiency-f{basis_points:04d}-s{int(seed)}"


def _write_examples(path: Path, examples: Sequence[TrainingExample]) -> str:
    body = "".join(
        _canonical_json(example.model_dump(mode="json")) + "\n"
        for example in sorted(examples, key=lambda item: item.incident_id)
    )
    path.write_text(body, encoding="utf-8")
    return hashlib.sha256(body.encode()).hexdigest()


def _dataset_paths(root: Path, protocol: Phase10Protocol) -> tuple[Path, Path]:
    dataset_dir = root / "datasets/incident_diagnosis/processed" / protocol.dataset_version
    return dataset_dir / "manifest.json", dataset_dir / "incidents.jsonl"


def prepare_efficiency_condition(
    *,
    root: Path,
    output_dir: Path,
    protocol: Phase10Protocol,
    fraction: float,
    seed: int,
) -> EfficiencyPreparedManifest:
    if fraction not in protocol.data_efficiency.fractions:
        raise ValueError("fraction is not declared in the Phase 10 data-efficiency matrix")
    if seed not in protocol.data_efficiency.seeds:
        raise ValueError("seed is not declared in the Phase 10 data-efficiency matrix")
    if protocol.data_efficiency.primary_adapter_selection_use is not False:
        raise ValueError("data-efficiency study cannot select the primary adapter")

    manifest_path, incident_path = _dataset_paths(root, protocol)
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("dataset_version") != protocol.dataset_version:
        raise ValueError("data-efficiency dataset version disagrees with source manifest")
    if source_manifest.get("manifest_checksum") != protocol.test_split_manifest_checksum:
        raise ValueError("data-efficiency source manifest checksum disagrees with Phase 10")

    records = read_jsonl(incident_path, IncidentRecord)
    train_records = tuple(record for record in records if record.split is Split.TRAIN)
    validation_records = tuple(record for record in records if record.split is Split.VALIDATION)
    if not train_records or not validation_records:
        raise ValueError("data-efficiency study requires non-empty train and validation splits")

    subset: DataEfficiencySubset = select_data_efficiency_subset(
        train_records,
        fraction=fraction,
        seed=seed,
    )
    selected_ids = set(subset.selected_incident_ids)
    selected_train = tuple(record for record in train_records if record.incident_id in selected_ids)
    if len(selected_train) != len(selected_ids):
        raise ValueError("selected data-efficiency incident IDs are incomplete")

    selected_by_id = {record.incident_id: record for record in selected_train}
    for record in selected_train:
        if not record.is_synthetic:
            continue
        parent = selected_by_id.get(record.parent_incident_id or "")
        if parent is None:
            raise ValueError("selected synthetic record is missing its train-family parent")
        if parent.incident_family_id != record.incident_family_id:
            raise ValueError("selected synthetic lineage crosses incident families")

    labels, _categories = load_taxonomy(root)
    train_examples = tuple(
        format_incident(
            record,
            allowed_labels=labels,
            include_reasoning=True,
        )
        for record in selected_train
    )
    validation_examples = tuple(
        format_incident(
            record,
            allowed_labels=labels,
            include_reasoning=True,
        )
        for record in validation_records
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_sha256 = _write_examples(output_dir / "train.jsonl", train_examples)
    validation_sha256 = _write_examples(output_dir / "validation.jsonl", validation_examples)
    lineage_payload = {
        "fraction": fraction,
        "seed": seed,
        "subset_policy": protocol.data_efficiency.subset_policy,
        "population_train_family_ids": sorted(
            {record.incident_family_id for record in train_records}
        ),
        "selected_records": [
            {
                "incident_id": record.incident_id,
                "incident_family_id": record.incident_family_id,
                "is_synthetic": record.is_synthetic,
                "parent_incident_id": record.parent_incident_id,
                "root_cause_code": record.root_cause_code,
            }
            for record in sorted(selected_train, key=lambda item: item.incident_id)
        ],
    }
    lineage_sha256 = hashlib.sha256(_canonical_json(lineage_payload).encode()).hexdigest()

    prepared = EfficiencyPreparedManifest(
        dataset_version=protocol.dataset_version,
        dataset_manifest_checksum=str(source_manifest["manifest_checksum"]),
        dataset_content_checksum=str(source_manifest["content_checksum"]),
        phase10_scientific_config_hash=protocol.scientific_config_hash(),
        fraction=fraction,
        seed=seed,
        subset_policy=protocol.data_efficiency.subset_policy,
        population_train_family_ids=tuple(
            sorted({record.incident_family_id for record in train_records})
        ),
        selected_family_ids=subset.selected_family_ids,
        selected_incident_ids=subset.selected_incident_ids,
        selected_root_cause_codes=subset.root_cause_codes,
        train_record_count=len(train_examples),
        validation_record_count=len(validation_examples),
        prompt_version=protocol.prompt_version,
        output_schema_version=protocol.output_schema_version,
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
        lineage_sha256=lineage_sha256,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            prepared.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return prepared


def build_efficiency_training_bundle(root: Path, *, seed: int) -> TrainingConfigBundle:
    base = load_training_config(root)
    return TrainingConfigBundle(
        lora=base.lora.model_copy(update={"seed": int(seed)}),
        training=base.training.model_copy(update={"seed": int(seed)}),
    )


def _error_prediction(
    *,
    incident_id: str,
    error: InferenceError,
    attempts: int,
    protocol: Phase10Protocol,
    allowed_labels: tuple[str, ...],
    adapter_sha256: str,
) -> Prediction:
    return Prediction.from_raw_output(
        incident_id=incident_id,
        raw_model_output="",
        allowed_labels=set(allowed_labels),
        pipeline_metadata={
            "pipeline_type": "FINETUNED",
            "prompt_version": protocol.prompt_version,
            "output_schema_version": protocol.output_schema_version,
            "generation_config": protocol.generation_config.model_dump(mode="json"),
            "phase10_efficiency_adapter_sha256": adapter_sha256,
            "inference_error": {
                "type": type(error).__name__,
                "message": str(error),
                "attempts": attempts,
            },
        },
    )


def evaluate_efficiency_validation(
    *,
    root: Path,
    protocol: Phase10Protocol,
    adapter_dir: Path,
    adapter_sha256: str,
    gpu_hour_usd: float,
    max_attempts: int = 2,
) -> dict[str, object]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if tree_sha256(adapter_dir) != adapter_sha256:
        raise ValueError("efficiency adapter checksum does not match its training result")

    labels, categories = load_taxonomy(root)
    backend = PeftTransformersBackend(
        protocol.runtime_config,
        adapter_id=str(adapter_dir),
        adapter_revision=adapter_sha256,
        use_adapter_revision_for_loading=False,
    )
    costed_backend = HourlyRateCostBackend(backend, gpu_hour_usd=gpu_hour_usd)
    pipeline = FineTunedAdapterPipeline(
        backend=costed_backend,
        allowed_labels=labels,
        prompt_version=protocol.prompt_version,
        generation_config=protocol.generation_config,
    )
    if pipeline.backend.runtime_config != protocol.runtime_config:
        raise ValueError("efficiency inference changed the frozen runtime configuration")
    if pipeline.allowed_labels != labels:
        raise ValueError("efficiency inference changed the frozen label order")
    if pipeline.prompt_template.version != protocol.prompt_version:
        raise ValueError("efficiency inference changed the frozen prompt")
    if pipeline.generation_config != protocol.generation_config:
        raise ValueError("efficiency inference changed the frozen generation configuration")

    _manifest_path, incident_path = _dataset_paths(root, protocol)
    validation = sorted(
        (
            record
            for record in read_jsonl(incident_path, IncidentRecord)
            if record.split is Split.VALIDATION
        ),
        key=lambda item: item.incident_id,
    )
    harness = EvaluationHarness(
        evaluator_version=protocol.evaluator_version,
        allowed_labels=labels,
        label_to_category=categories,
        ece_bins=protocol.ece_bins,
    )

    predictions: list[Prediction] = []
    examples: list[EvaluationExample] = []
    failures = 0
    for record in validation:
        last_error: InferenceError | None = None
        prediction: Prediction | None = None
        for _attempt in range(1, max_attempts + 1):
            try:
                prediction = pipeline.predict(
                    IncidentInput(
                        incident_id=record.incident_id,
                        title=record.title,
                        description=record.description,
                    )
                )
                break
            except InferenceError as exc:
                last_error = exc
        if prediction is None:
            if last_error is None:
                raise AssertionError("efficiency inference retry loop exited unexpectedly")
            prediction = _error_prediction(
                incident_id=record.incident_id,
                error=last_error,
                attempts=max_attempts,
                protocol=protocol,
                allowed_labels=labels,
                adapter_sha256=adapter_sha256,
            )
            failures += 1
        predictions.append(prediction)
        examples.append(
            EvaluationExample(
                incident_id=record.incident_id,
                split="validation",
                root_cause_code=record.root_cause_code,
                root_cause_category=record.root_cause_category,
            )
        )

    result = harness.evaluate(examples, predictions)
    return {
        "split": "validation",
        "prediction_count": len(predictions),
        "inference_failure_count": failures,
        "result_hash": result.result_hash,
        "metric_values": result.metric_map(),
        "evaluation": result.canonical_payload(),
        "predictions": [
            prediction.model_dump(mode="json", exclude_none=False) for prediction in predictions
        ],
    }


def validate_efficiency_condition_evidence(
    payload: Mapping[str, Any],
    *,
    protocol: Phase10Protocol,
) -> tuple[float, int]:
    if payload.get("evidence_version") != EFFICIENCY_CONDITION_EVIDENCE_VERSION:
        raise ValueError("unsupported Phase 10 efficiency condition evidence version")
    if payload.get("status") != "completed":
        raise ValueError("Phase 10 efficiency condition is not completed")
    if payload.get("study_role") != "secondary_descriptive_no_primary_selection":
        raise ValueError("efficiency evidence has an invalid study role")
    if payload.get("primary_adapter_selection_use") is not False:
        raise ValueError("efficiency evidence cannot select the primary adapter")
    if payload.get("test_split_used") is not False:
        raise ValueError("efficiency evidence must not use the locked test")
    if payload.get("phase10_scientific_config_hash") != protocol.scientific_config_hash():
        raise ValueError("efficiency evidence scientific identity mismatch")

    fraction = payload.get("fraction")
    seed = payload.get("seed")
    if isinstance(fraction, bool) or not isinstance(fraction, int | float):
        raise ValueError("efficiency evidence fraction is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("efficiency evidence seed is invalid")
    fraction = float(fraction)
    if fraction not in protocol.data_efficiency.fractions:
        raise ValueError("efficiency evidence fraction is outside the declared matrix")
    if seed not in protocol.data_efficiency.seeds:
        raise ValueError("efficiency evidence seed is outside the declared matrix")

    if payload.get("training_numeric_recovery_policy") != EFFICIENCY_NUMERIC_RECOVERY_POLICY:
        raise ValueError("efficiency evidence has an unexpected numeric recovery policy")
    recovery_steps = payload.get("nonfinite_gradient_norm_steps")
    recovery_count = payload.get("nonfinite_gradient_norm_count")
    if not isinstance(recovery_steps, list) or any(
        isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in recovery_steps
    ):
        raise ValueError("efficiency evidence has invalid non-finite grad-norm steps")
    if recovery_count != len(recovery_steps):
        raise ValueError("efficiency evidence has a stale non-finite grad-norm count")
    if len(set(recovery_steps)) != len(recovery_steps):
        raise ValueError("efficiency evidence repeats non-finite grad-norm steps")

    validation = payload.get("validation_evaluation")
    if not isinstance(validation, Mapping) or validation.get("split") != "validation":
        raise ValueError("efficiency evidence must evaluate validation only")
    if validation.get("prediction_count") != 6:
        raise ValueError("efficiency validation must cover all six validation incidents")
    return fraction, seed


def aggregate_efficiency_conditions(
    payloads: Sequence[Mapping[str, Any]],
    *,
    protocol: Phase10Protocol,
) -> dict[str, object]:
    expected = {
        (fraction, seed)
        for seed in protocol.data_efficiency.seeds
        for fraction in protocol.data_efficiency.fractions
    }
    by_key: dict[tuple[float, int], Mapping[str, Any]] = {}
    for payload in payloads:
        key = validate_efficiency_condition_evidence(payload, protocol=protocol)
        if key in by_key:
            raise ValueError(f"duplicate efficiency condition evidence: {key}")
        by_key[key] = payload
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        unexpected = sorted(set(by_key) - expected)
        raise ValueError(
            f"efficiency matrix coverage mismatch: missing={missing}, unexpected={unexpected}"
        )

    metric_names = (
        "primary.exact_accuracy",
        "quality.parse_failure_rate",
        "calibration.normalized_multiclass_brier",
        "latency.p50_ms",
        "latency.p95_ms",
        "cost.marginal_mean_usd",
    )
    fractions: list[dict[str, object]] = []
    for fraction in protocol.data_efficiency.fractions:
        members = [by_key[(fraction, seed)] for seed in protocol.data_efficiency.seeds]
        metrics: dict[str, object] = {}
        for name in metric_names:
            values: list[float] = []
            for member in members:
                validation = member["validation_evaluation"]
                assert isinstance(validation, Mapping)
                metric_values = validation.get("metric_values")
                if not isinstance(metric_values, Mapping):
                    raise ValueError("efficiency validation metric_values are malformed")
                raw = metric_values.get(name)
                if isinstance(raw, bool) or not isinstance(raw, int | float):
                    raise ValueError(f"efficiency condition is missing metric: {name}")
                values.append(float(raw))
            metrics[name] = aggregate_efficiency_metric(values)
        fractions.append(
            {
                "fraction": fraction,
                "seeds": list(protocol.data_efficiency.seeds),
                "condition_ids": [str(member["condition_id"]) for member in members],
                "metrics": metrics,
                "training_wall_seconds": aggregate_efficiency_metric(
                    [float(member["training_wall_seconds"]) for member in members]
                ),
                "training_direct_cost_usd": aggregate_efficiency_metric(
                    [float(member["training_direct_cost_usd"]) for member in members]
                ),
                "nonfinite_gradient_norm_count": aggregate_efficiency_metric(
                    [float(member["nonfinite_gradient_norm_count"]) for member in members]
                ),
            }
        )

    return {
        "evidence_version": EFFICIENCY_AGGREGATE_VERSION,
        "status": "completed",
        "study_role": "secondary_descriptive_no_primary_selection",
        "phase10_scientific_config_hash": protocol.scientific_config_hash(),
        "dataset_version": protocol.dataset_version,
        "fractions": fractions,
        "condition_count": len(payloads),
        "primary_adapter_selection_use": False,
        "test_split_used": False,
    }
