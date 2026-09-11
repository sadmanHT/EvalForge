#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.experiment_config import ExperimentConfig, PipelineType
from app.db import build_engine, session_scope
from app.evaluation.contracts import EvaluationExample, Prediction, RankedLabel
from app.evaluation.harness import EvaluationHarness
from app.evaluation.persistence import persist_evaluation, score_stored_run
from app.models import Incident
from app.repositories import PersistenceRepository

DATASET_VERSION = "evalforge-incident-diagnosis-v0.1.0"
EXPERIMENT_ID = "exp-phase5-evaluator-smoke"
RUN_ID = "run-phase5-evaluator-smoke"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_config(root: Path, manifest: dict[str, Any], model: dict[str, Any]) -> ExperimentConfig:
    lock_checksum = hashlib.sha256((root / "backend/requirements.full.lock").read_bytes()).hexdigest()
    return ExperimentConfig(
        study_id=model["study_id"],
        pipeline_type=PipelineType.ZERO_SHOT,
        dataset_version=DATASET_VERSION,
        test_split_manifest_checksum=manifest["manifest_checksum"],
        label_taxonomy_version=manifest["label_taxonomy_version"],
        base_model_id=model["base_model_id"],
        base_model_revision=model["base_model_revision"],
        prompt_version="phase5-evaluator-smoke-v1",
        output_schema_version="root-cause-prediction-v1",
        generation_config={"temperature": 0.0, "do_sample": False},
        temperature=0.0,
        confidence_method="normalized_label_sequence_log_likelihood",
        seed=20260908,
        evaluator_version="phase5-evaluator-v1",
        git_commit="phase5-smoke-fixture-v1",
        dependency_lock_checksum=lock_checksum,
        hardware_runtime_descriptor="phase5-deterministic-smoke-no-model-inference",
        cost_rate_snapshot_version="phase5-smoke-rates-v1",
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest = _load_json(
        root / "datasets/incident_diagnosis/processed" / DATASET_VERSION / "manifest.json"
    )
    model = _load_json(root / "configs/model.yaml")
    taxonomy = _load_json(root / "configs/label-taxonomy.yaml")
    labels = [item["id"] for item in taxonomy["labels"]]
    categories = {item["id"]: item["category"] for item in taxonomy["labels"]}
    config = _build_config(root, manifest, model)
    harness = EvaluationHarness(
        evaluator_version="phase5-evaluator-v1",
        allowed_labels=labels,
        label_to_category=categories,
        ece_bins=10,
    )

    engine = build_engine()
    with session_scope(engine) as session:
        repo = PersistenceRepository(session)
        experiment = repo.create_experiment(EXPERIMENT_ID, config)
        incidents = session.scalars(
            select(Incident)
            .where(Incident.dataset_version == DATASET_VERSION, Incident.split == "test")
            .order_by(Incident.incident_id)
        ).all()
        if len(incidents) != 6:
            raise SystemExit(f"PHASE05_SMOKE=FAIL expected 6 test incidents, found {len(incidents)}")
        examples: list[EvaluationExample] = []
        predictions: list[Prediction] = []
        for index, incident in enumerate(incidents):
            examples.append(
                EvaluationExample(
                    incident_id=incident.incident_id,
                    split="test",
                    root_cause_code=incident.root_cause_code,
                    root_cause_category=incident.root_cause_category,
                )
            )
            probabilities = {label: 0.3 / (len(labels) - 1) for label in labels}
            probabilities[incident.root_cause_code] = 0.7
            ranked = tuple(
                RankedLabel(
                    label=label,
                    score=probabilities[label],
                    probability=probabilities[label],
                )
                for label in sorted(labels, key=lambda label: (probabilities[label], label), reverse=True)
            )
            predictions.append(
                Prediction(
                    incident_id=incident.incident_id,
                    predicted_root_cause_code=incident.root_cause_code,
                    ranked_labels=ranked,
                    label_probabilities=probabilities,
                    confidence_probability=0.7,
                    confidence_source="normalized_label_sequence_log_likelihood",
                    latency_ms=20.0 + index,
                    cost_usd=0.001,
                    raw_model_output=f"phase5 smoke fixture {index}",
                    pipeline_metadata={"fixture": "phase5-smoke", "research_result": False},
                )
            )
        result = harness.evaluate(examples, predictions)
        snapshot = persist_evaluation(
            session,
            experiment_id=experiment.experiment_id,
            run_id=RUN_ID,
            dataset_version=DATASET_VERSION,
            predictions=predictions,
            result=result,
            split="test",
        )
        rescored = score_stored_run(session, run_id=RUN_ID, harness=harness)
        if snapshot.result_hash != result.result_hash or rescored.result_hash != result.result_hash:
            raise SystemExit("PHASE05_SMOKE=FAIL persisted/reloaded result identity mismatch")
        if snapshot.metric_values != result.metric_map() or rescored.metric_map() != result.metric_map():
            raise SystemExit("PHASE05_SMOKE=FAIL persisted/reloaded metric mismatch")

    print("PHASE05_SMOKE=PASS")
    print(f"EXPERIMENT_ID={EXPERIMENT_ID}")
    print(f"RUN_ID={RUN_ID}")
    print(f"RESULT_HASH={result.result_hash}")
    print(f"PREDICTIONS={result.prediction_count}")
    for name, value in sorted(result.metric_map().items()):
        print(f"METRIC {name}={value:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
