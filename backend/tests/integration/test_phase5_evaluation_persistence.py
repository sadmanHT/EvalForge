from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.core.experiment_config import ExperimentConfig, PipelineType
from app.db import build_engine, sqlalchemy_database_url
from app.evaluation.contracts import EvaluationExample, Prediction, RankedLabel
from app.evaluation.harness import EvaluationHarness
from app.evaluation.persistence import persist_evaluation, reload_evaluation, score_stored_run
from app.models import Incident
from app.repositories import PersistenceRepository
from app.services.dataset_import import import_phase3_dataset

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
DATASET_VERSION = "evalforge-incident-diagnosis-v0.1.0"


def _alembic_config() -> Config:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "migrations"))
    config.set_main_option(
        "sqlalchemy.url", sqlalchemy_database_url(Settings.from_env().database_url)
    )
    return config


def _manifest() -> dict[str, Any]:
    path = ROOT / "datasets/incident_diagnosis/processed" / DATASET_VERSION / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _model_config() -> dict[str, Any]:
    return json.loads((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))


def _taxonomy() -> tuple[list[str], dict[str, str]]:
    payload = json.loads((ROOT / "configs/label-taxonomy.yaml").read_text(encoding="utf-8"))
    labels = [item["id"] for item in payload["labels"]]
    categories = {item["id"]: item["category"] for item in payload["labels"]}
    return labels, categories


def _experiment_config() -> ExperimentConfig:
    manifest = _manifest()
    model = _model_config()
    lock_checksum = hashlib.sha256((BACKEND / "requirements.full.lock").read_bytes()).hexdigest()
    return ExperimentConfig(
        study_id=model["study_id"],
        pipeline_type=PipelineType.ZERO_SHOT,
        dataset_version=DATASET_VERSION,
        test_split_manifest_checksum=manifest["manifest_checksum"],
        label_taxonomy_version=manifest["label_taxonomy_version"],
        base_model_id=model["base_model_id"],
        base_model_revision=model["base_model_revision"],
        prompt_version="phase5-persistence-smoke-v1",
        output_schema_version="root-cause-prediction-v1",
        generation_config={"temperature": 0.0, "do_sample": False},
        temperature=0.0,
        confidence_method="normalized_label_sequence_log_likelihood",
        seed=20260908,
        evaluator_version="phase5-evaluator-v1",
        git_commit="phase5-integration",
        dependency_lock_checksum=lock_checksum,
        hardware_runtime_descriptor="phase5-postgres-integration-no-model-inference",
        cost_rate_snapshot_version="phase5-test-rates-v1",
    )


@pytest.fixture(scope="module")
def engine() -> Engine:
    engine = build_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    command.upgrade(_alembic_config(), "head")
    yield engine
    engine.dispose()


def _canonical_fixture(
    engine: Engine,
) -> tuple[list[EvaluationExample], list[Prediction], EvaluationHarness]:
    labels, categories = _taxonomy()
    with Session(engine) as session:
        incidents = session.scalars(
            select(Incident)
            .where(Incident.dataset_version == DATASET_VERSION, Incident.split == "test")
            .order_by(Incident.incident_id)
        ).all()
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
            RankedLabel(label=label, score=probabilities[label], probability=probabilities[label])
            for label in sorted(
                labels, key=lambda label: (probabilities[label], label), reverse=True
            )
        )
        predictions.append(
            Prediction(
                incident_id=incident.incident_id,
                predicted_root_cause_code=incident.root_cause_code,
                ranked_labels=ranked,
                label_probabilities=probabilities,
                confidence_probability=0.7,
                confidence_source="normalized_label_sequence_log_likelihood",
                latency_ms=10.0 + index,
                cost_usd=0.001 + index * 0.0001,
                raw_model_output=f"phase5 deterministic fixture {index}",
                pipeline_metadata={"fixture": "phase5-persistence-smoke"},
            )
        )
    harness = EvaluationHarness(
        evaluator_version="phase5-evaluator-v1",
        allowed_labels=labels,
        label_to_category=categories,
        ece_bins=10,
    )
    return examples, predictions, harness


def test_phase5_evaluation_persists_reloads_and_rescores_identically(engine: Engine) -> None:
    with Session(engine) as session:
        import_phase3_dataset(session, ROOT, DATASET_VERSION)
        repo = PersistenceRepository(session)
        experiment = repo.create_experiment("exp-phase5-persistence", _experiment_config())
        session.commit()
        assert experiment.dataset_version == DATASET_VERSION

    examples, predictions, harness = _canonical_fixture(engine)
    assert len(examples) == len(predictions) == 6
    result = harness.evaluate(examples, predictions)
    assert result.metric_map()["primary.exact_accuracy"] == 1.0

    with Session(engine) as session:
        first = persist_evaluation(
            session,
            experiment_id="exp-phase5-persistence",
            run_id="run-phase5-persistence",
            dataset_version=DATASET_VERSION,
            predictions=predictions,
            result=result,
            split="test",
        )
        session.commit()
        assert first.result_hash == result.result_hash
        assert first.metric_values == pytest.approx(result.metric_map())
        assert len(first.prediction_payloads) == 6

    with Session(engine) as session:
        reloaded = reload_evaluation(session, run_id="run-phase5-persistence")
        rescored = score_stored_run(
            session,
            run_id="run-phase5-persistence",
            harness=harness,
        )
        assert reloaded.result_hash == result.result_hash
        assert reloaded.metric_values == pytest.approx(result.metric_map())
        assert rescored.result_hash == result.result_hash
        assert rescored.metric_map() == pytest.approx(result.metric_map())

        second = persist_evaluation(
            session,
            experiment_id="exp-phase5-persistence",
            run_id="run-phase5-persistence",
            dataset_version=DATASET_VERSION,
            predictions=predictions,
            result=result,
            split="test",
        )
        session.commit()
        assert second == reloaded
