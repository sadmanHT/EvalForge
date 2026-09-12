from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import build_engine, sqlalchemy_database_url
from app.evaluation.contracts import ParseStatus
from app.evaluation.persistence import load_canonical_predictions, reload_evaluation
from app.inference.base_model import (
    BackendOutput,
    GenerationConfig,
    InferenceError,
    RuntimeConfig,
    ZeroShotBaselineAdapter,
)
from app.inference.evidence import export_phase6_run_evidence, validate_run_evidence
from app.inference.protocol import load_baseline_protocol, load_taxonomy
from app.inference.runner import run_baseline_experiment
from app.models import CostRecord, Incident
from app.services.dataset_import import import_phase3_dataset

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
DATASET_VERSION = "evalforge-incident-diagnosis-v0.1.0"


class FixtureBackend:
    def __init__(self, labels: tuple[str, ...], fail_incident_marker: str) -> None:
        protocol = load_baseline_protocol(ROOT)
        self._runtime_config = protocol.runtime_config
        self.labels = labels
        self.fail_incident_marker = fail_incident_marker
        self.calls_by_prompt: dict[str, int] = {}

    @property
    def runtime_config(self) -> RuntimeConfig:
        return self._runtime_config

    def generate(
        self,
        *,
        prompt: str,
        allowed_labels: tuple[str, ...],
        generation_config: GenerationConfig,
    ) -> BackendOutput:
        assert allowed_labels == self.labels
        self.calls_by_prompt[prompt] = self.calls_by_prompt.get(prompt, 0) + 1
        if self.fail_incident_marker in prompt:
            raise InferenceError("fixture per-example inference failure")
        label = next(label for label in self.labels if label in prompt)
        scores = {candidate: (-0.1 if candidate == label else -4.0) for candidate in self.labels}
        return BackendOutput(
            raw_text=f'{{"root_cause_code":"{label}","reasoning":"fixture"}}',
            label_log_likelihoods=scores,
            input_tokens=100,
            output_tokens=12,
            latency_ms=20.0,
            cost_usd=0.001,
            runtime_metadata={"fixture": "phase6-runner"},
        )


def _alembic_config() -> Config:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "migrations"))
    config.set_main_option(
        "sqlalchemy.url", sqlalchemy_database_url(Settings.from_env().database_url)
    )
    return config


@pytest.fixture(scope="module")
def engine() -> Engine:
    engine = build_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    command.upgrade(_alembic_config(), "head")
    with Session(engine) as session:
        import_phase3_dataset(session, ROOT, DATASET_VERSION)
        session.commit()
    yield engine
    engine.dispose()


def test_validation_runner_persists_every_incident_and_recomputes_metrics(
    engine: Engine,
) -> None:
    protocol = load_baseline_protocol(ROOT)
    labels, _categories = load_taxonomy(ROOT)
    with Session(engine) as session:
        validation_incidents = session.scalars(
            select(Incident)
            .where(
                Incident.dataset_version == DATASET_VERSION,
                Incident.split == "validation",
            )
            .order_by(Incident.incident_id)
        ).all()
        assert len(validation_incidents) == 6
        failure_marker = validation_incidents[0].title

    backend = FixtureBackend(labels, fail_incident_marker=failure_marker)
    adapter = ZeroShotBaselineAdapter(
        backend=backend,
        allowed_labels=labels,
        prompt_version=protocol.prompt_version,
        generation_config=protocol.generation_config,
    )
    run_kwargs = {
        "root": ROOT,
        "protocol": protocol,
        "adapter": adapter,
        "split": "validation",
        "git_commit": "phase6-integration",
        "hardware_runtime_descriptor": "phase6-ci-fixture-no-real-model",
        "cost_rate_snapshot_version": "phase6-ci-fixture-rates-v1",
        "experiment_id": "exp-phase6-runner-fixture",
        "run_id": "run-phase6-runner-fixture",
        "max_attempts": 2,
    }
    with Session(engine) as session:
        summary = run_baseline_experiment(session, **run_kwargs)
        session.commit()

    assert summary.prediction_count == 6
    assert summary.inference_failure_count == 1
    assert summary.cost_record_count == 5
    assert summary.recomputation_verified is True
    assert "primary.exact_accuracy" in summary.metric_values
    first_call_count = sum(backend.calls_by_prompt.values())
    assert first_call_count == 7

    with Session(engine) as session:
        predictions = load_canonical_predictions(session, run_id=summary.run_id)
        snapshot = reload_evaluation(session, run_id=summary.run_id)
        costs = session.scalars(select(CostRecord).where(CostRecord.run_id == summary.run_id)).all()
    assert len(predictions) == 6
    assert len({prediction.incident_id for prediction in predictions}) == 6
    invalid_json_count = sum(
        prediction.parse_status is ParseStatus.INVALID_JSON for prediction in predictions
    )
    assert invalid_json_count == 1
    assert len(costs) == 5
    assert {row.cost_rate_snapshot_version for row in costs} == {"phase6-ci-fixture-rates-v1"}
    assert snapshot.result_hash == summary.result_hash
    assert snapshot.metric_values == pytest.approx(summary.metric_values)

    with Session(engine) as session:
        reused = run_baseline_experiment(session, **run_kwargs)
        session.commit()
    assert reused == summary
    assert sum(backend.calls_by_prompt.values()) == first_call_count


def test_completed_validation_run_exports_sealed_recomputable_evidence(engine: Engine) -> None:
    protocol = load_baseline_protocol(ROOT)
    labels, _categories = load_taxonomy(ROOT)
    backend = FixtureBackend(labels, fail_incident_marker="marker-not-present")
    adapter = ZeroShotBaselineAdapter(
        backend=backend,
        allowed_labels=labels,
        prompt_version=protocol.prompt_version,
        generation_config=protocol.generation_config,
    )
    with Session(engine) as session:
        summary = run_baseline_experiment(
            session,
            root=ROOT,
            protocol=protocol,
            adapter=adapter,
            split="validation",
            git_commit="phase6-evidence-integration",
            hardware_runtime_descriptor="phase6-ci-evidence-fixture-no-real-model",
            cost_rate_snapshot_version="phase6-ci-evidence-rates-v1",
            experiment_id="exp-phase6-evidence-fixture",
            run_id="run-phase6-evidence-fixture",
            max_attempts=2,
            upfront_cost_usd=0.06,
        )
        session.commit()

    assert summary.prediction_count == 6
    assert summary.inference_failure_count == 0
    assert summary.cost_record_count == 6
    assert summary.metric_values["cost.amortized_mean_usd"] == pytest.approx(0.011)
    assert summary.metric_values["cost.marginal_mean_usd"] == pytest.approx(0.001)

    with Session(engine) as session:
        evidence = export_phase6_run_evidence(
            session,
            root=ROOT,
            run_id=summary.run_id,
            protocol=protocol,
        )
    validate_run_evidence(
        evidence,
        root=ROOT,
        protocol=protocol,
        expected_split="validation",
    )
    assert evidence["stored_prediction_count"] == 6
    assert evidence["stored_cost_record_count"] == 6
    assert evidence["upfront_cost_usd"] == pytest.approx(0.06)
    assert evidence["result_hash"] == summary.result_hash
    assert len(str(evidence["evidence_sha256"])) == 64


def test_runner_refuses_locked_test_before_protocol_freeze(engine: Engine) -> None:
    protocol = load_baseline_protocol(ROOT)
    labels, _categories = load_taxonomy(ROOT)
    adapter = ZeroShotBaselineAdapter(
        backend=FixtureBackend(labels, fail_incident_marker="marker-not-present"),
        allowed_labels=labels,
        prompt_version=protocol.prompt_version,
        generation_config=protocol.generation_config,
    )
    with (
        Session(engine) as session,
        pytest.raises(ValueError, match="locked test"),
    ):
        run_baseline_experiment(
            session,
            root=ROOT,
            protocol=protocol,
            adapter=adapter,
            split="test",
            git_commit="phase6-integration",
            hardware_runtime_descriptor="phase6-ci-fixture-no-real-model",
            cost_rate_snapshot_version="phase6-ci-fixture-rates-v1",
        )
