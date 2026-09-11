#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.db import build_engine, session_scope
from app.evaluation.calibration import CalibrationError, fit_temperature_scaling
from app.evaluation.contracts import EvaluationExample, Prediction
from app.evaluation.harness import EvaluationHarness
from app.evaluation.persistence import reload_evaluation, score_stored_run

RUN_ID = "run-phase5-evaluator-smoke"
EVALUATOR_VERSION = "phase5-evaluator-v1"
REQUIRED_PATHS = (
    "backend/app/evaluation/harness.py",
    "backend/app/evaluation/metrics.py",
    "backend/app/evaluation/calibration.py",
    "backend/app/evaluation/statistics.py",
    "backend/app/evaluation/failure_taxonomy.py",
    "backend/app/evaluation/graders/deterministic.py",
    "backend/app/evaluation/graders/supporting.py",
    "backend/tests/fixtures/phase5/golden_metrics.json",
    "docs/evaluation-methodology.md",
    "evidence/phase-05/golden-fixture-calculations.md",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(message: str) -> None:
    raise SystemExit(f"PHASE05_EXIT_GATE=FAIL {message}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing = [path for path in REQUIRED_PATHS if not (root / path).is_file()]
    if missing:
        _fail(f"required Phase 05 artifacts missing={missing}")

    golden = _load_json(root / "backend/tests/fixtures/phase5/golden_metrics.json")
    examples = [EvaluationExample.model_validate(item) for item in golden["examples"]]
    predictions = [Prediction.model_validate(item) for item in golden["predictions"]]
    harness = EvaluationHarness(
        evaluator_version=EVALUATOR_VERSION,
        allowed_labels=golden["allowed_labels"],
        label_to_category=golden["label_to_category"],
        ece_bins=golden["ece_bins"],
    )
    result = harness.evaluate(
        examples,
        predictions,
        upfront_cost_usd=golden["upfront_cost_usd"],
    )
    expected = golden["expected"]
    actual = result.metric_map()
    for name, expected_value in expected.items():
        if name not in actual or abs(actual[name] - float(expected_value)) > 1e-12:
            _fail(f"golden metric mismatch {name}: actual={actual.get(name)} expected={expected_value}")

    try:
        fit_temperature_scaling(
            [
                {label: 1.0 if index == 0 else 0.0 for index, label in enumerate(golden["allowed_labels"])}
            ],
            [golden["allowed_labels"][0]],
            allowed_labels=golden["allowed_labels"],
            source_split="test",
        )
    except CalibrationError:
        pass
    else:
        _fail("test split was accepted as a calibration fitting source")

    taxonomy = _load_json(root / "configs/label-taxonomy.yaml")
    labels = [item["id"] for item in taxonomy["labels"]]
    categories = {item["id"]: item["category"] for item in taxonomy["labels"]}
    stored_harness = EvaluationHarness(
        evaluator_version=EVALUATOR_VERSION,
        allowed_labels=labels,
        label_to_category=categories,
        ece_bins=10,
    )
    engine = build_engine()
    with session_scope(engine) as session:
        snapshot = reload_evaluation(session, run_id=RUN_ID)
        rescored = score_stored_run(session, run_id=RUN_ID, harness=stored_harness)
    if snapshot.result_hash != rescored.result_hash:
        _fail("persisted smoke evaluation does not rescore to identical result identity")
    if snapshot.metric_values != rescored.metric_map():
        _fail("persisted smoke evaluation metrics changed after reload/rescore")
    if len(snapshot.prediction_payloads) != 6:
        _fail(f"persisted smoke prediction count={len(snapshot.prediction_payloads)} expected=6")

    print("PHASE05_EXIT_GATE=PASS")
    print(f"EVALUATOR_VERSION={EVALUATOR_VERSION}")
    print(f"FAILURE_TAXONOMY_VERSION={rescored.failure_taxonomy_version}")
    print(f"GOLDEN_FIXTURE={golden['fixture_version']}")
    print(f"GOLDEN_RESULT_HASH={result.result_hash}")
    print(f"PERSISTED_SMOKE_RESULT_HASH={snapshot.result_hash}")
    print(f"PERSISTED_PREDICTIONS={len(snapshot.prediction_payloads)}")
    print(f"PERSISTED_METRICS={len(snapshot.metric_values)}")
    print("CALIBRATION_TEST_FIT_REJECTED=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
