from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.evaluation.contracts import EvaluationExample, Prediction


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/phase5/golden_metrics.json"


@pytest.fixture
def golden_payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def golden_examples(golden_payload: dict[str, Any]) -> list[EvaluationExample]:
    return [EvaluationExample.model_validate(item) for item in golden_payload["examples"]]


@pytest.fixture
def golden_predictions(golden_payload: dict[str, Any]) -> list[Prediction]:
    return [Prediction.model_validate(item) for item in golden_payload["predictions"]]
