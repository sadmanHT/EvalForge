from __future__ import annotations

import pytest

from app.inference.ragas import (
    FAITHFULNESS_PROMPT_VERSION,
    RAGAS_ADAPTER_VERSION,
    RagasFaithfulnessAdapter,
)


class FakeFaithfulnessClient:
    def __init__(self, score: float) -> None:
        self.score_value = score
        self.calls: list[dict[str, object]] = []

    def score(
        self,
        *,
        question: str,
        answer: str,
        contexts: tuple[str, ...],
    ) -> float:
        self.calls.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
            }
        )
        return self.score_value


def _adapter(client: FakeFaithfulnessClient) -> RagasFaithfulnessAdapter:
    return RagasFaithfulnessAdapter(
        client=client,
        evaluator_revision="ragas-faithfulness-fixture-v1",
        model_id="fixture/supporting-judge",
        model_revision="fixture-supporting-judge-v1",
    )


def test_faithfulness_adapter_records_supporting_evaluator_identity() -> None:
    client = FakeFaithfulnessClient(0.75)
    result = _adapter(client).evaluate(
        question="Why did the service fail?",
        answer="A connection pool was exhausted.",
        contexts=("Pool utilization reached the configured maximum.",),
    )

    assert result.score == 0.75
    assert result.status == "ok"
    assert result.adapter_version == RAGAS_ADAPTER_VERSION
    assert result.evaluator_name == "ragas.faithfulness"
    assert result.prompt_version == FAITHFULNESS_PROMPT_VERSION
    assert result.metadata()["model_revision"] == "fixture-supporting-judge-v1"
    assert len(client.calls) == 1


def test_faithfulness_adapter_skips_external_call_when_context_is_empty() -> None:
    client = FakeFaithfulnessClient(0.5)
    result = _adapter(client).evaluate(
        question="Why did the service fail?",
        answer="There was no eligible evidence.",
        contexts=("", "   "),
    )

    assert result.score is None
    assert result.status == "not_applicable_no_context"
    assert client.calls == []


@pytest.mark.parametrize("score", [-0.01, 1.01, float("inf"), float("nan")])
def test_faithfulness_adapter_rejects_invalid_scores(score: float) -> None:
    adapter = _adapter(FakeFaithfulnessClient(score))

    with pytest.raises(ValueError, match="faithfulness score"):
        adapter.evaluate(
            question="Why did the service fail?",
            answer="Because of the retrieved evidence.",
            contexts=("Evidence",),
        )


def test_faithfulness_adapter_requires_versioned_evaluator_identity() -> None:
    client = FakeFaithfulnessClient(0.5)

    with pytest.raises(ValueError, match="model_revision is required"):
        RagasFaithfulnessAdapter(
            client=client,
            evaluator_revision="ragas-fixture-v1",
            model_id="fixture/judge",
            model_revision="",
        )
