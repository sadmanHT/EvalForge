from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

RAGAS_ADAPTER_VERSION = "phase8-ragas-adapter-v1"
FAITHFULNESS_PROMPT_VERSION = "ragas-faithfulness-default-v1"


class FaithfulnessClient(Protocol):
    def score(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[str],
    ) -> float: ...


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float | None
    status: str
    adapter_version: str
    evaluator_name: str
    evaluator_revision: str
    model_id: str
    model_revision: str
    prompt_version: str

    def metadata(self) -> dict[str, object]:
        return {
            "status": self.status,
            "adapter_version": self.adapter_version,
            "evaluator_name": self.evaluator_name,
            "evaluator_revision": self.evaluator_revision,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_version": self.prompt_version,
        }


class RagasFaithfulnessAdapter:
    """Thin Phase 08 boundary around an external RAGAS-compatible faithfulness client.

    The client is injected so CI can use a local fake without making network calls. Real
    experiment evidence must record the exact RAGAS/evaluator/model revisions supplied here.
    """

    def __init__(
        self,
        *,
        client: FaithfulnessClient,
        evaluator_revision: str,
        model_id: str,
        model_revision: str,
        prompt_version: str = FAITHFULNESS_PROMPT_VERSION,
    ) -> None:
        for name, value in (
            ("evaluator_revision", evaluator_revision),
            ("model_id", model_id),
            ("model_revision", model_revision),
            ("prompt_version", prompt_version),
        ):
            if not value.strip():
                raise ValueError(f"{name} is required")
        self.client = client
        self.evaluator_revision = evaluator_revision
        self.model_id = model_id
        self.model_revision = model_revision
        self.prompt_version = prompt_version

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[str],
    ) -> FaithfulnessResult:
        if not question.strip():
            raise ValueError("question is required")
        if not answer.strip():
            raise ValueError("answer is required")
        normalized_contexts = tuple(context.strip() for context in contexts if context.strip())
        if not normalized_contexts:
            return FaithfulnessResult(
                score=None,
                status="not_applicable_no_context",
                adapter_version=RAGAS_ADAPTER_VERSION,
                evaluator_name="ragas.faithfulness",
                evaluator_revision=self.evaluator_revision,
                model_id=self.model_id,
                model_revision=self.model_revision,
                prompt_version=self.prompt_version,
            )
        score = float(
            self.client.score(
                question=question.strip(),
                answer=answer.strip(),
                contexts=normalized_contexts,
            )
        )
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("faithfulness score must be finite and in [0,1]")
        return FaithfulnessResult(
            score=score,
            status="ok",
            adapter_version=RAGAS_ADAPTER_VERSION,
            evaluator_name="ragas.faithfulness",
            evaluator_revision=self.evaluator_revision,
            model_id=self.model_id,
            model_revision=self.model_revision,
            prompt_version=self.prompt_version,
        )
