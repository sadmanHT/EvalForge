from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.worker import runner as worker_runner


def test_phase10_handler_is_registered_in_worker_dispatch(monkeypatch: Any) -> None:
    def handler(payload: dict[str, Any]) -> dict[str, object]:
        return {"payload": payload}

    factory: Callable[[], Callable[[dict[str, Any]], dict[str, object]]] = lambda: handler
    monkeypatch.setattr(worker_runner, "Phase6BaselineJobHandler", factory)
    monkeypatch.setattr(worker_runner, "Phase8RAGJobHandler", factory)
    monkeypatch.setattr(worker_runner, "Phase9TrainingJobHandler", factory)
    monkeypatch.setattr(worker_runner, "Phase10FineTunedJobHandler", factory)

    handlers = worker_runner.build_task_handlers()

    assert set(handlers) == {
        "phase6_baseline",
        "phase8_rag",
        "phase9_training",
        "phase10_finetuned",
    }
    assert handlers["phase10_finetuned"]({"split": "validation"}) == {
        "payload": {"split": "validation"}
    }
