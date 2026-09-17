from __future__ import annotations

import signal
from types import FrameType
from typing import Any

from redis import Redis

from app.config import Settings
from app.inference.finetuned_orchestration import Phase10FineTunedJobHandler
from app.inference.orchestration import Phase6BaselineJobHandler
from app.inference.rag_orchestration import Phase8RAGJobHandler
from app.training.orchestration import Phase9TrainingJobHandler
from app.worker.queue import TaskHandler, process_one

_running = True


def _stop(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    global _running
    _running = False


def build_task_handlers() -> dict[str, TaskHandler]:
    return {
        "phase6_baseline": Phase6BaselineJobHandler(),
        "phase8_rag": Phase8RAGJobHandler(),
        "phase9_training": Phase9TrainingJobHandler(),
        "phase10_finetuned": Phase10FineTunedJobHandler(),
    }


def main() -> None:
    settings = Settings.from_env()
    client: Any = Redis.from_url(settings.redis_url, decode_responses=True)
    task_handlers = build_task_handlers()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while _running:
        process_one(client, timeout=1, task_handlers=task_handlers)


if __name__ == "__main__":
    main()
