from __future__ import annotations

import signal
from types import FrameType
from typing import Any

from redis import Redis

from app.config import Settings
from app.worker.queue import process_one

_running = True


def _stop(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    global _running
    _running = False


def main() -> None:
    settings = Settings.from_env()
    client: Any = Redis.from_url(settings.redis_url, decode_responses=True)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while _running:
        process_one(client, timeout=1)


if __name__ == "__main__":
    main()
