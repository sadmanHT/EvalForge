from __future__ import annotations

import os
import time
from typing import Any

from redis import Redis

from app.worker.queue import QUEUE_KEY, JobState, enqueue, get_job

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def wait_for_terminal(client: Any, job_id: str, timeout: float = 15.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = get_job(client, job_id)
        if record is not None and record.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED}:
            return record
        time.sleep(0.1)
    raise TimeoutError(f"job did not reach a terminal state: {job_id}")


def main() -> None:
    client: Any = Redis.from_url(REDIS_URL, decode_responses=True)
    assert client.ping()
    client.delete(
        QUEUE_KEY,
        "evalforge:job:phase02-smoke-success",
        "evalforge:job:phase02-smoke-failure",
    )

    enqueue(client, "echo", {"value": "phase02"}, job_id="phase02-smoke-success")
    enqueue(client, "fail", {"message": "deliberate smoke failure"}, job_id="phase02-smoke-failure")

    success = wait_for_terminal(client, "phase02-smoke-success")
    failure = wait_for_terminal(client, "phase02-smoke-failure")

    assert success.state is JobState.SUCCEEDED
    assert success.result == {"value": "phase02"}
    assert failure.state is JobState.FAILED
    assert failure.error == "RuntimeError: deliberate smoke failure"

    print("WORKER_SMOKE=PASS")
    print(f"SUCCESS_STATE={success.state.value}")
    print(f"FAILURE_STATE={failure.state.value}")


if __name__ == "__main__":
    main()
