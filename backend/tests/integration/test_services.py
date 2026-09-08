from __future__ import annotations

from typing import Any

import psycopg
import pytest
from redis import Redis

from app.config import Settings
from app.worker.queue import QUEUE_KEY, JobState, enqueue, process_one

pytestmark = pytest.mark.integration


def test_postgres_connectivity() -> None:
    settings = Settings.from_env()
    with psycopg.connect(settings.database_url, connect_timeout=3) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            assert cursor.fetchone() == (1,)


def test_redis_connectivity() -> None:
    settings = Settings.from_env()
    client = Redis.from_url(settings.redis_url)
    assert client.ping()


def test_worker_success_and_failure_with_real_redis() -> None:
    settings = Settings.from_env()
    client: Any = Redis.from_url(settings.redis_url, decode_responses=True)
    client.delete(QUEUE_KEY, "evalforge:job:integration-success", "evalforge:job:integration-failure")

    enqueue(client, "echo", {"value": "ok"}, job_id="integration-success")
    succeeded = process_one(client, timeout=1)
    assert succeeded is not None
    assert succeeded.state is JobState.SUCCEEDED
    assert succeeded.result == {"value": "ok"}

    enqueue(client, "fail", {"message": "integration failure"}, job_id="integration-failure")
    failed = process_one(client, timeout=1)
    assert failed is not None
    assert failed.state is JobState.FAILED
    assert failed.error == "RuntimeError: integration failure"
