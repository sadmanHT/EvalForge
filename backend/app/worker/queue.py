from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

QUEUE_KEY = "evalforge:jobs:queue"
JOB_KEY_PREFIX = "evalforge:job:"


class JobState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


TERMINAL_STATES = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED}


class RedisQueueClient(Protocol):
    def hset(self, name: str, mapping: dict[str, str]) -> Any: ...

    def hgetall(self, name: str) -> dict[str, str]: ...

    def rpush(self, name: str, *values: str) -> Any: ...

    def blpop(self, keys: str, timeout: int = 0) -> tuple[str, str] | None: ...


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    task: str
    payload: dict[str, Any]
    state: JobState
    result: Any | None = None
    error: str | None = None


def _job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


def enqueue(
    client: RedisQueueClient,
    task: str,
    payload: dict[str, Any],
    *,
    job_id: str | None = None,
) -> JobRecord:
    identifier = job_id or str(uuid.uuid4())
    record = JobRecord(identifier, task, payload, JobState.PENDING)
    client.hset(
        _job_key(identifier),
        mapping={
            "job_id": identifier,
            "task": task,
            "payload": json.dumps(payload, sort_keys=True, separators=(",", ":")),
            "state": JobState.PENDING.value,
            "result": "",
            "error": "",
        },
    )
    client.rpush(QUEUE_KEY, identifier)
    return record


def get_job(client: RedisQueueClient, job_id: str) -> JobRecord | None:
    data = client.hgetall(_job_key(job_id))
    if not data:
        return None
    result_text = data.get("result", "")
    return JobRecord(
        job_id=data["job_id"],
        task=data["task"],
        payload=json.loads(data["payload"]),
        state=JobState(data["state"]),
        result=json.loads(result_text) if result_text else None,
        error=data.get("error") or None,
    )


def _execute(task: str, payload: dict[str, Any]) -> Any:
    if task == "echo":
        return payload
    if task == "add":
        values = payload.get("values")
        if not isinstance(values, list) or not all(isinstance(value, (int, float)) for value in values):
            raise ValueError("add task requires a numeric values list")
        return {"sum": sum(values)}
    if task == "fail":
        raise RuntimeError(str(payload.get("message", "deliberate task failure")))
    raise ValueError(f"unknown task: {task}")


def process_one(client: RedisQueueClient, *, timeout: int = 1) -> JobRecord | None:
    queued = client.blpop(QUEUE_KEY, timeout=timeout)
    if queued is None:
        return None
    _, job_id = queued
    current = get_job(client, job_id)
    if current is None:
        return None

    client.hset(_job_key(job_id), mapping={"state": JobState.RUNNING.value})
    try:
        result = _execute(current.task, current.payload)
    except Exception as exc:
        client.hset(
            _job_key(job_id),
            mapping={"state": JobState.FAILED.value, "error": f"{type(exc).__name__}: {exc}"},
        )
    else:
        client.hset(
            _job_key(job_id),
            mapping={
                "state": JobState.SUCCEEDED.value,
                "result": json.dumps(result, sort_keys=True, separators=(",", ":")),
                "error": "",
            },
        )
    return get_job(client, job_id)
