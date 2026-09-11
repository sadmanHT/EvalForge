from __future__ import annotations

from typing import Any

from fakeredis import FakeRedis

from app.worker.queue import JobState, enqueue, get_job, process_one


def _client() -> Any:
    return FakeRedis(decode_responses=True)


def test_job_state_contract_is_complete_and_stable() -> None:
    assert {state.value for state in JobState} == {
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "CANCELED",
    }


def test_worker_success_path_persists_result() -> None:
    client = _client()
    pending = enqueue(client, "add", {"values": [2, 3, 5]}, job_id="unit-success")
    assert pending.state is JobState.PENDING

    processed = process_one(client, timeout=1)
    assert processed is not None
    assert processed.state is JobState.SUCCEEDED
    assert processed.result == {"sum": 10}
    assert get_job(client, "unit-success") == processed


def test_worker_failure_path_persists_failure_without_crashing() -> None:
    client = _client()
    enqueue(client, "fail", {"message": "expected failure"}, job_id="unit-failure")

    processed = process_one(client, timeout=1)
    assert processed is not None
    assert processed.state is JobState.FAILED
    assert processed.result is None
    assert processed.error == "RuntimeError: expected failure"


def test_worker_dispatches_injected_long_running_handler() -> None:
    client = _client()
    enqueue(client, "phase6_baseline", {"split": "validation"}, job_id="unit-phase6")

    def handler(payload: dict[str, Any]) -> dict[str, object]:
        return {"accepted_split": payload["split"], "worker": True}

    processed = process_one(
        client,
        timeout=1,
        task_handlers={"phase6_baseline": handler},
    )
    assert processed is not None
    assert processed.state is JobState.SUCCEEDED
    assert processed.result == {"accepted_split": "validation", "worker": True}
