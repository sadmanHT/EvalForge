from __future__ import annotations

from typing import Any

import pytest

from app.inference import tracking
from app.inference.tracking import DisabledTracker, WandbTracker, build_tracker_from_env


class FakeLoggedArtifact:
    url = "https://wandb.example/artifact/phase6-result"


class FakeRun:
    id = "wandb-run-1"
    url = "https://wandb.example/run/wandb-run-1"

    def __init__(self) -> None:
        self.logged: list[dict[str, float]] = []
        self.finished = False

    def log(self, payload: dict[str, float]) -> None:
        self.logged.append(payload)

    def log_artifact(self, artifact: Any) -> FakeLoggedArtifact:
        assert artifact["type"] == "evaluation-result"
        return FakeLoggedArtifact()

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_kwargs: dict[str, Any] | None = None

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_kwargs = kwargs
        return self.run

    @staticmethod
    def Artifact(*, name: str, type: str, metadata: dict[str, object]) -> dict[str, object]:
        return {"name": name, "type": type, "metadata": metadata}


def test_tracking_is_disabled_when_wandb_is_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    tracker = build_tracker_from_env()
    assert isinstance(tracker, DisabledTracker)
    evidence = tracker.log_baseline(run_id="run-1", protocol_payload={}, summary={})
    assert evidence.configured is False
    assert evidence.run_reference is None


def test_wandb_tracker_mirrors_metrics_and_returns_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeWandb()
    monkeypatch.setattr(tracking.importlib, "import_module", lambda name: fake)
    tracker = WandbTracker(project="evalforge", entity="research")
    evidence = tracker.log_baseline(
        run_id="run-phase6",
        protocol_payload={"protocol_version": "phase6-zero-shot-baseline-v1"},
        summary={
            "result_hash": "a" * 64,
            "metric_values": {"primary.exact_accuracy": 0.5},
        },
    )
    assert evidence.configured is True
    assert evidence.run_reference == fake.run.url
    assert evidence.artifact_reference == FakeLoggedArtifact.url
    assert fake.run.logged == [{"metrics/primary.exact_accuracy": 0.5}]
    assert fake.run.finished is True
    assert fake.init_kwargs is not None
    assert fake.init_kwargs["job_type"] == "phase6-zero-shot-baseline"
