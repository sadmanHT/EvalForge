from __future__ import annotations

from types import SimpleNamespace

from app.training.runtime import TrainingRunResult
from app.training.tracking import DisabledTrainingTracker, WandbTrainingTracker


def _result() -> TrainingRunResult:
    return TrainingRunResult(
        output_dir="/tmp/out",
        adapter_dir="/tmp/out/adapter",
        adapter_sha256="a" * 64,
        selected_checkpoint=None,
        train_metrics={"train_loss": 0.25},
        log_history=({"step": 1, "loss": 0.5}, {"step": 2, "eval_loss": 0.4}),
        checkpoint_paths=(),
    )


def test_disabled_training_tracker_is_explicit() -> None:
    tracker = DisabledTrainingTracker()
    evidence = tracker.log_training(
        run_id="run",
        config={"x": 1},
        reproducibility_metadata={"git_commit": "abc"},
        result=_result(),
    )
    assert evidence.provider == "wandb"
    assert evidence.configured is False
    assert evidence.run_reference is None


def test_wandb_tracker_contract_with_network_mock(monkeypatch) -> None:
    logged_metrics: list[tuple[dict[str, float], int | None]] = []
    summary: dict[str, object] = {}
    artifact_metadata: dict[str, object] = {}

    class FakeArtifact:
        def __init__(self, *, name: str, type: str, metadata: dict[str, object]) -> None:
            del name, type
            artifact_metadata.update(metadata)

        def add_dir(self, path: str, *, name: str) -> None:
            del path, name

    class FakeRun:
        url = "https://wandb.invalid/run/mock"

        def __init__(self) -> None:
            self.summary = summary

        def log(self, metrics: dict[str, float], step: int | None = None) -> None:
            logged_metrics.append((metrics, step))

        def log_artifact(self, artifact: FakeArtifact):
            del artifact
            return SimpleNamespace(url="https://wandb.invalid/artifact/mock")

        def finish(self) -> None:
            return None

    fake_wandb = SimpleNamespace(
        init=lambda **_kwargs: FakeRun(),
        Artifact=FakeArtifact,
    )
    monkeypatch.setattr(
        "app.training.tracking.importlib.import_module",
        lambda name: fake_wandb if name == "wandb" else None,
    )

    tracker = WandbTrainingTracker(project="evalforge")
    evidence = tracker.log_training(
        run_id="phase9-mock",
        config={"rank": 16},
        reproducibility_metadata={"git_commit": "abc", "dataset_version": "v1"},
        result=_result(),
    )

    assert evidence.configured is True
    assert evidence.run_reference == "https://wandb.invalid/run/mock"
    assert evidence.artifact_reference == "https://wandb.invalid/artifact/mock"
    assert logged_metrics == [
        ({"training/loss": 0.5}, 1),
        ({"training/eval_loss": 0.4}, 2),
    ]
    assert summary["git_commit"] == "abc"
    assert summary["adapter_sha256"] == "a" * 64
    assert artifact_metadata["adapter_sha256"] == "a" * 64
