from pathlib import Path

import pytest

from app.training.config import load_training_config
from app.training.formatter import prepare_training_dataset
from app.training.orchestration import ConnectedPeftTrainingExecutor
from app.training.runtime import TrainingRunResult
from app.training.tracking import TrainingTrackingEvidence

ROOT = Path(__file__).resolve().parents[3]
DATASET_VERSION = "evalforge-incident-diagnosis-v0.1.0"


def test_connected_executor_fails_closed_without_wandb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    executor = ConnectedPeftTrainingExecutor(root=ROOT, output_root=tmp_path / "outputs")
    bundle = load_training_config(ROOT)

    with pytest.raises(RuntimeError, match="requires WANDB_PROJECT"):
        executor.execute(
            payload={"job_id": "phase9-real-worker"},
            prepared_dir=tmp_path / "prepared",
            bundle=bundle,
        )


def test_connected_executor_uses_single_explicit_wandb_tracking_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "evalforge-test")
    prepared_dir = tmp_path / "prepared"
    prepare_training_dataset(
        root=ROOT,
        output_dir=prepared_dir,
        dataset_version=DATASET_VERSION,
    )
    captured: dict[str, object] = {}

    class FakeRuntime:
        def __init__(self, _bundle: object) -> None:
            pass

        def train(self, **kwargs: object) -> TrainingRunResult:
            captured["report_to_wandb"] = kwargs["report_to_wandb"]
            output_dir = Path(str(kwargs["output_dir"]))
            adapter_dir = output_dir / "adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            checkpoint = output_dir / "checkpoint-1"
            checkpoint.mkdir()
            return TrainingRunResult(
                output_dir=str(output_dir),
                adapter_dir=str(adapter_dir),
                adapter_sha256="a" * 64,
                selected_checkpoint=str(checkpoint),
                train_metrics={"train_loss": 0.25},
                log_history=({"step": 1, "loss": 0.25},),
                checkpoint_paths=(str(checkpoint),),
            )

    class FakeTracker:
        def log_training(self, **kwargs: object) -> TrainingTrackingEvidence:
            captured["tracked_run_id"] = kwargs["run_id"]
            return TrainingTrackingEvidence(
                provider="wandb",
                configured=True,
                run_reference="https://wandb.invalid/run/phase9",
                artifact_reference="https://wandb.invalid/artifact/phase9",
            )

    monkeypatch.setattr("app.training.orchestration.PeftTrainingRuntime", FakeRuntime)
    monkeypatch.setattr(
        "app.training.orchestration.build_training_tracker_from_env",
        lambda: FakeTracker(),
    )

    executor = ConnectedPeftTrainingExecutor(root=ROOT, output_root=tmp_path / "outputs")
    result = executor.execute(
        payload={
            "job_id": "phase9-real-worker",
            "training_run_id": "phase9-connected-v1",
            "git_commit": "abc123",
            "hardware_runtime_descriptor": "test-cuda-runtime",
        },
        prepared_dir=prepared_dir,
        bundle=load_training_config(ROOT),
    )

    assert captured["report_to_wandb"] is False
    assert captured["tracked_run_id"] == "phase9-connected-v1"
    assert result.scientific_adapter is True
    assert result.metadata["tracking"] == {
        "provider": "wandb",
        "configured": True,
        "run_reference": "https://wandb.invalid/run/phase9",
        "artifact_reference": "https://wandb.invalid/artifact/phase9",
    }
