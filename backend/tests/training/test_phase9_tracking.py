from app.training.runtime import TrainingRunResult
from app.training.tracking import DisabledTrainingTracker


def test_disabled_training_tracker_is_explicit() -> None:
    tracker = DisabledTrainingTracker()
    result = TrainingRunResult(
        output_dir="/tmp/out",
        adapter_dir="/tmp/out/adapter",
        adapter_sha256="a" * 64,
        selected_checkpoint=None,
        train_metrics={},
        log_history=(),
        checkpoint_paths=(),
    )
    evidence = tracker.log_training(
        run_id="run",
        config={"x": 1},
        reproducibility_metadata={"git_commit": "abc"},
        result=result,
    )
    assert evidence.provider == "wandb"
    assert evidence.configured is False
    assert evidence.run_reference is None
