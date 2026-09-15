from pathlib import Path

from app.training.smoke import SmokeAdapter, run_smoke_training


def test_smoke_train_save_reload_and_schema_inference(tmp_path: Path) -> None:
    result = run_smoke_training(tmp_path / "run", total_steps=6)
    assert result.steps_completed == 6
    assert len(result.losses) == 6
    assert len(result.adapter_sha256) == 64
    adapter = SmokeAdapter.load(Path(result.adapter_path))
    code = adapter.predict_code(1.0)
    assert code in {"disk_exhaustion", "memory_leak"}


def test_smoke_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    partial = run_smoke_training(tmp_path / "partial", total_steps=3)
    resumed = run_smoke_training(
        tmp_path / "resumed",
        total_steps=6,
        resume_from_checkpoint=Path(partial.checkpoint_path),
    )
    uninterrupted = run_smoke_training(tmp_path / "full", total_steps=6)
    assert SmokeAdapter.load(Path(resumed.adapter_path)) == SmokeAdapter.load(
        Path(uninterrupted.adapter_path)
    )
    assert resumed.losses == uninterrupted.losses
