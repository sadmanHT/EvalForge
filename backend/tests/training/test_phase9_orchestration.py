from pathlib import Path

import pytest

from app.training.config import load_training_config
from app.training.orchestration import ConnectedPeftTrainingExecutor

ROOT = Path(__file__).resolve().parents[3]


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
