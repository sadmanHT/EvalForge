from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.inference.base_model import IncidentInput
from app.inference.protocol import load_taxonomy
from app.training.inference import FineTunedAdapterPipeline, SmokeAdapterBackend
from app.training.smoke import SmokeAdapter, run_smoke_training

ROOT = Path(__file__).resolve().parents[3]


def test_smoke_train_save_reload_and_schema_inference(tmp_path: Path) -> None:
    result = run_smoke_training(tmp_path / "run", total_steps=6)
    assert result.steps_completed == 6
    assert len(result.losses) == 6
    assert len(result.adapter_sha256) == 64
    backend = SmokeAdapterBackend(Path(result.adapter_path))
    labels, _categories = load_taxonomy(ROOT)
    prediction = FineTunedAdapterPipeline(backend=backend, allowed_labels=labels).predict(
        IncidentInput(
            incident_id="phase9-smoke-inference",
            title="Service memory pressure",
            description="Worker memory usage climbs until the process restarts.",
        )
    )
    assert prediction.parse_status.value == "OK"
    assert prediction.predicted_root_cause_code in labels
    assert prediction.pipeline_metadata["pipeline_type"] == "FINETUNED"


def test_smoke_adapter_reloads_in_fresh_process_through_canonical_pipeline(
    tmp_path: Path,
) -> None:
    result = run_smoke_training(tmp_path / "fresh-process", total_steps=6)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "training/scripts/smoke_inference.py"),
            "--adapter",
            result.adapter_path,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip())
    assert payload["parse_status"] == "OK"
    assert payload["pipeline_metadata"]["pipeline_type"] == "FINETUNED"
    assert payload["pipeline_metadata"]["runtime"]["contract_backend"] == "phase9-cpu-smoke-v1"
    assert payload["pipeline_metadata"]["runtime"]["adapter_revision"] == result.adapter_sha256


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
