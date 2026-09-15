from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("EVALFORGE_ROOT", Path(__file__).resolve().parents[2])).resolve()
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.data.schemas import IncidentRecord, Split  # noqa: E402
from app.inference.base_model import IncidentInput  # noqa: E402
from app.inference.protocol import load_taxonomy  # noqa: E402
from app.training.inference import FineTunedAdapterPipeline, SmokeAdapterBackend  # noqa: E402


def _first_validation_record(dataset_version: str) -> IncidentRecord:
    path = ROOT / "datasets/incident_diagnosis/processed" / dataset_version / "incidents.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        record = IncidentRecord.model_validate_json(line)
        if record.split is Split.VALIDATION:
            return record
    raise RuntimeError("validation split is empty")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fresh-process inference contract for a saved Phase 09 CPU smoke adapter."
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--dataset-version",
        default="evalforge-incident-diagnosis-v0.1.0",
    )
    args = parser.parse_args()

    labels, _categories = load_taxonomy(ROOT)
    backend = SmokeAdapterBackend(args.adapter)
    pipeline = FineTunedAdapterPipeline(backend=backend, allowed_labels=labels)
    record = _first_validation_record(args.dataset_version)
    prediction = pipeline.predict(
        IncidentInput(
            incident_id=record.incident_id,
            title=record.title,
            description=record.description,
        )
    )
    payload = prediction.model_dump(mode="json", exclude_none=False)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    if prediction.parse_status.value != "OK":
        raise RuntimeError(f"smoke adapter parse failed: {prediction.parse_status}")
    if prediction.pipeline_metadata.get("pipeline_type") != "FINETUNED":
        raise RuntimeError("smoke adapter did not traverse the fine-tuned pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
