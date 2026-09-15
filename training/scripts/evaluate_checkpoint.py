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

from app.data.schemas import IncidentRecord, Split
from app.inference.base_model import IncidentInput, RuntimeConfig, ZeroShotBaselineAdapter
from app.inference.protocol import load_taxonomy
from app.training.config import load_training_config
from app.training.inference import PeftTransformersBackend


def _first_validation_record(dataset_version: str) -> IncidentRecord:
    path = ROOT / "datasets/incident_diagnosis/processed" / dataset_version / "incidents.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        record = IncidentRecord.model_validate_json(line)
        if record.split is Split.VALIDATION:
            return record
    raise RuntimeError("validation split is empty")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load a Phase 09 adapter and run one validation-only schema smoke."
    )
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-revision", default="local")
    parser.add_argument(
        "--dataset-version",
        default="evalforge-incident-diagnosis-v0.1.0",
    )
    args = parser.parse_args()

    bundle = load_training_config(ROOT)
    labels, _categories = load_taxonomy(ROOT)
    runtime = RuntimeConfig(
        model_id=bundle.lora.base_model_id,
        revision=bundle.lora.base_model_revision,
        seed=bundle.training.seed,
        device="auto",
        dtype=bundle.lora.compute_dtype,
        quantization=bundle.lora.quantization,
        bnb_4bit_use_double_quant=bundle.lora.use_double_quant,
    )
    backend = PeftTransformersBackend(
        runtime,
        adapter_id=args.adapter,
        adapter_revision=args.adapter_revision,
    )
    pipeline = ZeroShotBaselineAdapter(backend=backend, allowed_labels=labels)
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
        raise RuntimeError(f"adapter validation smoke parse failed: {prediction.parse_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
