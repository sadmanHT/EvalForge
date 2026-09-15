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
from app.inference.base_model import IncidentInput, RuntimeConfig  # noqa: E402
from app.inference.protocol import load_taxonomy  # noqa: E402
from app.training.config import load_training_config  # noqa: E402
from app.training.evidence import (  # noqa: E402
    load_training_evidence_identity,
    tree_sha256,
    write_supporting_evidence,
)
from app.training.inference import (  # noqa: E402
    FineTunedAdapterPipeline,
    PeftTransformersBackend,
)


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
    parser.add_argument("--training-evidence", type=Path)
    parser.add_argument("--evidence-path", type=Path)
    args = parser.parse_args()

    if (args.training_evidence is None) != (args.evidence_path is None):
        raise ValueError("--training-evidence and --evidence-path must be supplied together")

    bundle = load_training_config(ROOT)
    identity = None
    if args.training_evidence is not None:
        identity = load_training_evidence_identity(args.training_evidence)
        if identity.training_config_hash != bundle.config_hash():
            raise RuntimeError("training evidence config hash does not match repository config")
        if identity.base_model_id != bundle.lora.base_model_id:
            raise RuntimeError("training evidence changed the frozen base model ID")
        if identity.base_model_revision != bundle.lora.base_model_revision:
            raise RuntimeError("training evidence changed the frozen base model revision")
        if identity.dataset_version != args.dataset_version:
            raise RuntimeError("reload proof dataset version must match training evidence")
        adapter_path = Path(args.adapter)
        if not adapter_path.is_dir():
            raise RuntimeError("Phase 09 reload evidence requires a local adapter directory")
        observed_adapter_sha256 = tree_sha256(adapter_path)
        if observed_adapter_sha256 != identity.adapter_sha256:
            raise RuntimeError("adapter checksum does not match connected training evidence")

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
        raise RuntimeError(f"adapter validation smoke parse failed: {prediction.parse_status}")
    pipeline_type = prediction.pipeline_metadata.get("pipeline_type")
    if pipeline_type != "FINETUNED":
        raise RuntimeError("adapter validation smoke did not use the fine-tuned pipeline contract")

    if identity is not None and args.training_evidence is not None and args.evidence_path is not None:
        evidence = write_supporting_evidence(
            args.evidence_path,
            evidence_version="phase9-adapter-reload-v1",
            training_evidence_path=args.training_evidence,
            identity=identity,
            details={
                "validation_split": "validation",
                "validation_incident_id": record.incident_id,
                "parse_status": prediction.parse_status.value,
                "predicted_root_cause_code": prediction.predicted_root_cause_code,
                "pipeline_type": pipeline_type,
                "adapter_id": args.adapter,
                "adapter_revision": args.adapter_revision,
                "base_model_id": bundle.lora.base_model_id,
                "base_model_revision": bundle.lora.base_model_revision,
                "dataset_version": args.dataset_version,
            },
        )
        print(json.dumps(evidence, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
