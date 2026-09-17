#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

from app.data.io import read_jsonl
from app.data.schemas import IncidentRecord
from app.inference.costing import HourlyRateCostBackend
from app.inference.finetuned_model import load_phase9_candidate_identity, verify_adapter_locator
from app.inference.finetuned_portable import evaluate_finetuned_records
from app.inference.finetuned_protocol import load_phase10_protocol
from app.inference.protocol import load_taxonomy
from app.training.evidence import tree_sha256
from app.training.inference import FineTunedAdapterPipeline, PeftTransformersBackend

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_VERSION = "phase10-finetuned-portable-run-v1"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_source_identity(expected_commit: str) -> None:
    actual = _git_output("rev-parse", "HEAD")
    if actual != expected_commit:
        raise ValueError(f"--git-commit {expected_commit} does not match HEAD {actual}")
    tracked_status = _git_output("status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise ValueError("Phase 10 connected evaluation requires a clean tracked worktree")


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_environment() -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Phase 10 connected evaluation requires the GPU runtime stack") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 10 scientific evaluation requires CUDA")
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count != 1:
        raise RuntimeError(
            "Phase 10 connected evaluation requires exactly one visible GPU; "
            "set CUDA_VISIBLE_DEVICES before starting Python"
        )
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "visible_gpu_count": visible_gpu_count,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_vram_bytes": int(properties.total_memory),
        "cuda_version": torch.version.cuda,
        "torch": str(torch.__version__),
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "bitsandbytes": _package_version("bitsandbytes"),
        "accelerate": _package_version("accelerate"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Phase 10 fine-tuned arm on a connected single-GPU host "
            "without requiring PostgreSQL. The locked test remains protocol-gated."
        )
    )
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--adapter-locator", required=True, type=Path)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--hardware-runtime-descriptor", required=True)
    parser.add_argument("--cost-rate-snapshot-version", required=True)
    parser.add_argument("--gpu-hour-usd", required=True, type=float)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--upfront-cost-usd", type=float, default=0.0)
    parser.add_argument("--evidence-output", required=True, type=Path)
    args = parser.parse_args()

    _assert_source_identity(args.git_commit)
    protocol = load_phase10_protocol(ROOT)
    evaluation_split = protocol.assert_split_allowed(args.split)
    identity = load_phase9_candidate_identity(ROOT)

    adapter_dir = _resolve(args.adapter_locator).resolve()
    if not adapter_dir.is_dir():
        raise ValueError("Phase 10 connected evaluation requires a local adapter directory")
    verify_adapter_locator(
        str(adapter_dir),
        identity=identity,
        adapter_revision=args.adapter_revision,
    )
    if args.adapter_revision != protocol.candidate_adapter_sha256:
        raise ValueError("adapter revision does not match the frozen Phase 10 candidate")
    observed_adapter_sha256 = tree_sha256(adapter_dir)
    if observed_adapter_sha256 != protocol.candidate_adapter_sha256:
        raise ValueError("local adapter checksum does not match the frozen Phase 10 candidate")

    gpu_environment = _gpu_environment()
    labels, _categories = load_taxonomy(ROOT)
    raw_backend = PeftTransformersBackend(
        protocol.runtime_config,
        adapter_id=str(adapter_dir),
        adapter_revision=args.adapter_revision,
        use_adapter_revision_for_loading=False,
    )
    backend = HourlyRateCostBackend(raw_backend, gpu_hour_usd=args.gpu_hour_usd)
    pipeline = FineTunedAdapterPipeline(
        backend=backend,
        allowed_labels=labels,
        prompt_version=protocol.prompt_version,
        generation_config=protocol.generation_config,
    )

    incident_path = (
        ROOT
        / "datasets/incident_diagnosis/processed"
        / protocol.dataset_version
        / "incidents.jsonl"
    )
    records = read_jsonl(incident_path, IncidentRecord)
    evaluated = evaluate_finetuned_records(
        root=ROOT,
        protocol=protocol,
        pipeline=pipeline,
        records=records,
        split=evaluation_split,
        max_attempts=args.max_attempts,
        upfront_cost_usd=args.upfront_cost_usd,
    )

    evidence: dict[str, Any] = {
        "evidence_version": EVIDENCE_VERSION,
        "status": "completed",
        "run_id": args.run_id,
        "split": evaluated.split,
        "protocol_version": protocol.protocol_version,
        "protocol_state": protocol.state.value,
        "locked_test_authorized": protocol.locked_test_authorized,
        "scientific_config_hash": protocol.scientific_config_hash(),
        "git_commit": args.git_commit,
        "dataset_version": protocol.dataset_version,
        "dataset_manifest_checksum": protocol.test_split_manifest_checksum,
        "base_model_id": protocol.base_model_id,
        "base_model_revision": protocol.base_model_revision,
        "adapter_id": protocol.candidate_adapter_artifact_reference,
        "adapter_revision": protocol.candidate_adapter_sha256,
        "adapter_sha256": observed_adapter_sha256,
        "source_training_run_id": protocol.source_training_run_id,
        "source_training_evidence_sha256": protocol.source_training_evidence_sha256,
        "source_training_config_hash": protocol.source_training_config_hash,
        "candidate_selection_rule": protocol.candidate_selection_rule,
        "evaluator_version": protocol.evaluator_version,
        "output_schema_version": protocol.output_schema_version,
        "prompt_version": protocol.prompt_version,
        "generation_config": protocol.generation_config.model_dump(mode="json"),
        "hardware_runtime_descriptor": args.hardware_runtime_descriptor,
        "gpu_environment": gpu_environment,
        "cost_rate_snapshot_version": args.cost_rate_snapshot_version,
        "gpu_hour_usd": args.gpu_hour_usd,
        "upfront_cost_usd": args.upfront_cost_usd,
        "max_attempts": args.max_attempts,
        "prediction_count": len(evaluated.predictions),
        "inference_failure_count": evaluated.inference_failure_count,
        "result_hash": evaluated.result.result_hash,
        "metric_values": evaluated.result.metric_map(),
        "evaluation": evaluated.result.canonical_payload(),
        "predictions": [
            prediction.model_dump(mode="json", exclude_none=False)
            for prediction in sorted(evaluated.predictions, key=lambda item: item.incident_id)
        ],
        "execution_mode": "portable_connected_gpu_no_database",
        "persistence_stack_used": False,
    }

    output = _resolve(args.evidence_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "split": evaluated.split,
                "prediction_count": len(evaluated.predictions),
                "inference_failure_count": evaluated.inference_failure_count,
                "result_hash": evaluated.result.result_hash,
                "metric_values": evaluated.result.metric_map(),
                "adapter_sha256": observed_adapter_sha256,
                "evidence_output": str(output),
                "locked_test_authorized": protocol.locked_test_authorized,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
