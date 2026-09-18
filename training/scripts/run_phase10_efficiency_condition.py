#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("EVALFORGE_ROOT", Path(__file__).resolve().parents[2])).resolve()
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.inference.efficiency_study import (  # noqa: E402
    EFFICIENCY_CONDITION_EVIDENCE_VERSION,
    EFFICIENCY_NUMERIC_RECOVERY_POLICY,
    build_efficiency_training_bundle,
    efficiency_condition_id,
    evaluate_efficiency_validation,
    prepare_efficiency_condition,
)
from app.inference.finetuned_protocol import load_phase10_protocol  # noqa: E402
from app.training.evidence import sha256_file  # noqa: E402
from app.training.runtime import PeftTrainingRuntime  # noqa: E402
from app.training.tracking import build_training_tracker_from_env  # noqa: E402


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _gpu_environment() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Phase 10 efficiency runs require exactly one visible CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    metadata: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_vram_bytes": int(props.total_memory),
    }
    for package in ("transformers", "accelerate", "bitsandbytes", "peft", "wandb"):
        module = __import__(package)
        metadata[package] = getattr(module, "__version__", "unknown")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one predeclared Phase 10 data-efficiency training/validation condition."
    )
    parser.add_argument("--fraction", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--hardware-runtime-descriptor", required=True)
    parser.add_argument("--cost-rate-snapshot-version", required=True)
    parser.add_argument("--gpu-hour-usd", required=True, type=float)
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()

    if _git_output("rev-parse", "HEAD") != args.git_commit:
        raise RuntimeError("--git-commit must exactly match HEAD")
    if _git_output("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Phase 10 efficiency run requires a clean tracked worktree")
    if not os.environ.get("WANDB_PROJECT", "").strip():
        raise RuntimeError("Phase 10 efficiency evidence requires WANDB_PROJECT")

    protocol = load_phase10_protocol(ROOT)
    condition_id = efficiency_condition_id(args.fraction, args.seed)
    condition_dir = args.work_dir / condition_id
    prepared_dir = condition_dir / "prepared"
    training_dir = condition_dir / "training"
    evidence_path = condition_dir / "condition-evidence.json"
    if evidence_path.exists():
        raise RuntimeError(f"refusing to overwrite efficiency evidence: {evidence_path}")

    prepared = prepare_efficiency_condition(
        root=ROOT,
        output_dir=prepared_dir,
        protocol=protocol,
        fraction=args.fraction,
        seed=args.seed,
    )
    bundle = build_efficiency_training_bundle(ROOT, seed=args.seed)
    base_bundle = build_efficiency_training_bundle(ROOT, seed=20260908)
    if base_bundle.config_hash() != protocol.source_training_config_hash:
        raise RuntimeError(
            "Phase 10 efficiency base hyperparameters disagree with Phase 09 evidence"
        )
    gpu_environment = _gpu_environment()

    tracker = build_training_tracker_from_env(
        job_type="phase10-data-efficiency",
        artifact_type="phase10-efficiency-peft-adapter",
    )
    started = time.perf_counter()
    result = PeftTrainingRuntime(bundle).train(
        train_path=prepared_dir / "train.jsonl",
        validation_path=prepared_dir / "validation.jsonl",
        output_dir=training_dir,
        report_to_wandb=False,
    )
    training_wall_seconds = time.perf_counter() - started
    reproducibility = {
        "phase": 10,
        "study_role": "secondary_descriptive_no_primary_selection",
        "condition_id": condition_id,
        "fraction": args.fraction,
        "seed": args.seed,
        "phase10_scientific_config_hash": protocol.scientific_config_hash(),
        "dataset_version": protocol.dataset_version,
        "dataset_manifest_checksum": protocol.test_split_manifest_checksum,
        "prepared_manifest_sha256": sha256_file(prepared_dir / "manifest.json"),
        "prepared_train_sha256": prepared.train_sha256,
        "prepared_validation_sha256": prepared.validation_sha256,
        "lineage_sha256": prepared.lineage_sha256,
        "base_phase09_training_config_hash": protocol.source_training_config_hash,
        "efficiency_training_config_hash": bundle.config_hash(),
        "git_commit": args.git_commit,
        "hardware_runtime_descriptor": args.hardware_runtime_descriptor,
        "gpu_environment": gpu_environment,
        "training_wall_seconds": training_wall_seconds,
        "training_numeric_recovery_policy": EFFICIENCY_NUMERIC_RECOVERY_POLICY,
        "nonfinite_gradient_norm_steps": list(result.nonfinite_gradient_norm_steps),
        "nonfinite_gradient_norm_count": len(result.nonfinite_gradient_norm_steps),
        "primary_adapter_selection_use": False,
        "test_split_used": False,
    }
    tracking = tracker.log_training(
        run_id=condition_id,
        config={
            "phase10_efficiency": {
                "fraction": args.fraction,
                "seed": args.seed,
                "study_role": "secondary_descriptive_no_primary_selection",
            },
            "training_bundle": bundle.canonical_payload(),
        },
        reproducibility_metadata=reproducibility,
        result=result,
    )
    if (
        not tracking.configured
        or not tracking.run_reference
        or not tracking.artifact_reference
    ):
        raise RuntimeError("Phase 10 efficiency evidence requires durable W&B references")

    del tracker
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    validation = evaluate_efficiency_validation(
        root=ROOT,
        protocol=protocol,
        adapter_dir=Path(result.adapter_dir),
        adapter_sha256=result.adapter_sha256,
        gpu_hour_usd=args.gpu_hour_usd,
        max_attempts=args.max_attempts,
    )
    locked_test_path = ROOT / "evidence/phase-10/locked-test/phase10-finetuned-test.json"
    if not locked_test_path.is_file():
        raise RuntimeError("secondary efficiency study requires preserved locked-test evidence")

    training_direct_cost_usd = training_wall_seconds / 3600.0 * args.gpu_hour_usd
    evidence = {
        "evidence_version": EFFICIENCY_CONDITION_EVIDENCE_VERSION,
        "status": "completed",
        "study_role": "secondary_descriptive_no_primary_selection",
        "condition_id": condition_id,
        "fraction": args.fraction,
        "seed": args.seed,
        "phase10_scientific_config_hash": protocol.scientific_config_hash(),
        "dataset_version": protocol.dataset_version,
        "dataset_manifest_checksum": protocol.test_split_manifest_checksum,
        "source_git_commit": args.git_commit,
        "base_model_id": protocol.base_model_id,
        "base_model_revision": protocol.base_model_revision,
        "prompt_version": protocol.prompt_version,
        "output_schema_version": protocol.output_schema_version,
        "generation_config": protocol.generation_config.model_dump(mode="json"),
        "evaluator_version": protocol.evaluator_version,
        "subset_manifest": prepared.model_dump(mode="json"),
        "subset_manifest_sha256": sha256_file(prepared_dir / "manifest.json"),
        "base_phase09_training_config_hash": protocol.source_training_config_hash,
        "efficiency_training_config_hash": bundle.config_hash(),
        "base_seed_training_config_hash": base_bundle.config_hash(),
        "adapter_sha256": result.adapter_sha256,
        "selected_checkpoint": result.selected_checkpoint,
        "checkpoint_paths": list(result.checkpoint_paths),
        "train_metrics": result.train_metrics,
        "training_wall_seconds": training_wall_seconds,
        "training_numeric_recovery_policy": EFFICIENCY_NUMERIC_RECOVERY_POLICY,
        "nonfinite_gradient_norm_steps": list(result.nonfinite_gradient_norm_steps),
        "nonfinite_gradient_norm_count": len(result.nonfinite_gradient_norm_steps),
        "cost_rate_snapshot_version": args.cost_rate_snapshot_version,
        "gpu_hour_usd": args.gpu_hour_usd,
        "training_direct_cost_usd": training_direct_cost_usd,
        "training_numeric_recovery_policy": EFFICIENCY_NUMERIC_RECOVERY_POLICY,
        "nonfinite_gradient_norm_steps": list(result.nonfinite_gradient_norm_steps),
        "nonfinite_gradient_norm_count": len(result.nonfinite_gradient_norm_steps),
        "tracking": tracking.as_dict(),
        "gpu_environment": gpu_environment,
        "validation_evaluation": validation,
        "locked_test_evidence_sha256": sha256_file(locked_test_path),
        "primary_adapter_selection_use": False,
        "test_split_used": False,
    }
    condition_dir.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "condition_id": condition_id,
        "fraction": args.fraction,
        "seed": args.seed,
        "adapter_sha256": result.adapter_sha256,
        "validation_result_hash": validation["result_hash"],
        "validation_metrics": validation["metric_values"],
        "training_wall_seconds": training_wall_seconds,
        "training_direct_cost_usd": training_direct_cost_usd,
        "tracking": tracking.as_dict(),
        "evidence_path": str(evidence_path),
        "test_split_used": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
