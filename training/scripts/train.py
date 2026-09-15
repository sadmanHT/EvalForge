from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(os.environ.get("EVALFORGE_ROOT", Path(__file__).resolve().parents[2])).resolve()
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.training.config import load_training_config
from app.training.formatter import PreparedDatasetManifest
from app.training.runtime import PeftTrainingRuntime
from app.training.tracking import build_training_tracker_from_env


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            metadata["gpu_name"] = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            metadata["gpu_vram_bytes"] = int(properties.total_memory)
    except ImportError:
        metadata["torch"] = None
        metadata["cuda_available"] = False
    for package in ("transformers", "accelerate", "bitsandbytes", "peft", "wandb"):
        try:
            module = __import__(package)
            metadata[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            metadata[package] = None
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the connected Phase 09 PEFT training job.")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--hardware-runtime-descriptor", required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument(
        "--allow-untracked",
        action="store_true",
        help="Developer-only: allow a local run without W&B. Never use for Phase 09 exit evidence.",
    )
    args = parser.parse_args()

    bundle = load_training_config(ROOT)
    prepared_manifest = PreparedDatasetManifest.model_validate_json(
        (args.prepared_dir / "manifest.json").read_text(encoding="utf-8")
    )
    tracker = build_training_tracker_from_env()
    result = PeftTrainingRuntime(bundle).train(
        train_path=args.prepared_dir / "train.jsonl",
        validation_path=args.prepared_dir / "validation.jsonl",
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        report_to_wandb=bool(os.environ.get("WANDB_PROJECT", "").strip()),
    )
    reproducibility = {
        "training_config_hash": bundle.config_hash(),
        "dataset_version": prepared_manifest.dataset_version,
        "dataset_manifest_checksum": prepared_manifest.dataset_manifest_checksum,
        "dataset_content_checksum": prepared_manifest.dataset_content_checksum,
        "prepared_train_sha256": prepared_manifest.train_sha256,
        "prepared_validation_sha256": prepared_manifest.validation_sha256,
        "lineage_sha256": prepared_manifest.lineage_sha256,
        "base_model_id": bundle.lora.base_model_id,
        "base_model_revision": bundle.lora.base_model_revision,
        "seed": bundle.training.seed,
        "git_commit": args.git_commit,
        "hardware_runtime_descriptor": args.hardware_runtime_descriptor,
        "gpu_environment": _environment_metadata(),
        "gpu_requirements_sha256": _sha256(ROOT / "training/requirements.gpu.txt"),
    }
    tracking = tracker.log_training(
        run_id=args.run_id,
        config=bundle.canonical_payload(),
        reproducibility_metadata=reproducibility,
        result=result,
    )
    if not tracking.configured and not args.allow_untracked:
        raise RuntimeError(
            "Phase 09 connected training requires WANDB_PROJECT; "
            "use --allow-untracked only for non-evidence developer trials"
        )

    evidence = {
        "evidence_version": "phase9-training-run-v1",
        "run_id": args.run_id,
        "status": "completed",
        **reproducibility,
        "adapter_dir": result.adapter_dir,
        "adapter_sha256": result.adapter_sha256,
        "selected_checkpoint": result.selected_checkpoint,
        "checkpoint_paths": list(result.checkpoint_paths),
        "train_metrics": result.train_metrics,
        "tracking": tracking.as_dict(),
    }
    args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
