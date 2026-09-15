from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("EVALFORGE_ROOT", Path(__file__).resolve().parents[2])).resolve()
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.training.evidence import (  # noqa: E402
    load_training_evidence_identity,
    read_json_object,
    sha256_file,
    write_supporting_evidence,
)


def _resolved_checkpoint_paths(payload: dict[str, object]) -> set[Path]:
    raw = payload.get("checkpoint_paths")
    if not isinstance(raw, list):
        raise ValueError("source training evidence is missing checkpoint_paths")
    paths: set[Path] = set()
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("source checkpoint_paths contains an invalid entry")
        paths.add(Path(value).resolve())
    if not paths:
        raise ValueError("source training evidence has no saved checkpoint")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume Phase 09 training from a recorded checkpoint and emit linked proof."
    )
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--hardware-runtime-descriptor", required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path, required=True)
    parser.add_argument("--source-training-evidence", type=Path, required=True)
    parser.add_argument("--resumed-training-evidence-path", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument(
        "--allow-untracked",
        action="store_true",
        help="Developer-only. A run using this flag cannot produce valid Phase 09 resume evidence.",
    )
    args = parser.parse_args()

    source_payload = read_json_object(args.source_training_evidence)
    source_identity = load_training_evidence_identity(args.source_training_evidence)
    checkpoint = args.resume_from_checkpoint.resolve()
    if checkpoint not in _resolved_checkpoint_paths(source_payload):
        raise RuntimeError("resume checkpoint is not recorded by the source connected training run")
    if not checkpoint.is_dir():
        raise RuntimeError(f"resume checkpoint directory does not exist: {checkpoint}")

    train_script = Path(__file__).with_name("train.py")
    command = [
        sys.executable,
        str(train_script),
        "--prepared-dir",
        str(args.prepared_dir),
        "--output-dir",
        str(args.output_dir),
        "--run-id",
        args.run_id,
        "--git-commit",
        args.git_commit,
        "--hardware-runtime-descriptor",
        args.hardware_runtime_descriptor,
        "--resume-from-checkpoint",
        str(checkpoint),
        "--evidence-path",
        str(args.resumed_training_evidence_path),
    ]
    if args.allow_untracked:
        command.append("--allow-untracked")
    subprocess.run(command, env=dict(os.environ), check=True)

    resumed_payload = read_json_object(args.resumed_training_evidence_path)
    resumed_identity = load_training_evidence_identity(args.resumed_training_evidence_path)
    for field in (
        "training_config_hash",
        "dataset_version",
        "dataset_manifest_checksum",
        "base_model_id",
        "base_model_revision",
    ):
        if getattr(resumed_identity, field) != getattr(source_identity, field):
            raise RuntimeError(f"resumed training changed scientific identity field: {field}")
    recorded_resume = resumed_payload.get("resume_from_checkpoint")
    if not isinstance(recorded_resume, str) or Path(recorded_resume).resolve() != checkpoint:
        raise RuntimeError("resumed training evidence did not record the requested checkpoint")

    evidence = write_supporting_evidence(
        args.evidence_path,
        evidence_version="phase9-resume-v1",
        training_evidence_path=args.source_training_evidence,
        identity=source_identity,
        details={
            "resume_from_checkpoint": str(checkpoint),
            "source_run_id": source_identity.run_id,
            "source_wandb_run_reference": source_identity.wandb_run_reference,
            "resumed_run_id": resumed_identity.run_id,
            "resumed_wandb_run_reference": resumed_identity.wandb_run_reference,
            "resumed_wandb_artifact_reference": resumed_identity.wandb_artifact_reference,
            "resumed_adapter_sha256": resumed_identity.adapter_sha256,
            "resumed_training_evidence_sha256": sha256_file(args.resumed_training_evidence_path),
            "dataset_version": source_identity.dataset_version,
            "dataset_manifest_checksum": source_identity.dataset_manifest_checksum,
        },
    )
    print(json.dumps(evidence, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
