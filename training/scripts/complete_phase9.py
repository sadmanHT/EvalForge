from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("EVALFORGE_ROOT", Path(__file__).resolve().parents[2])).resolve()
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.training.evidence import read_json_object  # noqa: E402

CANONICAL_EVIDENCE_FILES = (
    "training-run.json",
    "adapter-reload.json",
    "resume.json",
    "resume-training-run.json",
    "adapter-export.json",
)


def _run(script: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(script), *args], cwd=ROOT, env=dict(os.environ), check=True)


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _checkpoint_step(path: str) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.rstrip("/"))
    if match is None:
        raise ValueError(f"cannot determine checkpoint step from path: {path}")
    return int(match.group(1))


def _select_resume_checkpoint(training_payload: dict[str, object]) -> str:
    raw = training_payload.get("checkpoint_paths")
    if not isinstance(raw, list):
        raise ValueError("training evidence is missing checkpoint_paths")
    checkpoints = [value for value in raw if isinstance(value, str) and value.strip()]
    if len(checkpoints) < 2:
        raise RuntimeError(
            "connected completion requires at least two retained checkpoints so resume proves recovery"
        )
    ordered = sorted(checkpoints, key=_checkpoint_step)
    return ordered[0]


def _require_clean_repository(expected_commit: str) -> None:
    actual_commit = _git_output("rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"--git-commit does not match checked-out HEAD: {expected_commit} != {actual_commit}"
        )
    dirty = _git_output("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("connected Phase 09 completion must start from a clean tracked worktree")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the real connected Phase 09 QLoRA completion sequence without touching test data."
    )
    parser.add_argument(
        "--dataset-version",
        default="evalforge-incident-diagnosis-v0.1.0",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-run-id")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--hardware-runtime-descriptor", required=True)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=ROOT / "evidence/phase-09",
    )
    args = parser.parse_args()

    if not os.environ.get("WANDB_PROJECT", "").strip():
        raise RuntimeError("connected Phase 09 completion requires WANDB_PROJECT")
    _require_clean_repository(args.git_commit)

    evidence_dir = args.evidence_dir.resolve()
    existing = [name for name in CANONICAL_EVIDENCE_FILES if (evidence_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing Phase 09 evidence: " + ", ".join(existing)
        )

    work_dir = args.work_dir.resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"work directory must be empty or absent: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = work_dir / "prepared"
    train_dir = work_dir / "train"
    resume_dir = work_dir / "resume"
    export_dir = work_dir / "adapter-export"
    staged_evidence = work_dir / "evidence"
    staged_evidence.mkdir(parents=True, exist_ok=True)

    prepare_script = ROOT / "training/scripts/prepare_dataset.py"
    train_script = ROOT / "training/scripts/train.py"
    reload_script = ROOT / "training/scripts/evaluate_checkpoint.py"
    resume_script = ROOT / "training/scripts/resume.py"
    export_script = ROOT / "training/scripts/export_adapter.py"

    training_evidence = staged_evidence / "training-run.json"
    reload_evidence = staged_evidence / "adapter-reload.json"
    resume_evidence = staged_evidence / "resume.json"
    resumed_training_evidence = staged_evidence / "resume-training-run.json"
    export_evidence = staged_evidence / "adapter-export.json"

    _run(
        prepare_script,
        "--dataset-version",
        args.dataset_version,
        "--output-dir",
        str(prepared_dir),
    )
    _run(
        train_script,
        "--prepared-dir",
        str(prepared_dir),
        "--output-dir",
        str(train_dir),
        "--run-id",
        args.run_id,
        "--git-commit",
        args.git_commit,
        "--hardware-runtime-descriptor",
        args.hardware_runtime_descriptor,
        "--evidence-path",
        str(training_evidence),
    )

    training_payload = read_json_object(training_evidence)
    adapter_dir = training_payload.get("adapter_dir")
    if not isinstance(adapter_dir, str) or not adapter_dir.strip():
        raise RuntimeError("connected training evidence did not record adapter_dir")
    resume_checkpoint = _select_resume_checkpoint(training_payload)

    _run(
        reload_script,
        "--adapter",
        adapter_dir,
        "--adapter-revision",
        "local",
        "--dataset-version",
        args.dataset_version,
        "--training-evidence",
        str(training_evidence),
        "--evidence-path",
        str(reload_evidence),
    )

    resume_run_id = args.resume_run_id or f"{args.run_id}-resume"
    _run(
        resume_script,
        "--prepared-dir",
        str(prepared_dir),
        "--output-dir",
        str(resume_dir),
        "--run-id",
        resume_run_id,
        "--git-commit",
        args.git_commit,
        "--hardware-runtime-descriptor",
        args.hardware_runtime_descriptor,
        "--resume-from-checkpoint",
        resume_checkpoint,
        "--source-training-evidence",
        str(training_evidence),
        "--resumed-training-evidence-path",
        str(resumed_training_evidence),
        "--evidence-path",
        str(resume_evidence),
    )

    for key in (
        "training_config_hash",
        "dataset_manifest_checksum",
        "base_model_id",
        "base_model_revision",
    ):
        value = training_payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"connected training evidence is missing {key}")

    _run(
        export_script,
        "--adapter-dir",
        adapter_dir,
        "--destination",
        str(export_dir),
        "--training-config-hash",
        str(training_payload["training_config_hash"]),
        "--dataset-manifest-checksum",
        str(training_payload["dataset_manifest_checksum"]),
        "--base-model-id",
        str(training_payload["base_model_id"]),
        "--base-model-revision",
        str(training_payload["base_model_revision"]),
        "--training-evidence",
        str(training_evidence),
        "--evidence-path",
        str(export_evidence),
    )

    evidence_dir.mkdir(parents=True, exist_ok=True)
    staged = {
        "training-run.json": training_evidence,
        "adapter-reload.json": reload_evidence,
        "resume.json": resume_evidence,
        "resume-training-run.json": resumed_training_evidence,
        "adapter-export.json": export_evidence,
    }
    copied: list[Path] = []
    try:
        for filename, source in staged.items():
            destination = evidence_dir / filename
            shutil.copy2(source, destination)
            copied.append(destination)
        _run(ROOT / "scripts/check_phase9_exit.py")
    except Exception:
        for path in copied:
            path.unlink(missing_ok=True)
        raise

    summary = {
        "status": "pass",
        "phase": 9,
        "git_commit": args.git_commit,
        "run_id": args.run_id,
        "resume_run_id": resume_run_id,
        "resume_checkpoint": resume_checkpoint,
        "evidence_dir": str(evidence_dir),
        "work_dir": str(work_dir),
        "next_required_gates": ["make verify-all", "make fresh-smoke"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
