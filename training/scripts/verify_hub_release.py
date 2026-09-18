#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("EVALFORGE_ROOT", Path(__file__).resolve().parents[2])).resolve()
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.data.io import read_jsonl  # noqa: E402
from app.data.schemas import IncidentRecord, Split  # noqa: E402
from app.inference.base_model import IncidentInput  # noqa: E402
from app.inference.finetuned_protocol import load_phase10_protocol  # noqa: E402
from app.inference.protocol import load_taxonomy  # noqa: E402
from app.training.evidence import sha256_file, tree_sha256  # noqa: E402
from app.training.hub_release import (  # noqa: E402
    PHASE10_HUB_SMOKE_EVIDENCE_VERSION,
    validate_release_revision,
)
from app.training.inference import FineTunedAdapterPipeline, PeftTransformersBackend  # noqa: E402

SMOKE_VALIDATION_INCIDENT_ID = "incident-215133387692e86f737df1e9"


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean-download an immutable Phase 10 Hub revision and run validation smoke."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--evidence-output", required=True, type=Path)
    args = parser.parse_args()

    revision = validate_release_revision(args.revision)
    if _git_output("rev-parse", "HEAD") != args.git_commit:
        raise RuntimeError("--git-commit must exactly match HEAD")
    if _git_output("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Hub verification requires a clean tracked worktree")
    if args.evidence_output.exists():
        raise RuntimeError(f"refusing to overwrite Hub smoke evidence: {args.evidence_output}")
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise RuntimeError("Hub smoke work directory must be empty")
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)

    try:
        import torch
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Hub smoke requires torch and huggingface_hub") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Hub smoke requires exactly one visible CUDA GPU")

    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN", ""
    ).strip() or None
    api = HfApi(token=token)
    info = api.model_info(args.repo_id, revision=revision, token=token)
    if getattr(info, "sha", None) != revision:
        raise RuntimeError("requested Hub revision did not resolve immutably")
    if getattr(info, "private", None) is not False:
        raise RuntimeError("Phase 10 Hub artifact must be publicly readable")

    snapshot_root = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            revision=revision,
            local_dir=args.work_dir / "snapshot",
            token=token,
        )
    ).resolve()
    adapter_dir = snapshot_root / "adapter"
    protocol = load_phase10_protocol(ROOT)
    observed_sha = tree_sha256(adapter_dir)
    if observed_sha != protocol.candidate_adapter_sha256:
        raise RuntimeError("clean-downloaded Hub adapter hash disagrees with frozen candidate")

    labels, _categories = load_taxonomy(ROOT)
    backend = PeftTransformersBackend(
        protocol.runtime_config,
        adapter_id=str(adapter_dir),
        adapter_revision=observed_sha,
        use_adapter_revision_for_loading=False,
    )
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
    matches = [
        record
        for record in records
        if record.split is Split.VALIDATION
        and record.incident_id == SMOKE_VALIDATION_INCIDENT_ID
    ]
    if len(matches) != 1:
        raise RuntimeError("frozen Hub smoke validation incident is unavailable")
    record = matches[0]
    prediction = pipeline.predict(
        IncidentInput(
            incident_id=record.incident_id,
            title=record.title,
            description=record.description,
        )
    )
    if prediction.parse_status.value != "OK":
        raise RuntimeError(
            f"Hub smoke inference did not produce valid JSON: {prediction.parse_status.value}"
        )
    if prediction.pipeline_metadata.get("pipeline_type") != "FINETUNED":
        raise RuntimeError("Hub smoke inference did not use the FINETUNED pipeline")
    if prediction.retrieved_chunk_ids:
        raise RuntimeError("Hub smoke FINETUNED prediction unexpectedly used retrieval")

    props = torch.cuda.get_device_properties(0)
    evidence = {
        "evidence_version": PHASE10_HUB_SMOKE_EVIDENCE_VERSION,
        "status": "completed",
        "repo_id": args.repo_id,
        "revision": revision,
        "public": True,
        "source_git_commit": args.git_commit,
        "phase10_scientific_config_hash": protocol.scientific_config_hash(),
        "base_model_id": protocol.base_model_id,
        "base_model_revision": protocol.base_model_revision,
        "adapter_sha256": observed_sha,
        "readme_sha256": sha256_file(snapshot_root / "README.md"),
        "smoke_split": "validation",
        "smoke_incident_id": record.incident_id,
        "parse_status": prediction.parse_status.value,
        "predicted_root_cause_code": prediction.predicted_root_cause_code,
        "pipeline_type": prediction.pipeline_metadata.get("pipeline_type"),
        "retrieved_chunk_ids": list(prediction.retrieved_chunk_ids),
        "gpu_environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "visible_gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_vram_bytes": int(props.total_memory),
            "transformers": _version("transformers"),
            "peft": _version("peft"),
            "bitsandbytes": _version("bitsandbytes"),
            "huggingface_hub": _version("huggingface_hub"),
        },
        "locked_test_split_used": False,
    }
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PHASE10_HUGGINGFACE_CLEAN_SMOKE=PASS")
    print(f"HF_REPO_ID={args.repo_id}")
    print(f"HF_REVISION={revision}")
    print(f"ADAPTER_SHA256={observed_sha}")
    print(f"SMOKE_INCIDENT_ID={record.incident_id}")
    print("LOCKED_TEST_SPLIT_USED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
