from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.training.evidence import (  # noqa: E402
    load_training_evidence_identity,
    tree_sha256,
    write_supporting_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a Phase 09 adapter artifact.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--training-config-hash", required=True)
    parser.add_argument("--dataset-manifest-checksum", required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument("--training-evidence", type=Path)
    parser.add_argument("--evidence-path", type=Path)
    parser.add_argument(
        "--push-hub-repo",
        help="Capability only; actual public release belongs to Phase 10.",
    )
    args = parser.parse_args()

    if (args.training_evidence is None) != (args.evidence_path is None):
        raise ValueError("--training-evidence and --evidence-path must be supplied together")

    identity = None
    if args.training_evidence is not None:
        identity = load_training_evidence_identity(args.training_evidence)
        source_sha256 = tree_sha256(args.adapter_dir)
        if source_sha256 != identity.adapter_sha256:
            raise RuntimeError("adapter directory checksum does not match training evidence")
        expected = {
            "training_config_hash": identity.training_config_hash,
            "dataset_manifest_checksum": identity.dataset_manifest_checksum,
            "base_model_id": identity.base_model_id,
            "base_model_revision": identity.base_model_revision,
        }
        supplied = {
            "training_config_hash": args.training_config_hash,
            "dataset_manifest_checksum": args.dataset_manifest_checksum,
            "base_model_id": args.base_model_id,
            "base_model_revision": args.base_model_revision,
        }
        if supplied != expected:
            raise RuntimeError("export identity does not match connected training evidence")

    if args.destination.exists():
        raise FileExistsError(f"destination already exists: {args.destination}")
    shutil.copytree(args.adapter_dir, args.destination)
    adapter_checksum = tree_sha256(args.destination)
    if identity is not None and adapter_checksum != identity.adapter_sha256:
        raise RuntimeError("copied adapter checksum changed during export")

    manifest = {
        "manifest_version": "phase9-adapter-export-v1",
        "adapter_sha256": adapter_checksum,
        "training_config_hash": args.training_config_hash,
        "dataset_manifest_checksum": args.dataset_manifest_checksum,
        "base_model_id": args.base_model_id,
        "base_model_revision": args.base_model_revision,
    }
    manifest_path = args.destination / "evalforge-adapter-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    export_tree_sha256 = tree_sha256(args.destination)

    if args.push_hub_repo:
        hub = importlib.import_module("huggingface_hub")
        api = hub.HfApi()
        api.upload_folder(
            repo_id=args.push_hub_repo,
            folder_path=str(args.destination),
            repo_type="model",
            commit_message="Upload EvalForge Phase 09 adapter candidate",
        )

    print(json.dumps(manifest, sort_keys=True))
    if (
        identity is not None
        and args.training_evidence is not None
        and args.evidence_path is not None
    ):
        evidence = write_supporting_evidence(
            args.evidence_path,
            evidence_version="phase9-adapter-export-evidence-v1",
            training_evidence_path=args.training_evidence,
            identity=identity,
            details={
                "destination": str(args.destination.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "exported_adapter_sha256": adapter_checksum,
                "export_tree_sha256": export_tree_sha256,
                "dataset_manifest_checksum": args.dataset_manifest_checksum,
                "base_model_id": args.base_model_id,
                "base_model_revision": args.base_model_revision,
                "hub_push_requested": bool(args.push_hub_repo),
            },
        )
        print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
