from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
from pathlib import Path


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("adapter directory is empty")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        payload = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a Phase 09 adapter artifact.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--training-config-hash", required=True)
    parser.add_argument("--dataset-manifest-checksum", required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument(
        "--push-hub-repo",
        help="Capability only; actual public release belongs to Phase 10.",
    )
    args = parser.parse_args()

    if args.destination.exists():
        raise FileExistsError(f"destination already exists: {args.destination}")
    shutil.copytree(args.adapter_dir, args.destination)
    checksum = tree_sha256(args.destination)
    manifest = {
        "manifest_version": "phase9-adapter-export-v1",
        "adapter_sha256": checksum,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
