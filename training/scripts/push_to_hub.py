#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("EVALFORGE_ROOT", Path(__file__).resolve().parents[2])).resolve()
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.training.evidence import tree_sha256  # noqa: E402
from app.training.hub_release import (  # noqa: E402
    build_phase10_model_card,
    build_release_evidence,
)
from app.inference.finetuned_protocol import load_phase10_protocol  # noqa: E402


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish the exact frozen Phase 10 adapter and truthful model card."
    )
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--evidence-output", required=True, type=Path)
    parser.add_argument(
        "--commit-message",
        default="Release EvalForge Phase 10 frozen QLoRA adapter",
    )
    args = parser.parse_args()

    if _git_output("rev-parse", "HEAD") != args.git_commit:
        raise RuntimeError("--git-commit must exactly match HEAD")
    if _git_output("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Hugging Face release requires a clean tracked worktree")
    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN", ""
    ).strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required for the Phase 10 Hugging Face release")
    if args.evidence_output.exists():
        raise RuntimeError(f"refusing to overwrite release evidence: {args.evidence_output}")

    protocol = load_phase10_protocol(ROOT)
    adapter_dir = args.adapter_dir.resolve()
    if tree_sha256(adapter_dir) != protocol.candidate_adapter_sha256:
        raise RuntimeError("local adapter is not the exact frozen Phase 10 candidate")

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for Phase 10 publication") from exc

    model_card = build_phase10_model_card(root=ROOT, repo_id=args.repo_id)
    model_card_sha256 = hashlib.sha256(model_card.encode()).hexdigest()

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
        token=token,
    )
    with tempfile.TemporaryDirectory(prefix="evalforge-hf-release-") as temp:
        readme_path = Path(temp) / "README.md"
        readme_path.write_text(model_card, encoding="utf-8")
        operations = [
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme_path))
        ]
        for path in sorted(item for item in adapter_dir.rglob("*") if item.is_file()):
            operations.append(
                CommitOperationAdd(
                    path_in_repo="adapter/" + path.relative_to(adapter_dir).as_posix(),
                    path_or_fileobj=str(path),
                )
            )
        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type="model",
            operations=operations,
            commit_message=args.commit_message,
            token=token,
        )

    revision = getattr(commit, "oid", None) or getattr(commit, "commit_oid", None)
    commit_url = getattr(commit, "commit_url", None)
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("Hugging Face create_commit did not return an immutable revision")
    if not isinstance(commit_url, str) or not commit_url:
        commit_url = f"https://huggingface.co/{args.repo_id}/commit/{revision}"

    info = api.model_info(args.repo_id, revision=revision, token=token)
    if getattr(info, "private", None) is not False:
        raise RuntimeError("Phase 10 Hugging Face release must be public")
    if getattr(info, "sha", None) != revision:
        raise RuntimeError("Hugging Face model_info revision disagrees with created commit")

    evidence = build_release_evidence(
        root=ROOT,
        repo_id=args.repo_id,
        revision=revision,
        commit_url=commit_url,
        adapter_dir=adapter_dir,
        source_git_commit=args.git_commit,
        model_card_sha256=model_card_sha256,
    )
    import json

    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PHASE10_HUGGINGFACE_RELEASE=PASS")
    print(f"HF_REPO_ID={args.repo_id}")
    print(f"HF_REVISION={revision}")
    print(f"ADAPTER_SHA256={protocol.candidate_adapter_sha256}")
    print(f"EVIDENCE_OUTPUT={args.evidence_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
