#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from app.data.postmortems import load_candidate_index
from app.data.primary_sources import (
    PrimarySourceKind,
    PrimarySourceManifest,
    PrimarySourcePlanEntry,
    PrimarySourceSnapshot,
    load_primary_source_manifest,
    load_primary_source_plan,
    sha256_bytes,
    validate_primary_source_preservation,
    validate_snapshot_payload,
)


def github_issue_url(entry: PrimarySourcePlanEntry) -> str:
    assert entry.repository is not None
    return f"https://api.github.com/repos/{entry.repository}/issues/{entry.issue_number}"


def git_blob_raw_url(entry: PrimarySourcePlanEntry) -> str:
    assert entry.repository is not None
    assert entry.commit is not None
    assert entry.source_path is not None
    return (
        f"https://raw.githubusercontent.com/{entry.repository}/{entry.commit}/{entry.source_path}"
    )


def fetch_entry(entry: PrimarySourcePlanEntry) -> tuple[bytes, str]:
    headers = {
        "User-Agent": ("Mozilla/5.0 (compatible; EvalForge-Phase03-PrimarySourcePreserver/2.0)")
    }
    if entry.kind == PrimarySourceKind.GITHUB_ISSUE:
        url = github_issue_url(entry)
        headers["Accept"] = "application/vnd.github+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    elif entry.kind == PrimarySourceKind.GIT_BLOB:
        url = git_blob_raw_url(entry)
        headers["Accept"] = "text/plain"
    else:
        assert entry.source_url is not None
        url = entry.source_url
        headers["Accept"] = "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8"
        headers["Accept-Language"] = "en-US,en;q=0.8"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
        media_type = response.headers.get_content_type()
    return payload, media_type


def _reuse_existing_snapshot(
    root: Path,
    entry: PrimarySourcePlanEntry,
    candidate_original_url: str,
    snapshot: PrimarySourceSnapshot,
) -> PrimarySourceSnapshot:
    if snapshot.snapshot_path != entry.snapshot_path:
        raise ValueError(f"existing preserved-source path mismatch: {entry.candidate_id}")
    if snapshot.snapshot_kind != entry.kind:
        raise ValueError(f"existing preserved-source kind mismatch: {entry.candidate_id}")
    if snapshot.original_url != candidate_original_url:
        raise ValueError(f"existing preserved-source URL mismatch: {entry.candidate_id}")
    payload = (root / snapshot.snapshot_path).read_bytes()
    if len(payload) != snapshot.byte_length:
        raise ValueError(f"existing preserved-source byte length mismatch: {entry.candidate_id}")
    if sha256_bytes(payload) != snapshot.snapshot_sha256:
        raise ValueError(f"existing preserved-source checksum mismatch: {entry.candidate_id}")
    validate_snapshot_payload(
        entry,
        original_url=candidate_original_url,
        payload=payload,
    )
    return snapshot


def bootstrap(root: Path) -> None:
    plan_path = root / "configs/phase3-primary-sources.json"
    index_path = root / "configs/postmortem-candidates.json"
    manifest_path = (
        root
        / "datasets/incident_diagnosis/raw/public_incidents/"
        / "phase3-primary-source-v1/manifest.json"
    )
    plan = load_primary_source_plan(plan_path)
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    index = load_candidate_index(index_path)
    by_candidate = {item.candidate_id: item for item in index.candidates}
    raw_by_candidate = {item["candidate_id"]: item for item in index_payload["candidates"]}

    existing_snapshots: dict[str, PrimarySourceSnapshot] = {}
    if manifest_path.exists():
        existing_manifest = load_primary_source_manifest(manifest_path)
        existing_snapshots = {item.candidate_id: item for item in existing_manifest.snapshots}

    captured_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    snapshots: list[PrimarySourceSnapshot] = []
    with tempfile.TemporaryDirectory(prefix="evalforge-phase3-primary-") as temp_dir:
        temp_root = Path(temp_dir)
        fetched: list[tuple[PrimarySourcePlanEntry, bytes, str, str]] = []
        for entry in plan.entries:
            candidate = by_candidate.get(entry.candidate_id)
            if candidate is None:
                raise ValueError(f"unknown preservation candidate: {entry.candidate_id}")
            existing = existing_snapshots.get(entry.candidate_id)
            if existing is not None:
                snapshots.append(
                    _reuse_existing_snapshot(
                        root,
                        entry,
                        candidate.original_url,
                        existing,
                    )
                )
                continue

            payload, media_type = fetch_entry(entry)
            validate_snapshot_payload(
                entry,
                original_url=candidate.original_url,
                payload=payload,
            )
            temp_path = temp_root / Path(entry.snapshot_path).name
            temp_path.write_bytes(payload)
            fetched.append((entry, payload, media_type, str(temp_path)))

        for entry, payload, media_type, temp_path_text in fetched:
            candidate = by_candidate[entry.candidate_id]
            destination = root / entry.snapshot_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(temp_path_text, destination)
            sha = sha256_bytes(payload)
            if entry.kind == PrimarySourceKind.GITHUB_ISSUE:
                assert entry.repository is not None
                source_identity = f"github_issue:{entry.repository}#{entry.issue_number}"
            elif entry.kind == PrimarySourceKind.GIT_BLOB:
                assert entry.repository is not None
                source_identity = (
                    f"git_blob:{entry.repository}@{entry.commit}:"
                    f"{entry.source_path}:{entry.expected_git_blob_sha}"
                )
            else:
                assert entry.source_url is not None
                source_identity = f"http_document:{entry.source_url}"
            snapshots.append(
                PrimarySourceSnapshot(
                    candidate_id=entry.candidate_id,
                    original_url=candidate.original_url,
                    snapshot_path=entry.snapshot_path,
                    snapshot_sha256=sha,
                    snapshot_kind=entry.kind,
                    source_identity=source_identity,
                    captured_at=captured_at,
                    media_type=media_type,
                    byte_length=len(payload),
                )
            )
            raw_candidate = raw_by_candidate[entry.candidate_id]
            raw_candidate["original_source_snapshot_preserved"] = True
            raw_candidate["preserved_primary_source_path"] = entry.snapshot_path
            raw_candidate["preserved_primary_source_sha256"] = sha
            raw_candidate["preserved_primary_source_kind"] = entry.kind.value

    if index_payload.get("index_version") == "phase3-public-postmortem-candidates-v5":
        index_payload["index_version"] = "phase3-public-postmortem-candidates-v6"
    index_path.write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = PrimarySourceManifest(
        manifest_version="phase3-primary-source-manifest-v2",
        plan_version=plan.plan_version,
        snapshots=sorted(snapshots, key=lambda item: item.candidate_id),
    )
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify(root: Path) -> None:
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    plan = load_primary_source_plan(root / "configs/phase3-primary-sources.json")
    manifest = load_primary_source_manifest(
        root
        / "datasets/incident_diagnosis/raw/public_incidents/"
        / "phase3-primary-source-v1/manifest.json"
    )
    report = validate_primary_source_preservation(root, index, plan, manifest)
    print("PHASE03_PRIMARY_SOURCE_PRESERVATION=PASS")
    print(f"PRESERVED_CANDIDATES={report.preserved_candidate_count}")
    for code, count in report.preserved_supported_family_counts_by_root_cause_code.items():
        print(f"PRESERVED_{code.upper()}={count}")
    print(
        "PRESERVED_FAMILY_DEPTH_SUFFICIENT="
        f"{str(report.preserved_supported_family_depth_sufficient_for_split).lower()}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.bootstrap:
        bootstrap(root)
    verify(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
