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
    return f"https://api.github.com/repos/{entry.repository}/issues/{entry.issue_number}"


def git_blob_raw_url(entry: PrimarySourcePlanEntry) -> str:
    return (
        f"https://raw.githubusercontent.com/{entry.repository}/{entry.commit}/{entry.source_path}"
    )


def fetch_entry(entry: PrimarySourcePlanEntry) -> tuple[bytes, str]:
    if entry.kind == PrimarySourceKind.GITHUB_ISSUE:
        url = github_issue_url(entry)
        accept = "application/vnd.github+json"
    else:
        url = git_blob_raw_url(entry)
        accept = "text/plain"

    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": "EvalForge-Phase03-PrimarySourcePreserver/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
        media_type = response.headers.get_content_type()
    return payload, media_type


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

    captured_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    snapshots: list[PrimarySourceSnapshot] = []
    with tempfile.TemporaryDirectory(prefix="evalforge-phase3-primary-") as temp_dir:
        temp_root = Path(temp_dir)
        fetched: list[tuple[PrimarySourcePlanEntry, bytes, str, str]] = []
        for entry in plan.entries:
            candidate = by_candidate.get(entry.candidate_id)
            if candidate is None:
                raise ValueError(f"unknown preservation candidate: {entry.candidate_id}")
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
                source_identity = f"github_issue:{entry.repository}#{entry.issue_number}"
            else:
                source_identity = (
                    f"git_blob:{entry.repository}@{entry.commit}:"
                    f"{entry.source_path}:{entry.expected_git_blob_sha}"
                )
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

    if index_payload.get("index_version") == "phase3-public-postmortem-candidates-v4":
        index_payload["index_version"] = "phase3-public-postmortem-candidates-v5"
    index_path.write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = PrimarySourceManifest(
        manifest_version="phase3-primary-source-manifest-v1",
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
