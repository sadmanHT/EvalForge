#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, config: dict[str, object]) -> tuple[str, str]:
    archive_sha = sha256(path)
    accepted = set(config["accepted_archive_sha256s"])  # type: ignore[arg-type]
    if archive_sha not in accepted:
        raise ValueError(f"unreviewed ServiceNow transport archive SHA-256: {archive_sha}")

    member_name = str(config["canonical_member_filename"])
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != [member_name]:
            raise ValueError(f"unexpected ServiceNow archive members: {archive.namelist()!r}")
        member_sha = hashlib.sha256(archive.read(member_name)).hexdigest()
    expected_member_sha = str(config["canonical_member_sha256"])
    if member_sha != expected_member_sha:
        raise ValueError(
            "ServiceNow canonical member checksum mismatch: "
            f"expected={expected_member_sha} actual={member_sha}"
        )
    return archive_sha, member_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "configs/dataset.yaml").read_text(encoding="utf-8"))

    source = config["source_snapshot"]
    source_target = root / source["path"]
    source_pack = root / source["transport_pack_path"]
    pack_sha = sha256(source_pack)
    if pack_sha != source["transport_pack_sha256"]:
        raise ValueError(
            "OpsSentinel snapshot transport-pack checksum mismatch: "
            f"expected={source['transport_pack_sha256']} actual={pack_sha}"
        )
    if not source_target.exists():
        encoded = b"".join(source_pack.read_bytes().splitlines())
        source_bytes = gzip.decompress(base64.b64decode(encoded))
        source_target.parent.mkdir(parents=True, exist_ok=True)
        source_target.write_bytes(source_bytes)
    source_sha = sha256(source_target)
    source_metadata = json.loads(
        (source_target.parent / "SOURCE.json").read_text(encoding="utf-8")
    )
    if source_sha != source_metadata["snapshot_sha256"]:
        raise ValueError(
            "OpsSentinel reconstructed snapshot checksum mismatch: "
            f"expected={source_metadata['snapshot_sha256']} actual={source_sha}"
        )

    auxiliary = config["auxiliary_source_snapshots"][0]
    target = root / auxiliary["path"]
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        with tempfile.NamedTemporaryFile(
            prefix="evalforge-servicenow-", suffix=".zip", delete=False
        ) as temp_handle:
            temp = Path(temp_handle.name)
        try:
            with (
                urllib.request.urlopen(auxiliary["download_url"], timeout=60) as response,
                temp.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            verify_archive(temp, auxiliary)
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)

    archive_sha, member_sha = verify_archive(target, auxiliary)
    print("PHASE03_SOURCE_FETCH=PASS")
    print(f"OPSSENTINEL_SNAPSHOT_SHA256={source_sha}")
    print(f"SERVICENOW_ARCHIVE_SHA256={archive_sha}")
    print(f"SERVICENOW_CANONICAL_MEMBER_SHA256={member_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
