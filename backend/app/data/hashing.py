from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from app.data.schemas import IncidentFamily, IncidentRecord


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dataset_content_checksum(
    records: Iterable[IncidentRecord],
    families: Iterable[IncidentFamily],
) -> str:
    payload = {
        "records": [
            item.model_dump(mode="json")
            for item in sorted(records, key=lambda item: item.incident_id)
        ],
        "families": [
            item.model_dump(mode="json")
            for item in sorted(families, key=lambda item: item.family_id)
        ],
    }
    return sha256_hex(canonical_json_bytes(payload))


def manifest_checksum(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("manifest_checksum", None)
    return sha256_hex(canonical_json_bytes(clean))
