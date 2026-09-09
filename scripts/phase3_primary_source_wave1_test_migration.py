#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path


def main() -> int:
    path = Path("backend/tests/data/test_phase3_data.py")
    text = path.read_text(encoding="utf-8")
    marker = (
        "\ndef test_supported_postmortem_candidates_still_require_primary_source_snapshot() "
        "-> None:\n"
    )
    prefix, separator, _ = text.partition(marker)
    if not separator:
        raise SystemExit("expected obsolete primary-source regression test was not found")

    replacement = '''
def test_supported_postmortem_primary_source_wave_remains_nonadmitted() -> None:
    root = Path(__file__).resolve().parents[3]
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    supported = [item for item in index.candidates if item.decision == CandidateDecision.SUPPORTED]
    assert len(supported) == 18
    assert len({item.candidate_id for item in supported}) == 18
    by_id = {item.candidate_id: item for item in supported}

    medoc = by_id["github:Nikhil-Gautam-dev:medoc-n-plus-one:2026-01-11"]
    assert medoc.evidence_repository == "Nikhil-Gautam-dev/nikhil-gautam-dev.github.io"
    assert medoc.evidence_commit == "34d1ecc14b54166608b4197c44e1e7efc82e48b6"
    assert medoc.source_blob_sha == "750ba91067e5e4d2e192a3eceffec357a58e9d77"

    preserved_ids = {
        "github:Nikhil-Gautam-dev:medoc-n-plus-one:2026-01-11",
        "github:Dispatcharr:issue-1416:2026-07-06",
        "openlibrary:issue-12432:2026-04-22:n-plus-one",
        "dify:issue-40036:2026-08-05:n-plus-one",
    }
    assert {
        item.candidate_id for item in supported if item.original_source_snapshot_preserved
    } == preserved_ids
    for candidate_id in preserved_ids:
        candidate = by_id[candidate_id]
        assert candidate.preserved_primary_source_path
        assert candidate.preserved_primary_source_sha256
        assert candidate.preserved_primary_source_kind in {"git_blob", "github_issue"}
        assert not candidate.research_admitted

    expected_normalized_snapshots = {
        "github:Dispatcharr:issue-1416:2026-07-06": "database_connection_leak",
        "google-cloud-status:E18Caoo5X1m6dTa1PVr1": "broken_payment_configuration",
        "google-cloud-status:fLYHLzSGXGkLkAjc8MJG": "no_fault",
    }
    for candidate_id, label in expected_normalized_snapshots.items():
        candidate = by_id[candidate_id]
        assert candidate.proposed_root_cause_code == label
        snapshot = (root / candidate.source_path).read_bytes()
        header = f"blob {len(snapshot)}\\0".encode()
        assert hashlib.sha1(header + snapshot).hexdigest() == candidate.source_blob_sha
        assert not candidate.research_admitted

    for candidate_id in (
        "google-cloud-status:E18Caoo5X1m6dTa1PVr1",
        "google-cloud-status:fLYHLzSGXGkLkAjc8MJG",
    ):
        assert not by_id[candidate_id].original_source_snapshot_preserved
    assert all(not item.research_admitted for item in supported)
'''
    path.write_text(prefix + replacement, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
