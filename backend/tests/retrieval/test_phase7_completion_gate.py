from __future__ import annotations

from pathlib import Path

import pytest

from app.retrieval.phase7_gate import (
    Phase7Stage,
    _validate_review_payload,
    evaluate_phase7_repository_state,
)

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SAMPLE_IDS = (
    "incident-0557d490e1a492950c4ac3e4",
    "incident-215133387692e86f737df1e9",
)


def _approved_review() -> dict[str, object]:
    return {
        "review_version": "phase7-retrieval-review-v1",
        "approved": True,
        "reviewer": "project-owner",
        "reviewed_at": "2026-09-13T00:00:00+00:00",
        "reviewed_sample_incident_ids": list(EXPECTED_SAMPLE_IDS),
        "relevance_reviewed": True,
        "provenance_reviewed": True,
        "leakage_guard_reviewed": True,
        "limitations_acknowledged": True,
        "notes": "Reviewed train and validation retrieval traces and acknowledged baseline limits.",
    }


def test_committed_phase7_state_waits_for_manual_review() -> None:
    status = evaluate_phase7_repository_state(ROOT)

    assert status.stage is Phase7Stage.AWAITING_MANUAL_REVIEW
    assert status.blocker is not None
    assert "manual" in status.blocker
    assert status.kb_version == "evalforge-kb-v0.1.0"
    assert status.manifest_checksum == (
        "a55b666c6508c4a1190b70c5fcae1694c54275afc5380f6bed967c95894c51b5"
    )
    assert status.document_count == 24
    assert status.chunk_count == 25
    assert status.leakage_query_count == 12
    assert status.sample_incident_ids == EXPECTED_SAMPLE_IDS
    assert status.ci_run_id == 34718610607
    assert status.ci_head_sha == "bca42ed037833558a12f8881b75b6ee93c927c6a"
    assert status.reviewer is None


def test_review_contract_accepts_complete_approval() -> None:
    reviewer = _validate_review_payload(_approved_review(), EXPECTED_SAMPLE_IDS)
    assert reviewer == "project-owner"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("approved", False, "not approved"),
        ("relevance_reviewed", False, "relevance_reviewed"),
        ("provenance_reviewed", False, "provenance_reviewed"),
        ("leakage_guard_reviewed", False, "leakage_guard_reviewed"),
        ("limitations_acknowledged", False, "limitations_acknowledged"),
    ],
)
def test_review_contract_rejects_incomplete_approval(
    field: str,
    value: object,
    message: str,
) -> None:
    review = _approved_review()
    review[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_review_payload(review, EXPECTED_SAMPLE_IDS)


def test_review_contract_rejects_different_sample_ids() -> None:
    review = _approved_review()
    review["reviewed_sample_incident_ids"] = [EXPECTED_SAMPLE_IDS[0]]
    with pytest.raises(ValueError, match="sample IDs do not match"):
        _validate_review_payload(review, EXPECTED_SAMPLE_IDS)
