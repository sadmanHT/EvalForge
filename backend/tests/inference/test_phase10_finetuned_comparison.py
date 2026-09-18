from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.inference.finetuned_comparison import (
    build_paired_baseline_finetuned_comparison,
    validate_paired_baseline_finetuned_comparison,
)
from app.inference.finetuned_evidence import validate_portable_finetuned_evidence
from app.inference.finetuned_protocol import load_phase10_protocol
from app.inference.protocol import load_baseline_protocol
from app.training.evidence import sha256_file

ROOT = Path(__file__).resolve().parents[3]
BASELINE_PATH = ROOT / "evidence/phase-06/test-run.json"
FINETUNED_PATH = ROOT / "evidence/phase-10/locked-test/phase10-finetuned-test.json"
SOURCE_COMMIT = "c59910e00f5e4fd0a722d2796da416c977753ddd"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_locked_test_evidence_matches_frozen_phase10_identity() -> None:
    protocol = load_phase10_protocol(ROOT)
    evidence = _load(FINETUNED_PATH)

    validate_portable_finetuned_evidence(
        evidence,
        root=ROOT,
        protocol=protocol,
        expected_split="test",
        expected_run_id="phase10-finetuned-test-v1",
        expected_source_commit=SOURCE_COMMIT,
    )


def test_paired_baseline_finetuned_comparison_is_reproducible() -> None:
    baseline_protocol = load_baseline_protocol(ROOT)
    finetuned_protocol = load_phase10_protocol(ROOT)
    baseline = _load(BASELINE_PATH)
    finetuned = _load(FINETUNED_PATH)
    file_sha = sha256_file(FINETUNED_PATH)

    comparison = build_paired_baseline_finetuned_comparison(
        root=ROOT,
        baseline_protocol=baseline_protocol,
        finetuned_protocol=finetuned_protocol,
        baseline_evidence=baseline,
        finetuned_evidence=finetuned,
        finetuned_evidence_file_sha256=file_sha,
        expected_finetuned_source_commit=SOURCE_COMMIT,
    )

    assert comparison["baseline"]["exact_accuracy"] == 1.0
    assert comparison["finetuned"]["exact_accuracy"] == pytest.approx(5 / 6)
    paired = comparison["paired_accuracy"]
    assert paired["finetuned_minus_baseline"] == pytest.approx(-1 / 6)
    assert paired["bootstrap_ci_lower"] == pytest.approx(-0.5)
    assert paired["bootstrap_ci_upper"] == 0.0
    assert paired["mcnemar_baseline_only_correct"] == 1
    assert paired["mcnemar_finetuned_only_correct"] == 0
    assert paired["mcnemar_discordant_pairs"] == 1
    assert paired["mcnemar_p_value"] == 1.0

    discordant = [
        row
        for row in comparison["paired_rows"]
        if row["baseline_correct"] != row["finetuned_correct"]
    ]
    assert [row["incident_id"] for row in discordant] == ["incident-420f3a36538bb8156dd897d8"]
    assert discordant[0]["finetuned_parse_status"] == "INVALID_JSON"

    validate_paired_baseline_finetuned_comparison(
        comparison,
        root=ROOT,
        baseline_protocol=baseline_protocol,
        finetuned_protocol=finetuned_protocol,
        baseline_evidence=baseline,
        finetuned_evidence=finetuned,
        finetuned_evidence_file_sha256=file_sha,
        expected_finetuned_source_commit=SOURCE_COMMIT,
    )


def test_comparison_rejects_missing_finetuned_prediction() -> None:
    baseline_protocol = load_baseline_protocol(ROOT)
    finetuned_protocol = load_phase10_protocol(ROOT)
    baseline = _load(BASELINE_PATH)
    finetuned = copy.deepcopy(_load(FINETUNED_PATH))
    finetuned["predictions"] = finetuned["predictions"][:-1]
    finetuned["prediction_count"] = 5

    with pytest.raises(ValueError, match="missing or unexpected predictions"):
        build_paired_baseline_finetuned_comparison(
            root=ROOT,
            baseline_protocol=baseline_protocol,
            finetuned_protocol=finetuned_protocol,
            baseline_evidence=baseline,
            finetuned_evidence=finetuned,
            finetuned_evidence_file_sha256=sha256_file(FINETUNED_PATH),
            expected_finetuned_source_commit=SOURCE_COMMIT,
        )
