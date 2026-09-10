#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one migration anchor in {path}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_all(path: Path, old: str, new: str, *, expected_count: int) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != expected_count:
        raise SystemExit(
            f"expected {expected_count} migration anchors in {path}, "
            f"found {text.count(old)}: {old[:100]!r}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_test = root / "backend/tests/data/test_phase3_data.py"
    primary_test = root / "backend/tests/data/test_phase3_primary_sources.py"

    replace_once(
        data_test,
        "    assert coverage.admitted_research_record_count == 0\n"
        "    assert not coverage.taxonomy_coverage_sufficient_for_locked_holdout\n"
        "    assert coverage.blocker == \"original_source_preservation_and_research_admission_required\"\n",
        "    assert coverage.admitted_research_record_count == 18\n"
        "    assert coverage.taxonomy_coverage_sufficient_for_locked_holdout\n"
        "    assert coverage.blocker == \"none\"\n",
    )
    replace_once(
        data_test,
        "    assert replacement.original_source_snapshot_preserved\n"
        "    assert not replacement.research_admitted\n",
        "    assert replacement.original_source_snapshot_preserved\n"
        "    assert replacement.research_admitted\n",
    )
    replace_once(
        data_test,
        "def test_supported_postmortem_primary_source_wave_remains_nonadmitted() -> None:\n",
        "def test_supported_preserved_postmortems_are_explicitly_research_admitted() -> None:\n",
    )
    replace_all(
        data_test,
        "        assert not candidate.research_admitted\n",
        "        assert candidate.research_admitted\n",
        expected_count=2,
    )
    replace_once(
        data_test,
        "    assert all(not item.research_admitted for item in supported)\n",
        "    assert all(item.research_admitted for item in supported)\n",
    )

    replace_once(
        primary_test,
        "def test_committed_primary_source_wave_is_checksum_valid_and_nonadmitted() -> None:\n",
        "def test_committed_primary_sources_are_checksum_valid_and_research_admitted() -> None:\n",
    )
    replace_once(
        primary_test,
        "    assert all(not item.research_admitted for item in preserved)\n",
        "    assert all(item.research_admitted for item in preserved)\n",
    )

    print("PHASE03_RESEARCH_LOCK_EXPECTATION_MIGRATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
