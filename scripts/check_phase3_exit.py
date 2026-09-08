#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    version = "evalforge-incident-diagnosis-v0.1.0-candidate"
    manifest_path = root / "datasets/incident_diagnosis/processed" / version / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("research_ready"):
        print("PHASE03_EXIT_GATE=BLOCKED")
        print("BLOCKER=independent_taxonomy_coverage_insufficient")
        print(f"SOURCE_CANDIDATES={manifest.get('source_candidate_count', 0)}")
        auxiliary = manifest.get("auxiliary_source_counts", {})
        print(f"AUXILIARY_REAL_INCIDENTS={auxiliary.get('servicenow_uci_incidents', 0)}")
        print(
            f"INDEPENDENT_POSTMORTEM_CANDIDATES={auxiliary.get('public_postmortem_candidates', 0)}"
        )
        print(
            "SUPPORTED_MAPPING_CANDIDATES="
            f"{auxiliary.get('public_postmortem_supported_mappings', 0)}"
        )
        print(
            "SUPPORTED_TAXONOMY_LABELS="
            f"{auxiliary.get('public_postmortem_supported_taxonomy_labels', 0)}"
        )
        coverage_path = manifest_path.parent / "public-postmortem-candidate-coverage.json"
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))["coverage"]
        print("MISSING_TAXONOMY_LABELS=" + ",".join(coverage["missing_root_cause_codes"]))
        print("BLOCKER_DETAIL=credible_family_stratified_holdout_not_yet_possible")
        print("RESEARCH_TRAIN=0")
        print("RESEARCH_VALIDATION=0")
        print("RESEARCH_TEST=0")
        return 2
    if manifest["split_counts"].get("validation", 0) <= 0:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL validation split empty")
    if manifest["split_counts"].get("test", 0) <= 0:
        raise SystemExit("PHASE03_EXIT_GATE=FAIL test split empty")
    print("PHASE03_EXIT_GATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
