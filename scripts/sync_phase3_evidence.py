#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VERSION = "evalforge-incident-diagnosis-v0.1.0-candidate"
MIRRORS = {
    "manifest.json": "candidate-manifest.json",
    "source-candidate-summary.json": "source-candidate-summary.json",
    "source-audit-report.json": "source-audit-report.json",
    "auxiliary-servicenow-summary.json": "auxiliary-servicenow-summary.json",
    "public-postmortem-candidate-coverage.json": "public-postmortem-candidate-coverage.json",
}


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()

    processed = root / "datasets/incident_diagnosis/processed" / VERSION
    evidence = root / "evidence/phase-03"
    for source_name, evidence_name in MIRRORS.items():
        (evidence / evidence_name).write_bytes((processed / source_name).read_bytes())

    coverage = json.loads(
        (processed / "public-postmortem-candidate-coverage.json").read_text(
            encoding="utf-8"
        )
    )["coverage"]
    class_summary_path = evidence / "class-split-summary.json"
    class_summary = json.loads(class_summary_path.read_text(encoding="utf-8"))
    class_summary["public_postmortem_preserved_original_source_count"] = coverage[
        "preserved_original_source_count"
    ]
    class_summary["public_postmortem_preserved_family_counts_by_root_cause_code"] = (
        coverage["preserved_family_counts_by_root_cause_code"]
    )
    class_summary["public_postmortem_preserved_family_depth_sufficient_for_split"] = coverage[
        "preserved_family_depth_sufficient_for_split"
    ]
    write_json(class_summary_path, class_summary)

    checksum_names = [
        "candidate-manifest.json",
        "source-candidate-summary.json",
        "source-audit-report.json",
        "auxiliary-servicenow-summary.json",
        "public-postmortem-candidate-coverage.json",
        "class-split-summary.json",
    ]
    lines = [
        f"{hashlib.sha256((evidence / name).read_bytes()).hexdigest()}  {name}"
        for name in checksum_names
    ]
    (evidence / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("PHASE03_EVIDENCE_SYNC=PASS")
    print("PRESERVED_PRIMARY_SOURCES=" f"{coverage['preserved_original_source_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
