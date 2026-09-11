#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

CANDIDATE_VERSION = "evalforge-incident-diagnosis-v0.1.0-candidate"
RESEARCH_VERSION = "evalforge-incident-diagnosis-v0.1.0"
CANDIDATE_MIRRORS = {
    "manifest.json": "candidate-manifest.json",
    "source-candidate-summary.json": "source-candidate-summary.json",
    "source-audit-report.json": "source-audit-report.json",
    "auxiliary-servicenow-summary.json": "auxiliary-servicenow-summary.json",
    "public-postmortem-candidate-coverage.json": "public-postmortem-candidate-coverage.json",
}
RESEARCH_MIRRORS = {
    "manifest.json": "research-manifest.json",
    "leakage-audit-report.json": "leakage-audit-report.json",
    "class-split-summary.json": "class-split-summary.json",
    "research-admission-summary.json": "research-admission-summary.json",
}
DOC_MIRRORS = {
    "docs/data-card.md": "data-card.md",
    "docs/phase-03-handoff.md": "phase-03-handoff.md",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()

    candidate = root / "datasets/incident_diagnosis/processed" / CANDIDATE_VERSION
    research = root / "datasets/incident_diagnosis/processed" / RESEARCH_VERSION
    evidence = root / "evidence/phase-03"
    evidence.mkdir(parents=True, exist_ok=True)

    for source_name, evidence_name in CANDIDATE_MIRRORS.items():
        (evidence / evidence_name).write_bytes((candidate / source_name).read_bytes())
    for source_name, evidence_name in RESEARCH_MIRRORS.items():
        (evidence / evidence_name).write_bytes((research / source_name).read_bytes())
    for source_name, evidence_name in DOC_MIRRORS.items():
        (evidence / evidence_name).write_bytes((root / source_name).read_bytes())

    checksum_names = sorted(
        [
            *CANDIDATE_MIRRORS.values(),
            *RESEARCH_MIRRORS.values(),
            *DOC_MIRRORS.values(),
        ]
    )
    lines = [
        f"{hashlib.sha256((evidence / name).read_bytes()).hexdigest()}  {name}"
        for name in checksum_names
    ]
    (evidence / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("PHASE03_EVIDENCE_SYNC=PASS")
    print("RESEARCH_DATASET_VERSION=evalforge-incident-diagnosis-v0.1.0")
    print("RESEARCH_EVIDENCE_FILES=" + str(len(checksum_names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
