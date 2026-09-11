#!/usr/bin/env python
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from build_phase3_research_dataset import build_research_outputs

FILES = (
    "incidents.jsonl",
    "families.jsonl",
    "manifest.json",
    "leakage-audit-report.json",
    "class-split-summary.json",
    "research-admission-summary.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    version = "evalforge-incident-diagnosis-v0.1.0"
    committed = root / "datasets/incident_diagnosis/processed" / version
    with tempfile.TemporaryDirectory(prefix="evalforge-phase3-research-rebuild-") as tmp:
        temp_root = Path(tmp)
        shutil.copytree(root / "configs", temp_root / "configs")
        public_raw = root / "datasets/incident_diagnosis/raw/public_incidents"
        shutil.copytree(
            public_raw,
            temp_root / "datasets/incident_diagnosis/raw/public_incidents",
        )
        build_research_outputs(temp_root)
        rebuilt = temp_root / "datasets/incident_diagnosis/processed" / version
        for name in FILES:
            if sha(committed / name) != sha(rebuilt / name):
                raise SystemExit(f"PHASE03_RESEARCH_REPRODUCIBILITY=FAIL file={name}")
    print("PHASE03_RESEARCH_REPRODUCIBILITY=PASS")
    for name in FILES:
        print(f"{name.upper().replace('-', '_').replace('.', '_')}_SHA256={sha(committed / name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
