#!/usr/bin/env python
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from build_phase3_candidate import build_outputs

FILES = (
    "manifest.json",
    "source-candidate-summary.json",
    "source-audit-report.json",
    "auxiliary-servicenow-summary.json",
    "public-postmortem-candidate-coverage.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    version = "evalforge-incident-diagnosis-v0.1.0-candidate"
    committed = root / "datasets/incident_diagnosis/processed" / version
    with tempfile.TemporaryDirectory(prefix="evalforge-phase3-rebuild-") as tmp:
        temp_root = Path(tmp)
        shutil.copytree(root / "configs", temp_root / "configs")
        raw_root = root / "datasets/incident_diagnosis/raw"
        shutil.copytree(raw_root, temp_root / "datasets/incident_diagnosis/raw")
        build_outputs(temp_root)
        rebuilt = temp_root / "datasets/incident_diagnosis/processed" / version
        for name in FILES:
            if sha(committed / name) != sha(rebuilt / name):
                raise SystemExit(f"PHASE03_REPRODUCIBILITY=FAIL file={name}")
    print("PHASE03_REPRODUCIBILITY=PASS")
    for name in FILES:
        print(f"{name.upper().replace('-', '_').replace('.', '_')}_SHA256={sha(committed / name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
