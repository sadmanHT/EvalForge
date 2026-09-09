#!/usr/bin/env python
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FINAL_EXIT_CHECKER = '''#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    version = "evalforge-incident-diagnosis-v0.1.0-candidate"
    manifest_path = root / "datasets/incident_diagnosis/processed" / version / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("research_ready"):
        coverage_path = manifest_path.parent / "public-postmortem-candidate-coverage.json"
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))["coverage"]
        blocker = coverage.get("blocker", "independent_taxonomy_coverage_insufficient")
        print("PHASE03_EXIT_GATE=BLOCKED")
        print(f"BLOCKER={blocker}")
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
        print("MISSING_TAXONOMY_LABELS=" + ",".join(coverage["missing_root_cause_codes"]))
        if blocker == "original_source_preservation_and_research_admission_required":
            print(
                "BLOCKER_DETAIL=original_primary_source_preservation_and_canonical_"
                "research_admission_required"
            )
        else:
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
'''


def normalize_preconditions() -> None:
    build = ROOT / "scripts/build_phase3_candidate.py"
    text = build.read_text(encoding="utf-8")
    pattern = re.compile(
        r'''            \(\n'''
        r'''\s+"Independent public postmortem candidates exist, but the frozen taxonomy "\n'''
        r'''\s+"is not covered sufficiently for a credible family-stratified holdout\."\n'''
        r'''\s+\),'''
    )
    replacement = '''            (
                "Independent public postmortem candidates exist, but the frozen taxonomy is not covered sufficiently for a credible family-stratified holdout."
            ),'''
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"expected one family-depth limitation source block, found {count}")
    build.write_text(text, encoding="utf-8")

    ci = ROOT / ".github/workflows/ci.yml"
    ci_text = ci.read_text(encoding="utf-8")
    old = "grep -q '^BLOCKER=independent_taxonomy_coverage_insufficient$' /tmp/phase03-exit.txt"
    new = 'grep -q "^BLOCKER=independent_taxonomy_coverage_insufficient$" /tmp/phase03-exit.txt'
    if ci_text.count(old) != 1:
        raise SystemExit(f"expected one Phase 03 CI blocker grep, found {ci_text.count(old)}")
    ci.write_text(ci_text.replace(old, new), encoding="utf-8")


def normalize_generated_output() -> None:
    build = ROOT / "scripts/build_phase3_candidate.py"
    text = build.read_text(encoding="utf-8")
    old = '''            (
                "Independent public incident candidates now provide the splitter's minimum three supported families per frozen taxonomy label, but no candidate is research-admitted until original primary-source evidence is preserved and canonical records are built."
            ),'''
    new = '''            (
                "Independent public incident candidates now provide the splitter's minimum "
                "three supported families per frozen taxonomy label, but no candidate is "
                "research-admitted until original primary-source evidence is preserved "
                "and canonical records are built."
            ),'''
    if text.count(old) != 1:
        raise SystemExit(f"expected one generated family-depth limitation block, found {text.count(old)}")
    build.write_text(text.replace(old, new), encoding="utf-8")

    (ROOT / "scripts/check_phase3_exit.py").write_text(FINAL_EXIT_CHECKER, encoding="utf-8")


def main() -> int:
    normalize_preconditions()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "backend")
    subprocess.run(
        [sys.executable, "scripts/phase3_family_depth_increment.py"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    normalize_generated_output()
    print("PHASE03_FAMILY_DEPTH_FINALIZE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
