#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.data.postmortems import load_candidate_index
from app.data.research import (
    load_research_admission_plan,
    validate_research_admission_plan,
)
from app.data.taxonomy import load_taxonomy


def apply_admission(root: Path) -> None:
    index_path = root / "configs/postmortem-candidates.json"
    plan = load_research_admission_plan(root / "configs/phase3-research-admissions.json")
    taxonomy = load_taxonomy(root / "configs/label-taxonomy.yaml")
    index = load_candidate_index(index_path)
    if index.index_version != plan.candidate_index_version_before_admission:
        raise ValueError(
            "candidate index version is not the expected pre-admission version: "
            f"expected={plan.candidate_index_version_before_admission} "
            f"actual={index.index_version}"
        )
    validate_research_admission_plan(index, plan, taxonomy, require_applied=False)
    if any(candidate.research_admitted for candidate in index.candidates):
        raise ValueError("pre-admission candidate index must contain zero research admissions")

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    planned = set(plan.candidate_ids)
    for candidate in payload["candidates"]:
        candidate["research_admitted"] = candidate["candidate_id"] in planned
    payload["index_version"] = plan.candidate_index_version_after_admission
    index_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def verify_admission(root: Path) -> None:
    plan = load_research_admission_plan(root / "configs/phase3-research-admissions.json")
    taxonomy = load_taxonomy(root / "configs/label-taxonomy.yaml")
    index = load_candidate_index(root / "configs/postmortem-candidates.json")
    if index.index_version != plan.candidate_index_version_after_admission:
        raise ValueError(
            "candidate index version is not the expected admitted version: "
            f"expected={plan.candidate_index_version_after_admission} actual={index.index_version}"
        )
    candidates = validate_research_admission_plan(
        index,
        plan,
        taxonomy,
        require_applied=True,
    )
    print("PHASE03_RESEARCH_ADMISSION=PASS")
    print(f"RESEARCH_ADMITTED_CANDIDATES={len(candidates)}")
    print(f"RESEARCH_ADMISSION_VERSION={plan.admission_version}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.apply:
        apply_admission(root)
    verify_admission(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
