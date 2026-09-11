#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.data.audit import audit_dataset
from app.data.io import read_jsonl
from app.data.schemas import IncidentFamily, IncidentRecord
from app.data.taxonomy import load_taxonomy


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an EvalForge incident dataset")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("configs/label-taxonomy.yaml"),
    )
    parser.add_argument("--research-mode", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    records = read_jsonl(args.dataset_dir / "incidents.jsonl", IncidentRecord)
    families = read_jsonl(args.dataset_dir / "families.jsonl", IncidentFamily)
    taxonomy = load_taxonomy(args.taxonomy)
    report = audit_dataset(
        records,
        families,
        taxonomy,
        research_mode=args.research_mode,
    )
    rendered = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
