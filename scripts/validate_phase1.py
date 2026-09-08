#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.phase1 import (  # noqa: E402
    load_json_yaml,
    validate_label_taxonomy,
    validate_model_config,
    validate_study_config,
)


def main() -> int:
    taxonomy = load_json_yaml(ROOT / "configs/label-taxonomy.yaml")
    model = load_json_yaml(ROOT / "configs/model.yaml")
    study = load_json_yaml(ROOT / "configs/study.yaml")

    validate_label_taxonomy(taxonomy)
    validate_model_config(model)
    validate_study_config(study, model, taxonomy)

    required_docs = [
        "docs/research-protocol.md",
        "docs/architecture.md",
        "docs/quality-gates.md",
        "datasets/schema.md",
    ]
    for rel in required_docs:
        path = ROOT / rel
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"required artifact missing/empty: {rel}")

    adrs = sorted((ROOT / "docs/adrs").glob("ADR-*.md"))
    if len(adrs) < 9:
        raise RuntimeError(f"expected at least 9 ADRs, found {len(adrs)}")

    print("Phase 01 static contract validation: PASS")
    print(f"taxonomy labels: {len(taxonomy['labels'])}")
    print(f"ADRs: {len(adrs)}")
    print(f"frozen model: {model['base_model_id']}@{model['base_model_revision']}")
    print("NOTE: full model weight-load/generation smoke remains a separate hard gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
