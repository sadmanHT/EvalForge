#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.inference.protocol import load_baseline_protocol  # noqa: E402
from app.inference.rag_comparison import (  # noqa: E402
    build_paired_baseline_rag_comparison,
    validate_paired_baseline_rag_comparison,
)
from app.inference.rag_protocol import load_rag_protocol  # noqa: E402


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        text = gzip.decompress(path.read_bytes()).decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return {str(key): value for key, value in payload.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a sealed paired Phase 06 baseline versus Phase 08 RAG comparison."
    )
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--rag-evidence", required=True, type=Path)
    parser.add_argument("--baseline-evidence", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline_path = args.baseline_evidence
    if baseline_path is None:
        baseline_path = Path(f"evidence/phase-06/{args.split}-run.json")
    baseline_evidence = _load_json(_resolve(baseline_path))
    rag_evidence = _load_json(_resolve(args.rag_evidence))

    baseline_protocol = load_baseline_protocol(ROOT)
    rag_protocol, suite = load_rag_protocol(ROOT)
    variant_id = rag_evidence.get("variant_id")
    if not isinstance(variant_id, str):
        raise ValueError("RAG evidence is missing variant_id")
    variant = rag_protocol.assert_variant_allowed(
        split=args.split,
        variant_id=variant_id,
        suite=suite,
    )

    comparison = build_paired_baseline_rag_comparison(
        root=ROOT,
        split=args.split,
        baseline_protocol=baseline_protocol,
        rag_protocol=rag_protocol,
        rag_variant=variant,
        baseline_evidence=baseline_evidence,
        rag_evidence=rag_evidence,
    )
    validate_paired_baseline_rag_comparison(
        comparison,
        root=ROOT,
        split=args.split,
        baseline_protocol=baseline_protocol,
        rag_protocol=rag_protocol,
        rag_variant=variant,
        baseline_evidence=baseline_evidence,
        rag_evidence=rag_evidence,
    )

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE08_BASELINE_RAG_COMPARISON=PASS")
    print(f"SPLIT={args.split}")
    print(f"RAG_VARIANT_ID={variant.variant_id}")
    print(f"COMPARISON_SHA256={comparison['comparison_sha256']}")
    print(f"OUTPUT={output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
