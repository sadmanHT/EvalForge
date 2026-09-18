#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.inference.finetuned_comparison import (  # noqa: E402
    build_paired_baseline_finetuned_comparison,
    validate_paired_baseline_finetuned_comparison,
)
from app.inference.finetuned_protocol import load_phase10_protocol  # noqa: E402
from app.inference.protocol import load_baseline_protocol  # noqa: E402
from app.training.evidence import sha256_file  # noqa: E402


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return {str(key): value for key, value in payload.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the sealed paired Phase 06 zero-shot versus Phase 10 fine-tuned "
            "locked-test comparison."
        )
    )
    parser.add_argument(
        "--baseline-evidence",
        type=Path,
        default=Path("evidence/phase-06/test-run.json"),
    )
    parser.add_argument(
        "--finetuned-evidence",
        type=Path,
        default=Path("evidence/phase-10/locked-test/phase10-finetuned-test.json"),
    )
    parser.add_argument(
        "--finetuned-source-commit",
        default="c59910e00f5e4fd0a722d2796da416c977753ddd",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/phase-10/baseline-finetuned-comparison.json"),
    )
    args = parser.parse_args()

    baseline_path = _resolve(args.baseline_evidence)
    finetuned_path = _resolve(args.finetuned_evidence)
    baseline_evidence = _load_json(baseline_path)
    finetuned_evidence = _load_json(finetuned_path)

    baseline_protocol = load_baseline_protocol(ROOT)
    finetuned_protocol = load_phase10_protocol(ROOT)
    finetuned_file_sha256 = sha256_file(finetuned_path)

    comparison = build_paired_baseline_finetuned_comparison(
        root=ROOT,
        baseline_protocol=baseline_protocol,
        finetuned_protocol=finetuned_protocol,
        baseline_evidence=baseline_evidence,
        finetuned_evidence=finetuned_evidence,
        finetuned_evidence_file_sha256=finetuned_file_sha256,
        expected_finetuned_source_commit=args.finetuned_source_commit,
    )
    validate_paired_baseline_finetuned_comparison(
        comparison,
        root=ROOT,
        baseline_protocol=baseline_protocol,
        finetuned_protocol=finetuned_protocol,
        baseline_evidence=baseline_evidence,
        finetuned_evidence=finetuned_evidence,
        finetuned_evidence_file_sha256=finetuned_file_sha256,
        expected_finetuned_source_commit=args.finetuned_source_commit,
    )

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE10_BASELINE_FINETUNED_COMPARISON=PASS")
    print(f"COMPARISON_SHA256={comparison['comparison_sha256']}")
    print(f"OUTPUT={output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
