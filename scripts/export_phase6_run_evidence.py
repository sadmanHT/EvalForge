#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.db import build_engine, session_scope  # noqa: E402
from app.inference.evidence import (  # noqa: E402
    export_phase6_run_evidence,
    validate_run_evidence,
)
from app.inference.protocol import load_baseline_protocol  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export deterministic Phase 06 run evidence from PostgreSQL."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    protocol = load_baseline_protocol(ROOT)
    engine = build_engine()
    with session_scope(engine) as session:
        evidence = export_phase6_run_evidence(
            session,
            root=ROOT,
            run_id=args.run_id,
            protocol=protocol,
        )
    validate_run_evidence(
        evidence,
        root=ROOT,
        protocol=protocol,
        expected_split=args.split,
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("PHASE06_RUN_EVIDENCE_EXPORT=PASS")
    print(f"RUN_ID={args.run_id}")
    print(f"SPLIT={args.split}")
    print(f"EVIDENCE_SHA256={evidence['evidence_sha256']}")
    print(f"OUTPUT={output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
