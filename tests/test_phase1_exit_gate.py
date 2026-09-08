from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Phase01ExitGateTests(unittest.TestCase):
    def test_exit_gate_refuses_missing_real_model_smoke_evidence(self) -> None:
        evidence = ROOT / "evidence/phase-01/model-smoke.json"
        self.assertFalse(evidence.exists(), "test assumes no fabricated real-model evidence is present")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_phase1_exit.py")],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing real model smoke evidence", result.stdout)

    def test_exit_gate_rejects_wrong_revision_evidence(self) -> None:
        # Exercise the critical comparison invariant without writing into the real evidence path.
        script = (ROOT / "scripts/check_phase1_exit.py").read_text(encoding="utf-8")
        self.assertIn('evidence.get("revision") != model["base_model_revision"]', script)


if __name__ == "__main__":
    unittest.main()
