from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("phase1_exit_gate", ROOT / "scripts/check_phase1_exit.py")
assert spec and spec.loader
exit_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exit_gate)


class Phase01ExitGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence_path = ROOT / "evidence/phase-01/model-smoke.json"
        cls.evidence = json.loads(cls.evidence_path.read_text(encoding="utf-8"))
        cls.model = exit_gate.load_json_yaml(ROOT / "configs/model.yaml")
        cls.taxonomy = exit_gate.load_json_yaml(ROOT / "configs/label-taxonomy.yaml")

    def test_exit_gate_accepts_preserved_real_gpu_evidence(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_phase1_exit.py")],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PHASE01_EXIT_GATE=PASS", result.stdout)
        self.assertIn("Tesla T4", result.stdout)
        self.assertIn("n_plus_one_query", result.stdout)

    def test_exit_gate_rejects_wrong_resolved_revision(self) -> None:
        bad = copy.deepcopy(self.evidence)
        bad["model"]["resolved_revision"] = "0" * 40
        bad["evidence_sha256"] = exit_gate._canonical_evidence_hash(bad)
        with self.assertRaisesRegex(exit_gate.GateError, "resolved_revision"):
            exit_gate.validate_smoke_evidence(bad, self.model, self.taxonomy, ROOT / "evidence/phase-01")

    def test_exit_gate_rejects_tampered_internal_evidence_hash(self) -> None:
        bad = copy.deepcopy(self.evidence)
        bad["incident"] += " tampered"
        with self.assertRaisesRegex(exit_gate.GateError, "evidence_sha256"):
            exit_gate.validate_smoke_evidence(bad, self.model, self.taxonomy, ROOT / "evidence/phase-01")

    def test_exit_gate_rejects_noncanonical_structured_output(self) -> None:
        bad = copy.deepcopy(self.evidence)
        bad["parsed_output"]["root_cause_code"] = "invented_root_cause"
        bad["raw_output"] = json.dumps(bad["parsed_output"])
        bad["evidence_sha256"] = exit_gate._canonical_evidence_hash(bad)
        with self.assertRaisesRegex(exit_gate.GateError, "non-canonical"):
            exit_gate.validate_smoke_evidence(bad, self.model, self.taxonomy, ROOT / "evidence/phase-01")

    def test_evidence_bundle_checksums_are_valid(self) -> None:
        exit_gate._validate_bundle_checksums(ROOT / "evidence/phase-01")


if __name__ == "__main__":
    unittest.main()
