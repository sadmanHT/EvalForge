from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Phase01ArtifactTests(unittest.TestCase):
    def test_required_phase_documents_exist(self) -> None:
        expected = [
            "docs/research-protocol.md",
            "docs/model-selection.md",
            "docs/architecture.md",
            "docs/quality-gates.md",
            "datasets/schema.md",
            "configs/model.yaml",
        ]
        for relative in expected:
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertGreater(path.stat().st_size, 0, relative)

    def test_nine_adrs_exist(self) -> None:
        self.assertGreaterEqual(len(list((ROOT / "docs/adrs").glob("ADR-*.md"))), 9)

    def test_phase_handoff_does_not_claim_completion_before_model_smoke(self) -> None:
        text = (ROOT / "docs/phase-01-handoff.md").read_text(encoding="utf-8")
        self.assertIn("NOT COMPLETE", text)
        self.assertIn("model-smoke.json", text)


if __name__ == "__main__":
    unittest.main()
