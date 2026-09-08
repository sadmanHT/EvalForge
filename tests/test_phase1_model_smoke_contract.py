from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.model_smoke import extract_json, validate_smoke_payload


class Phase01ModelSmokeContractTests(unittest.TestCase):
    def test_extracts_strict_json_object(self) -> None:
        payload = extract_json('{"root_cause_code":"n_plus_one_query","reasoning":"query explosion"}')
        self.assertEqual(payload["root_cause_code"], "n_plus_one_query")

    def test_rejects_non_json_text(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_json("the answer is n_plus_one_query")

    def test_rejects_unknown_smoke_label(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_smoke_payload({"root_cause_code": "invented", "reasoning": "x"})

    def test_rejects_empty_reasoning(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_smoke_payload({"root_cause_code": "n_plus_one_query", "reasoning": ""})

    def test_accepts_canonical_smoke_payload(self) -> None:
        validate_smoke_payload({"root_cause_code": "n_plus_one_query", "reasoning": "query count rose 18x"})


if __name__ == "__main__":
    unittest.main()
