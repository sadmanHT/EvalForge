from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.phase1 import (  # noqa: E402
    ContractError,
    load_json_yaml,
    validate_experiment_can_run,
    validate_label_taxonomy,
    validate_primary_comparison,
    validate_root_cause_code,
    validate_study_config,
)


class Phase01ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomy = load_json_yaml(ROOT / "configs/label-taxonomy.yaml")
        cls.model = load_json_yaml(ROOT / "configs/model.yaml")
        cls.study = load_json_yaml(ROOT / "configs/study.yaml")

    def valid_experiment(self, pipeline: str = "ZERO_SHOT") -> dict:
        experiment = {key: None for key in self.study["required_experiment_fields"]}
        experiment.update(
            {
                "experiment_id": f"exp-{pipeline.lower()}",
                "study_id": self.study["study_id"],
                "pipeline_type": pipeline,
                "dataset_version": "dataset-fixture-v1",
                "test_split_manifest_checksum": "abc123",
                "label_taxonomy_version": self.taxonomy["taxonomy_version"],
                "base_model_id": self.model["base_model_id"],
                "base_model_revision": self.model["base_model_revision"],
                "prompt_version": "prompt-v1",
                "output_schema_version": "prediction-v1",
                "generation_config": {"temperature": 0.0, "max_new_tokens": 96},
                "confidence_method": self.study["confidence_method"],
                "seed": 7,
                "evaluator_version": "eval-v1",
                "git_commit": "deadbeef",
                "dependency_lock_checksum": "lock123",
                "hardware_runtime_descriptor": "fixture-cpu",
                "cost_rate_snapshot_version": "rates-v1",
                "created_at": "2026-09-08T00:00:00Z",
                "config_hash": "hash123",
            }
        )
        return experiment

    def test_taxonomy_is_valid(self) -> None:
        validate_label_taxonomy(self.taxonomy)

    def test_duplicate_label_is_rejected(self) -> None:
        bad = copy.deepcopy(self.taxonomy)
        bad["labels"].append(copy.deepcopy(bad["labels"][0]))
        with self.assertRaises(ContractError):
            validate_label_taxonomy(bad)

    def test_unknown_root_cause_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            validate_root_cause_code("invented_root_cause", self.taxonomy)

    def test_known_root_cause_is_accepted(self) -> None:
        validate_root_cause_code("n_plus_one_query", self.taxonomy)

    def test_study_contract_is_valid(self) -> None:
        validate_study_config(self.study, self.model, self.taxonomy)

    def test_study_rejects_taxonomy_version_mismatch(self) -> None:
        bad_study = copy.deepcopy(self.study)
        bad_study["label_taxonomy_version"] = "999.0.0"
        with self.assertRaises(ContractError):
            validate_study_config(bad_study, self.model, self.taxonomy)

    def test_experiment_missing_reproducibility_field_cannot_run(self) -> None:
        experiment = self.valid_experiment()
        del experiment["hardware_runtime_descriptor"]
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_required_reproducibility_field_cannot_be_null(self) -> None:
        experiment = self.valid_experiment()
        experiment["hardware_runtime_descriptor"] = None
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_required_reproducibility_string_cannot_be_blank(self) -> None:
        experiment = self.valid_experiment()
        experiment["dataset_version"] = "   "
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_wrong_taxonomy_version_cannot_run(self) -> None:
        experiment = self.valid_experiment()
        experiment["label_taxonomy_version"] = "999.0.0"
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_wrong_confidence_method_cannot_run(self) -> None:
        experiment = self.valid_experiment()
        experiment["confidence_method"] = "self_reported"
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_wrong_model_revision_cannot_run(self) -> None:
        experiment = self.valid_experiment()
        experiment["base_model_revision"] = "different-revision"
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_primary_comparison_rejects_model_revision_mismatch(self) -> None:
        baseline = self.valid_experiment("ZERO_SHOT")
        rag = self.valid_experiment("RAG")
        rag["base_model_revision"] = "different-revision"
        with self.assertRaises(ContractError):
            validate_primary_comparison([baseline, rag], self.study, self.model)

    def test_primary_comparison_rejects_test_manifest_mismatch(self) -> None:
        baseline = self.valid_experiment("ZERO_SHOT")
        rag = self.valid_experiment("RAG")
        rag["test_split_manifest_checksum"] = "different-test-set"
        with self.assertRaises(ContractError):
            validate_primary_comparison([baseline, rag], self.study, self.model)


if __name__ == "__main__":
    unittest.main()
