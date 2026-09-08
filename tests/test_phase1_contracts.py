from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.phase1 import (
    ContractError,
    category_for_label,
    load_json_yaml,
    validate_experiment_can_run,
    validate_label_taxonomy,
    validate_model_config,
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
        if pipeline in {"RAG", "COMBINED"}:
            experiment.update(
                {
                    "knowledge_base_version": "kb-v1",
                    "embedding_model_revision": "embed@sha",
                    "chunker_version": "chunker-v1",
                    "chunk_size": 512,
                    "overlap": 64,
                    "top_k": 5,
                }
            )
        if pipeline in {"FINETUNED", "COMBINED"}:
            experiment.update({"adapter_id": "adapter-v1", "adapter_revision": "adapter-sha"})
        return experiment

    def test_taxonomy_is_valid(self) -> None:
        validate_label_taxonomy(self.taxonomy)

    def test_model_config_is_valid(self) -> None:
        validate_model_config(self.model)

    def test_duplicate_label_is_rejected(self) -> None:
        bad = copy.deepcopy(self.taxonomy)
        bad["labels"].append(copy.deepcopy(bad["labels"][0]))
        with self.assertRaises(ContractError):
            validate_label_taxonomy(bad)

    def test_duplicate_category_is_rejected(self) -> None:
        bad = copy.deepcopy(self.taxonomy)
        bad["categories"].append(copy.deepcopy(bad["categories"][0]))
        with self.assertRaises(ContractError):
            validate_label_taxonomy(bad)

    def test_noncanonical_label_format_is_rejected(self) -> None:
        bad = copy.deepcopy(self.taxonomy)
        bad["labels"][0]["id"] = "N Plus One"
        with self.assertRaises(ContractError):
            validate_label_taxonomy(bad)

    def test_unknown_root_cause_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            validate_root_cause_code("invented_root_cause", self.taxonomy)

    def test_known_root_cause_is_accepted(self) -> None:
        validate_root_cause_code("n_plus_one_query", self.taxonomy)
        self.assertEqual(category_for_label("n_plus_one_query", self.taxonomy), "database_behavior")

    def test_study_contract_is_valid(self) -> None:
        validate_study_config(self.study, self.model, self.taxonomy)

    def test_study_rejects_taxonomy_version_mismatch(self) -> None:
        bad = copy.deepcopy(self.study)
        bad["label_taxonomy_version"] = "999.0.0"
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)

    def test_model_revision_must_be_full_git_sha(self) -> None:
        bad = copy.deepcopy(self.model)
        bad["base_model_revision"] = "short-revision"
        with self.assertRaises(ContractError):
            validate_model_config(bad)

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
        experiment["base_model_revision"] = "0" * 40
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_zero_shot_cannot_secretly_enable_adapter(self) -> None:
        experiment = self.valid_experiment("ZERO_SHOT")
        experiment["adapter_id"] = "hidden-adapter"
        experiment["adapter_revision"] = "hidden-adapter-sha"
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_rag_requires_retrieval_reproducibility_fields(self) -> None:
        experiment = self.valid_experiment("RAG")
        experiment["knowledge_base_version"] = None
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_finetuned_requires_adapter_revision(self) -> None:
        experiment = self.valid_experiment("FINETUNED")
        experiment["adapter_revision"] = None
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_combined_requires_both_adapter_and_retrieval_identity(self) -> None:
        experiment = self.valid_experiment("COMBINED")
        validate_experiment_can_run(experiment, self.study, self.model)
        experiment["top_k"] = None
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_primary_comparison_rejects_model_revision_mismatch(self) -> None:
        baseline = self.valid_experiment("ZERO_SHOT")
        rag = self.valid_experiment("RAG")
        rag["base_model_revision"] = "0" * 40
        with self.assertRaises(ContractError):
            validate_primary_comparison([baseline, rag], self.study, self.model)

    def test_primary_comparison_rejects_test_manifest_mismatch(self) -> None:
        baseline = self.valid_experiment("ZERO_SHOT")
        rag = self.valid_experiment("RAG")
        rag["test_split_manifest_checksum"] = "different-test-set"
        with self.assertRaises(ContractError):
            validate_primary_comparison([baseline, rag], self.study, self.model)

    def test_all_four_valid_pipeline_identities_can_be_compared(self) -> None:
        rows = [self.valid_experiment(name) for name in ("ZERO_SHOT", "RAG", "FINETUNED", "COMBINED")]
        validate_primary_comparison(rows, self.study, self.model)

    def test_load_json_yaml_rejects_non_object(self) -> None:
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.yaml"
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(ContractError):
                load_json_yaml(path)

    def test_taxonomy_rejects_invalid_semver(self) -> None:
        bad = copy.deepcopy(self.taxonomy)
        bad["taxonomy_version"] = "v1"
        with self.assertRaises(ContractError):
            validate_label_taxonomy(bad)

    def test_taxonomy_requires_change_policy(self) -> None:
        bad = copy.deepcopy(self.taxonomy)
        bad["change_policy"] = ""
        with self.assertRaises(ContractError):
            validate_label_taxonomy(bad)

    def test_taxonomy_requires_nonempty_categories_and_labels(self) -> None:
        bad = copy.deepcopy(self.taxonomy)
        bad["categories"] = []
        with self.assertRaises(ContractError):
            validate_label_taxonomy(bad)
        bad = copy.deepcopy(self.taxonomy)
        bad["labels"] = []
        with self.assertRaises(ContractError):
            validate_label_taxonomy(bad)

    def test_taxonomy_rejects_non_object_category_or_label(self) -> None:
        bad = copy.deepcopy(self.taxonomy)
        bad["categories"][0] = "database_behavior"
        with self.assertRaises(ContractError):
            validate_label_taxonomy(bad)
        bad = copy.deepcopy(self.taxonomy)
        bad["labels"][0] = "n_plus_one_query"
        with self.assertRaises(ContractError):
            validate_label_taxonomy(bad)

    def test_taxonomy_rejects_invalid_category_id_and_unknown_category_reference(self) -> None:
        bad = copy.deepcopy(self.taxonomy)
        bad["categories"][0]["id"] = "Database Behavior"
        with self.assertRaises(ContractError):
            validate_label_taxonomy(bad)
        bad = copy.deepcopy(self.taxonomy)
        bad["labels"][0]["category"] = "missing_category"
        with self.assertRaises(ContractError):
            validate_label_taxonomy(bad)

    def test_model_rejects_missing_field_unfrozen_and_invalid_hub_id(self) -> None:
        bad = copy.deepcopy(self.model)
        del bad["selection_record"]
        with self.assertRaises(ContractError):
            validate_model_config(bad)
        bad = copy.deepcopy(self.model)
        bad["frozen"] = False
        with self.assertRaises(ContractError):
            validate_model_config(bad)
        bad = copy.deepcopy(self.model)
        bad["base_model_id"] = "local-model"
        with self.assertRaises(ContractError):
            validate_model_config(bad)

    def test_model_smoke_must_be_hard_gate(self) -> None:
        bad = copy.deepcopy(self.model)
        bad["smoke_test"]["required_for_phase_complete"] = False
        with self.assertRaises(ContractError):
            validate_model_config(bad)

    def test_study_rejects_model_id_mismatch_and_noncanonical_pipeline_set(self) -> None:
        bad = copy.deepcopy(self.study)
        bad["study_id"] = "other-study"
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)
        bad = copy.deepcopy(self.study)
        bad["pipelines"] = ["ZERO_SHOT", "RAG", "FINETUNED"]
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)

    def test_study_rejects_wrong_metric_unlock_split_or_missing_hypotheses(self) -> None:
        bad = copy.deepcopy(self.study)
        bad["primary_metric"] = "semantic_similarity"
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)
        bad = copy.deepcopy(self.study)
        bad["locked_test"] = False
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)
        bad = copy.deepcopy(self.study)
        bad["split_unit"] = "row"
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)
        bad = copy.deepcopy(self.study)
        bad["hypotheses_record"] = None
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)

    def test_study_rejects_bad_required_field_declarations(self) -> None:
        bad = copy.deepcopy(self.study)
        bad["required_experiment_fields"] = []
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)
        bad = copy.deepcopy(self.study)
        bad["required_experiment_fields"].append(bad["required_experiment_fields"][0])
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)
        bad = copy.deepcopy(self.study)
        bad["required_non_null_run_fields"] = []
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)
        bad = copy.deepcopy(self.study)
        bad["required_non_null_run_fields"].append(bad["required_non_null_run_fields"][0])
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)
        bad = copy.deepcopy(self.study)
        bad["required_non_null_run_fields"].append("not_a_declared_field")
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)

    def test_study_rejects_bad_pipeline_rule_declarations(self) -> None:
        bad = copy.deepcopy(self.study)
        del bad["pipeline_required_fields"]["RAG"]
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)
        bad = copy.deepcopy(self.study)
        bad["pipeline_required_fields"]["RAG"] = "knowledge_base_version"
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)
        bad = copy.deepcopy(self.study)
        bad["pipeline_required_fields"]["RAG"].append("unknown_field")
        with self.assertRaises(ContractError):
            validate_study_config(bad, self.model, self.taxonomy)

    def test_experiment_rejects_wrong_study_pipeline_and_base_model_id(self) -> None:
        experiment = self.valid_experiment()
        experiment["study_id"] = "wrong-study"
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)
        experiment = self.valid_experiment()
        experiment["pipeline_type"] = "OTHER"
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)
        experiment = self.valid_experiment()
        experiment["base_model_id"] = "other/model"
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_reranker_id_and_revision_must_move_together(self) -> None:
        experiment = self.valid_experiment("RAG")
        experiment["reranker_id"] = "reranker"
        experiment["reranker_revision"] = None
        with self.assertRaises(ContractError):
            validate_experiment_can_run(experiment, self.study, self.model)

    def test_primary_comparison_rejects_empty_input(self) -> None:
        with self.assertRaises(ContractError):
            validate_primary_comparison([], self.study, self.model)


if __name__ == "__main__":
    unittest.main()
