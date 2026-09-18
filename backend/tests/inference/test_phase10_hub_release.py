from pathlib import Path

from app.training.hub_release import (
    build_phase10_model_card,
    validate_release_revision,
)

ROOT = Path(__file__).resolve().parents[3]


def test_phase10_model_card_is_bound_to_frozen_evidence() -> None:
    card = build_phase10_model_card(
        root=ROOT,
        repo_id="example/evalforge-mistral7b-incident-lora",
    )
    assert "mistralai/Mistral-7B-Instruct-v0.3" in card
    assert "e8ebf0c51d241516bd3c6bb44e476df6d412aaf6926705ecc53ca8cc3fcec065" in card
    assert "Fine-tuned locked test | 0.833333 | 0.166667" in card
    assert "Zero-shot locked test | 1.000000 | 0.000000" in card
    assert "post-test retuning" in card
    assert "does not support a broad superiority claim" in card


def test_hub_revision_requires_immutable_commit_hash() -> None:
    good = "a" * 40
    assert validate_release_revision(good) == good
    for bad in ("main", "latest", "a" * 39, "g" * 40):
        try:
            validate_release_revision(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted mutable/invalid revision: {bad}")
