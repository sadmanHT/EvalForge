from __future__ import annotations

from copy import deepcopy

from app.data.schemas import IncidentRecord, Split


def make_synthetic_child(
    parent: IncidentRecord,
    *,
    incident_id: str,
    title: str,
    description: str,
    generator_model_id: str,
    generator_model_revision: str,
    generator_prompt_version: str,
) -> IncidentRecord:
    if parent.split != Split.TRAIN:
        raise ValueError("synthetic augmentation may only use training parents")
    if parent.is_synthetic:
        raise ValueError("synthetic augmentation must descend directly from an independent parent")

    payload = parent.model_dump(mode="python")
    payload.update(
        {
            "incident_id": incident_id,
            "title": title,
            "description": description,
            "split": Split.TRAIN,
            "is_synthetic": True,
            "parent_incident_id": parent.incident_id,
            "generator_model_id": generator_model_id,
            "generator_model_revision": generator_model_revision,
            "generator_prompt_version": generator_prompt_version,
            "metadata": {
                **deepcopy(parent.metadata),
                "augmentation_parent": parent.incident_id,
            },
        }
    )
    return IncidentRecord.model_validate(payload)
