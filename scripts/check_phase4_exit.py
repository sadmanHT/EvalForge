#!/usr/bin/env python
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import psycopg

from app.config import Settings
from app.core.experiment_config import experiment_config_hash
from app.persistence_seed import build_smoke_config

EXPECTED_TABLES = {
    "adapter_versions",
    "artifacts",
    "cost_records",
    "dataset_versions",
    "experiments",
    "failure_annotations",
    "incident_families",
    "incidents",
    "jobs",
    "kb_chunks",
    "kb_documents",
    "knowledge_base_versions",
    "metrics",
    "model_versions",
    "predictions",
    "retrieval_traces",
    "runs",
}
EXPECTED_TRIGGERS = {
    "adapter_versions_immutable",
    "dataset_versions_immutable",
    "experiments_freeze_completed",
    "knowledge_base_versions_immutable",
    "model_versions_immutable",
}
DATASET_VERSION = "evalforge-incident-diagnosis-v0.1.0"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        detail = f"actual={actual!r} expected={expected!r}"
        raise SystemExit(f"PHASE04_EXIT_GATE=FAIL {message}: {detail}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    dataset_path = (
        root / "datasets/incident_diagnosis/processed" / DATASET_VERSION / "manifest.json"
    )
    dataset = _load_json(dataset_path)
    model = _load_json(root / "configs/model.yaml")
    smoke_config = build_smoke_config(root, dataset, model)
    expected_config_hash = experiment_config_hash(smoke_config)

    with (
        psycopg.connect(Settings.from_env().database_url) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute("SELECT version_num FROM alembic_version")
        migration_row = cursor.fetchone()
        if migration_row is None or not migration_row[0]:
            raise SystemExit("PHASE04_EXIT_GATE=FAIL migration head missing")
        migration_head = str(migration_row[0])

        cursor.execute("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
        _assert_equal(cursor.fetchone(), (1,), "pgvector extension missing")

        cursor.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        )
        tables = {row[0] for row in cursor.fetchall()}
        missing_tables = sorted(EXPECTED_TABLES - tables)
        if missing_tables:
            raise SystemExit(f"PHASE04_EXIT_GATE=FAIL missing persistence tables={missing_tables}")

        cursor.execute(
            """
            SELECT manifest_checksum, content_checksum, record_count, family_count,
                   label_taxonomy_version, research_ready
            FROM dataset_versions
            WHERE version = %s
            """,
            (DATASET_VERSION,),
        )
        dataset_row = cursor.fetchone()
        if dataset_row is None:
            raise SystemExit("PHASE04_EXIT_GATE=FAIL locked Phase 03 dataset is not imported")
        _assert_equal(
            dataset_row[0],
            dataset["manifest_checksum"],
            "manifest checksum mismatch",
        )
        _assert_equal(dataset_row[1], dataset["content_checksum"], "content checksum mismatch")
        _assert_equal(dataset_row[2], 18, "record count mismatch")
        _assert_equal(dataset_row[3], 18, "family count mismatch")
        _assert_equal(
            dataset_row[4],
            dataset["label_taxonomy_version"],
            "label taxonomy version mismatch",
        )
        _assert_equal(dataset_row[5], True, "research readiness mismatch")

        cursor.execute(
            """
            SELECT split, count(*)
            FROM incidents
            WHERE dataset_version = %s
            GROUP BY split
            """,
            (DATASET_VERSION,),
        )
        incident_splits = Counter(dict(cursor.fetchall()))
        cursor.execute(
            """
            SELECT split, count(*)
            FROM incident_families
            WHERE dataset_version = %s
            GROUP BY split
            """,
            (DATASET_VERSION,),
        )
        family_splits = Counter(dict(cursor.fetchall()))
        expected_splits = Counter({"train": 6, "validation": 6, "test": 6})
        _assert_equal(incident_splits, expected_splits, "incident split round-trip mismatch")
        _assert_equal(family_splits, expected_splits, "family split round-trip mismatch")

        cursor.execute("SELECT tgname FROM pg_trigger WHERE NOT tgisinternal")
        triggers = {row[0] for row in cursor.fetchall()}
        missing_triggers = sorted(EXPECTED_TRIGGERS - triggers)
        if missing_triggers:
            detail = f"historical immutability triggers missing={missing_triggers}"
            raise SystemExit(f"PHASE04_EXIT_GATE=FAIL {detail}")

        cursor.execute(
            """
            SELECT experiment_id, config_hash, dataset_version, pipeline_type,
                   test_split_manifest_checksum, label_taxonomy_version
            FROM experiments
            WHERE config_hash = %s
            """,
            (expected_config_hash,),
        )
        experiment = cursor.fetchone()
        if experiment is None:
            raise SystemExit("PHASE04_EXIT_GATE=FAIL canonical smoke experiment is not seeded")
        _assert_equal(
            experiment[0],
            "exp-phase4-persistence-smoke",
            "smoke experiment identity mismatch",
        )
        _assert_equal(experiment[1], expected_config_hash, "experiment config hash mismatch")
        _assert_equal(experiment[2], DATASET_VERSION, "experiment dataset mismatch")
        _assert_equal(experiment[3], "ZERO_SHOT", "smoke pipeline mismatch")
        _assert_equal(
            experiment[4],
            dataset["manifest_checksum"],
            "experiment test manifest mismatch",
        )
        _assert_equal(
            experiment[5],
            dataset["label_taxonomy_version"],
            "experiment taxonomy mismatch",
        )

        cursor.execute(
            """
            SELECT model_id, revision
            FROM model_versions
            WHERE model_id = %s AND revision = %s
            """,
            (model["base_model_id"], model["base_model_revision"]),
        )
        _assert_equal(
            cursor.fetchone(),
            (model["base_model_id"], model["base_model_revision"]),
            "frozen base-model identity missing",
        )

        cursor.execute(
            """
            SELECT count(*) FROM jobs
            WHERE idempotency_key = 'phase4-persistence-smoke-v1'
            """
        )
        _assert_equal(cursor.fetchone(), (1,), "idempotent smoke job missing or duplicated")

    print("PHASE04_EXIT_GATE=PASS")
    print(f"ALEMBIC_HEAD={migration_head}")
    print(f"PERSISTENCE_TABLES={len(EXPECTED_TABLES)}")
    print(f"DATASET_VERSION={DATASET_VERSION}")
    print("DATASET_RECORDS=18")
    print("DATASET_FAMILIES=18")
    print("DATASET_TRAIN=6")
    print("DATASET_VALIDATION=6")
    print("DATASET_TEST=6")
    print(f"CONTENT_CHECKSUM={dataset['content_checksum']}")
    print(f"MANIFEST_CHECKSUM={dataset['manifest_checksum']}")
    print(f"EXPERIMENT_CONFIG_HASH={expected_config_hash}")
    print(f"BASE_MODEL_ID={model['base_model_id']}")
    print(f"BASE_MODEL_REVISION={model['base_model_revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
