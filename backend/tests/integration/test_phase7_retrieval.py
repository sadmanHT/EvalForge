from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.engine import Engine

from app.config import Settings
from app.db import build_engine, session_scope, sqlalchemy_database_url
from app.models import KBChunk
from app.retrieval.embeddings import HashEmbeddingAdapter
from app.retrieval.indexing import build_index_plan, load_kb_config, persist_index
from app.retrieval.search import RetrievalQuery, research_document_allowed, search_chunks

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"


def _alembic_config() -> Config:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "migrations"))
    config.set_main_option(
        "sqlalchemy.url",
        sqlalchemy_database_url(Settings.from_env().database_url),
    )
    return config


@pytest.fixture(scope="module")
def engine() -> Engine:
    engine = build_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    command.upgrade(_alembic_config(), "head")
    yield engine
    engine.dispose()


def _incidents(dataset_version: str) -> list[dict[str, Any]]:
    path = (
        ROOT
        / "datasets/incident_diagnosis/processed"
        / dataset_version
        / "incidents.jsonl"
    )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_real_kb_reindex_is_logically_idempotent(engine: Engine) -> None:
    config = load_kb_config(ROOT / "configs/phase7-kb.json")
    plan = build_index_plan(ROOT, config)

    with session_scope(engine) as session:
        first_counts = persist_index(session, plan)
    with session_scope(engine) as session:
        first_ids = tuple(
            session.scalars(
                select(KBChunk.chunk_id)
                .where(KBChunk.kb_version == config.kb_version)
                .order_by(KBChunk.chunk_id)
            )
        )
        second_counts = persist_index(session, plan)
    with session_scope(engine) as session:
        second_ids = tuple(
            session.scalars(
                select(KBChunk.chunk_id)
                .where(KBChunk.kb_version == config.kb_version)
                .order_by(KBChunk.chunk_id)
            )
        )

    assert first_counts == second_counts == (len(plan.documents), len(plan.chunks))
    assert first_ids == second_ids


def test_pgvector_nearest_neighbor_and_research_filter(engine: Engine) -> None:
    config = load_kb_config(ROOT / "configs/phase7-kb.json")
    plan = build_index_plan(ROOT, config)
    adapter = HashEmbeddingAdapter(config.embedding)
    with session_scope(engine) as session:
        persist_index(session, plan)

    generic = next(item for item in plan.documents if item.source_type == "runbook")
    generic_chunk = next(item for item in plan.chunks if item.document_id == generic.document_id)
    with session_scope(engine) as session:
        exact = search_chunks(
            session,
            adapter,
            RetrievalQuery(
                kb_version=config.kb_version,
                text=generic_chunk.text,
                top_k=1,
                research_mode=False,
            ),
        )
    assert exact[0].chunk_id == generic_chunk.chunk_id
    assert exact[0].score == pytest.approx(1.0, abs=1e-8)

    heldout = next(
        item
        for item in _incidents(config.dataset_version)
        if item["split"] == "validation"
    )
    family_id = str(heldout["incident_family_id"])
    heldout_document = next(
        item for item in plan.documents if item.source_family_id == family_id
    )
    heldout_chunk = next(
        item for item in plan.chunks if item.document_id == heldout_document.document_id
    )
    query_text = heldout_chunk.text

    with session_scope(engine) as session:
        unrestricted = search_chunks(
            session,
            adapter,
            RetrievalQuery(
                kb_version=config.kb_version,
                text=query_text,
                top_k=1,
                research_mode=False,
            ),
        )
        filtered = search_chunks(
            session,
            adapter,
            RetrievalQuery(
                kb_version=config.kb_version,
                text=query_text,
                top_k=config.default_top_k,
                research_mode=True,
                query_family_id=family_id,
            ),
        )

    assert unrestricted[0].source_family_id == family_id
    assert all(result.source_family_id != family_id for result in filtered)
    assert all(
        research_document_allowed(result.document_metadata, query_family_id=family_id)
        for result in filtered
    )
