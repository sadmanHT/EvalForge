"""Phase 07 retrieval indexes.

Revision ID: 0003_phase07
Revises: 0002_phase04
"""

from __future__ import annotations

from typing import Sequence

from alembic import op

revision: str = "0003_phase07"
down_revision: str | None = "0002_phase04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DDL = [
    """
    CREATE INDEX IF NOT EXISTS ix_kb_chunks_embedding_hnsw_cosine
    ON kb_chunks USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_kb_chunks_version_family
    ON kb_chunks (kb_version, source_family_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_kb_documents_version_source_split
    ON kb_documents (kb_version, (document_metadata ->> 'source_split'))
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_kb_documents_version_source_type
    ON kb_documents (kb_version, (document_metadata ->> 'source_type'))
    """,
]


def upgrade() -> None:
    for statement in DDL:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_kb_documents_version_source_type")
    op.execute("DROP INDEX IF EXISTS ix_kb_documents_version_source_split")
    op.execute("DROP INDEX IF EXISTS ix_kb_chunks_version_family")
    op.execute("DROP INDEX IF EXISTS ix_kb_chunks_embedding_hnsw_cosine")
