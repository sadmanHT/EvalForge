"""Initialize the Phase 02 migration environment.

Revision ID: 0001_phase02
Revises: None
"""
from __future__ import annotations

from typing import Sequence

revision: str = "0001_phase02"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """No domain tables are introduced before Phase 04."""


def downgrade() -> None:
    """No domain tables are removed by this foundation revision."""
