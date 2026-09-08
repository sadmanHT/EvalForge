from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=os.environ.get(
                "DATABASE_URL",
                "postgresql://evalforge:evalforge@localhost:5432/evalforge",
            ),
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        )
