from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import psycopg
from redis import Redis

from app.config import Settings


@dataclass(frozen=True)
class ReadinessStatus:
    database: bool
    redis: bool

    @property
    def ready(self) -> bool:
        return self.database and self.redis


class ReadinessProbe(Protocol):
    def check(self) -> ReadinessStatus: ...


class SystemReadinessProbe:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def check(self) -> ReadinessStatus:
        database_ok = False
        redis_ok = False

        try:
            with (
                psycopg.connect(self._settings.database_url, connect_timeout=2) as connection,
                connection.cursor() as cursor,
            ):
                cursor.execute("SELECT 1")
                database_ok = cursor.fetchone() == (1,)
        except Exception:
            database_ok = False

        try:
            client = Redis.from_url(
                self._settings.redis_url,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            redis_ok = bool(client.ping())
        except Exception:
            redis_ok = False

        return ReadinessStatus(database=database_ok, redis=redis_ok)
