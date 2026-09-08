from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import psycopg
from redis import Redis

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://evalforge:evalforge@localhost:5432/evalforge",
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def wait_http(url: str, timeout: float = 120.0) -> int:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if 200 <= response.status < 300:
                    return response.status
                last_error = f"status {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = repr(exc)
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {url}: {last_error}")


def main() -> None:
    health_status = wait_http(f"{BACKEND_URL}/health")
    ready_status = wait_http(f"{BACKEND_URL}/ready")
    frontend_status = wait_http(FRONTEND_URL)

    redis_client = Redis.from_url(REDIS_URL)
    assert redis_client.ping()
    with psycopg.connect(DATABASE_URL, connect_timeout=3) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            assert cursor.fetchone() == (1,)

    subprocess.run([sys.executable, "scripts/worker_smoke.py"], check=True)

    print("PHASE02_COMPOSE_HEALTH=PASS")
    print(f"BACKEND_HEALTH_HTTP={health_status}")
    print(f"BACKEND_READY_HTTP={ready_status}")
    print(f"FRONTEND_HTTP={frontend_status}")
    print("POSTGRES=PASS")
    print("REDIS=PASS")


if __name__ == "__main__":
    main()
