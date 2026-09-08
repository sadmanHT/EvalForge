from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.readiness import ReadinessStatus


class HealthyProbe:
    def check(self) -> ReadinessStatus:
        return ReadinessStatus(database=True, redis=True)


class FailedDatabaseProbe:
    def check(self) -> ReadinessStatus:
        return ReadinessStatus(database=False, redis=True)


def test_health_is_liveness_only() -> None:
    client = TestClient(create_app(readiness_probe=FailedDatabaseProbe()))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "backend"}


def test_ready_reports_database_and_redis_success() -> None:
    client = TestClient(create_app(readiness_probe=HealthyProbe()))
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"database": True, "redis": True},
    }


def test_ready_fails_when_a_required_dependency_is_unavailable() -> None:
    client = TestClient(create_app(readiness_probe=FailedDatabaseProbe()))
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"] == {"database": False, "redis": True}
