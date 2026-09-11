from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import build_engine
from app.readiness import ReadinessProbe, SystemReadinessProbe
from app.results import load_run_evidence


def create_app(
    readiness_probe: ReadinessProbe | None = None,
    *,
    engine: Engine | None = None,
) -> FastAPI:
    application = FastAPI(title="EvalForge API", version="0.2.0")
    application.state.readiness_probe = readiness_probe or SystemReadinessProbe(Settings.from_env())
    application.state.engine = engine or build_engine()

    @application.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "backend"}

    @application.get("/ready", tags=["system"])
    def ready(request: Request) -> JSONResponse:
        status = request.app.state.readiness_probe.check()
        payload = {
            "status": "ready" if status.ready else "not_ready",
            "checks": {"database": status.database, "redis": status.redis},
        }
        return JSONResponse(status_code=200 if status.ready else 503, content=payload)

    @application.get("/api/runs/{run_id}", tags=["experiments"])
    def run_evidence(run_id: str, request: Request) -> dict[str, object]:
        with Session(request.app.state.engine) as session:
            payload = load_run_evidence(session, run_id=run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="run not found")
        return payload

    return application


app = create_app()
