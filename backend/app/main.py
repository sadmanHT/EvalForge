from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.readiness import ReadinessProbe, SystemReadinessProbe


def create_app(readiness_probe: ReadinessProbe | None = None) -> FastAPI:
    application = FastAPI(title="EvalForge API", version="0.2.0")
    application.state.readiness_probe = readiness_probe or SystemReadinessProbe(Settings.from_env())

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

    return application


app = create_app()
