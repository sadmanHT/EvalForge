FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY backend/requirements.full.lock ./requirements.full.lock
COPY backend/pyproject.toml ./pyproject.toml
COPY backend/app ./app
COPY backend/alembic.ini ./alembic.ini
COPY backend/migrations ./migrations
COPY configs/model.yaml ./configs/model.yaml
COPY datasets/incident_diagnosis/processed/evalforge-incident-diagnosis-v0.1.0 ./datasets/incident_diagnosis/processed/evalforge-incident-diagnosis-v0.1.0

RUN pip install --no-cache-dir -r requirements.full.lock \
    && pip install --no-cache-dir --no-deps .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
