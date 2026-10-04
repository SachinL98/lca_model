"""FastAPI application: inference, record ingestion, sync, retraining and status endpoints."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from src.config import Settings, get_settings
from src.data_sync import DataSyncClient, RecordIn
from src.database import LCARecord, init_db, make_engine, make_session_factory
from src.model_manager import ModelManager, ModelUnavailableError, RetrainInProgressError
from src.predict import predict
from src.preprocess import seed_if_empty
from src.scheduler import create_scheduler, retrain_if_needed, sync_external
from src.train import InsufficientDataError, TrainingError

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    material: str = Field(min_length=1, max_length=100, examples=["Cotton"])
    country: str = Field(min_length=1, max_length=100, examples=["India"])
    supply_chain_step: str = Field(min_length=1, max_length=100, examples=["Dyeing"])
    amount_kg: float = Field(default=1.0, ge=0, allow_inf_nan=False)


class RecordsPayload(BaseModel):
    records: list[RecordIn] = Field(min_length=1)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.DATABASE_URL)
        init_db(engine)
        factory = make_session_factory(engine)
        with factory() as session:
            seed_if_empty(session, settings.LIFE_CYCLE_TSV_PATH)
        manager = ModelManager(factory, settings)
        sync_client = DataSyncClient(factory, settings)
        app.state.session_factory = factory
        app.state.manager = manager
        app.state.sync_client = sync_client

        if not manager.load_active_model():
            logger.info("No stored model found; training an initial one")
            retrain_if_needed(manager, min_new_records=1)

        scheduler = None
        if settings.ENABLE_SCHEDULER:
            scheduler = create_scheduler(manager, sync_client, settings)
            scheduler.start()
        app.state.scheduler = scheduler
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.shutdown(wait=False)
            engine.dispose()

    app = FastAPI(title="LCA Prediction Service", version="1.0.0", lifespan=lifespan)

    origins = [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
        )

    def require_admin(x_api_key: str | None = Header(default=None)) -> None:
        if settings.ADMIN_API_KEY and not secrets.compare_digest(
            x_api_key or "", settings.ADMIN_API_KEY
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    admin = [Depends(require_admin)]

    def manager_of(request: Request) -> ModelManager:
        return request.app.state.manager

    def background_retrain(manager: ModelManager) -> None:
        retrain_if_needed(manager, min_new_records=settings.RETRAIN_THRESHOLD_NEW_RECORDS)

    def background_sync(manager: ModelManager, client: DataSyncClient) -> None:
        sync_external(client)
        background_retrain(manager)

    # ---------------------------------------------------------------- inference
    @app.post("/predict")
    def predict_endpoint(body: PredictRequest, request: Request) -> dict[str, Any]:
        try:
            return predict(
                manager_of(request),
                body.material,
                body.country,
                body.supply_chain_step,
                body.amount_kg,
            )
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    # ---------------------------------------------------------------- ingestion
    @app.post("/records", dependencies=admin, status_code=201)
    def add_records(
        payload: RecordIn | list[RecordIn] | RecordsPayload,
        request: Request,
        background: BackgroundTasks,
    ) -> dict[str, Any]:
        """Ingest one record, a list of records, or ``{"records": [...]}``."""
        if isinstance(payload, RecordsPayload):
            items = payload.records
        elif isinstance(payload, list):
            items = payload
        else:
            items = [payload]
        if not items:
            raise HTTPException(status_code=422, detail="No records supplied")

        manager = manager_of(request)
        result = request.app.state.sync_client.insert_custom_records(
            [r.model_dump() for r in items], source="api"
        )
        new = manager.new_records_since_last_train()
        queued = new >= settings.RETRAIN_THRESHOLD_NEW_RECORDS and not manager.retraining
        if queued:
            background.add_task(background_retrain, manager)
        return {
            "inserted": result["inserted"],
            "new_records_since_last_train": new,
            "retrain_threshold": settings.RETRAIN_THRESHOLD_NEW_RECORDS,
            "retrain_queued": queued,
        }

    @app.post("/sync/external", dependencies=admin, status_code=202)
    def sync_external_endpoint(request: Request, background: BackgroundTasks) -> dict[str, str]:
        if not settings.EXTERNAL_DATA_API_URL:
            raise HTTPException(status_code=400, detail="EXTERNAL_DATA_API_URL is not configured")
        background.add_task(background_sync, manager_of(request), request.app.state.sync_client)
        return {"status": "sync queued"}

    # ---------------------------------------------------------------- model lifecycle
    @app.post("/retrain", dependencies=admin)
    def retrain_endpoint(request: Request) -> dict[str, Any]:
        """Force a retrain now; the new model is hot-reloaded without restarting the server."""
        try:
            result = manager_of(request).retrain()
        except RetrainInProgressError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InsufficientDataError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except TrainingError as exc:  # validation gate: previous model stays active
            raise HTTPException(status_code=422, detail=f"Candidate rejected: {exc}") from exc
        return {
            "status": "retrained",
            "version_id": result["version_id"],
            "previous_version": result["previous_version"],
            "record_count": result["record_count"],
            "metrics": result["metrics"],
        }

    @app.post("/model/rollback", dependencies=admin)
    def rollback_endpoint(request: Request) -> dict[str, str]:
        try:
            return {"status": "rolled back", "active_version": manager_of(request).rollback()}
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/model/status")
    def model_status(request: Request) -> dict[str, Any]:
        manager = manager_of(request)
        with request.app.state.session_factory() as session:
            run = manager.registry.active_run(session)
            total = session.scalar(select(func.count()).select_from(LCARecord)) or 0
        return {
            "model_loaded": manager.is_loaded,
            "active_version": manager.version,
            "created_at": run.created_at.isoformat() if run else None,
            "training_record_count": run.record_count if run else 0,
            "metrics": run.metrics if run else None,
            "total_records": total,
            "new_records_since_last_train": manager.new_records_since_last_train(),
            "retrain_threshold": settings.RETRAIN_THRESHOLD_NEW_RECORDS,
            "retraining": manager.retraining,
        }

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        manager = manager_of(request)
        return {
            "status": "ok" if manager.is_loaded else "degraded",
            "model_available": manager.is_loaded,
            "model_version": manager.version,
        }

    return app


app = create_app()
