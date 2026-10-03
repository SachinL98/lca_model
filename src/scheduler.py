"""Background jobs: external sync plus threshold- and cron-driven retraining."""

from __future__ import annotations

import logging
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import Settings
from src.data_sync import DataSyncClient, SyncError
from src.model_manager import ModelManager, RetrainInProgressError
from src.train import TrainingError

logger = logging.getLogger(__name__)


def retrain_if_needed(
    manager: ModelManager, *, min_new_records: int, force_if_no_model: bool = True
) -> dict[str, Any]:
    """Retrain when at least ``min_new_records`` arrived since the last training run.

    ``min_new_records=1`` therefore means "retrain if anything new arrived". A service without
    any model retrains regardless (when ``force_if_no_model``). Rejections by the validation
    guards are logged, never raised: the previously active model keeps serving.
    """
    new = manager.new_records_since_last_train()
    needed = min_new_records <= new or (force_if_no_model and not manager.is_loaded)
    if not needed:
        return {"retrained": False, "new_records": new, "reason": "below threshold"}
    try:
        result = manager.retrain()
    except RetrainInProgressError:
        return {"retrained": False, "new_records": new, "reason": "retrain already running"}
    except TrainingError as exc:
        logger.warning("Retrain rejected, keeping active model: %s", exc)
        return {"retrained": False, "new_records": new, "reason": f"rejected: {exc}"}
    except Exception:
        logger.exception("Retrain failed, keeping active model")
        return {"retrained": False, "new_records": new, "reason": "error"}
    return {"retrained": True, "new_records": new, "version_id": result["version_id"]}


def sync_external(sync_client: DataSyncClient) -> dict[str, Any] | None:
    try:
        return sync_client.sync_from_external_api()
    except SyncError as exc:
        logger.warning("External sync failed: %s", exc)
    except Exception:
        logger.exception("External sync crashed")
    return None


def threshold_job(manager: ModelManager, sync_client: DataSyncClient, settings: Settings) -> dict:
    """Frequent job: pull upstream records, retrain once enough new ones accumulated."""
    sync_external(sync_client)
    return retrain_if_needed(manager, min_new_records=settings.RETRAIN_THRESHOLD_NEW_RECORDS)


def cron_job(manager: ModelManager, sync_client: DataSyncClient) -> dict:
    """Scheduled job: pull upstream records and retrain on any new data, however little."""
    sync_external(sync_client)
    return retrain_if_needed(manager, min_new_records=1)


def create_scheduler(
    manager: ModelManager, sync_client: DataSyncClient, settings: Settings
) -> BackgroundScheduler:
    scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 3600}
    )
    if settings.RETRAIN_CRON_SCHEDULE.strip():
        scheduler.add_job(
            cron_job,
            CronTrigger.from_crontab(settings.RETRAIN_CRON_SCHEDULE),
            args=[manager, sync_client],
            id="scheduled_retrain",
            replace_existing=True,
        )
    if settings.SYNC_POLL_INTERVAL_MINUTES > 0:
        scheduler.add_job(
            threshold_job,
            "interval",
            minutes=settings.SYNC_POLL_INTERVAL_MINUTES,
            args=[manager, sync_client, settings],
            id="threshold_retrain",
            replace_existing=True,
        )
    return scheduler
