import httpx
import pytest
from sqlalchemy import select

from src.data_sync import DataSyncClient
from src.database import LCARecord, ModelRun
from src.model_manager import ModelManager, ModelUnavailableError, RetrainInProgressError
from src.preprocess import seed_baseline_from_tsv
from src.scheduler import create_scheduler, cron_job, retrain_if_needed, threshold_job
from src.train import InsufficientDataError, ValidationGateError
from tests.conftest import TSV_PATH, make_record


@pytest.fixture
def manager(session_factory, settings):
    with session_factory() as s:
        seed_baseline_from_tsv(TSV_PATH, s)
    return ModelManager(session_factory, settings)


def new_records(session_factory, settings, n, **kw):
    DataSyncClient(session_factory, settings).insert_custom_records(
        [make_record(i, **kw) for i in range(n)]
    )


def test_manager_without_model_raises(manager):
    assert not manager.load_active_model()
    with pytest.raises(ModelUnavailableError):
        manager.get()


def test_first_retrain_activates_and_persists(manager, session_factory, settings):
    result = manager.retrain()
    assert result["version_id"] == "v1" and manager.version == "v1"
    assert manager.registry.current_path.exists()
    meta = manager.registry.read_metadata()
    assert meta["active_version"] == "v1"
    assert "Cotton" in meta["vocabulary"]["material"]
    with session_factory() as s:
        runs = s.scalars(select(ModelRun)).all()
        assert [(r.version_id, r.is_active) for r in runs] == [("v1", True)]
        assert runs[0].metrics["avg_r2"] >= 0

    # a fresh process picks the persisted model up again
    other = ModelManager(session_factory, settings)
    assert other.load_active_model() and other.version == "v1"


def test_new_records_then_retrain_hot_reloads(manager, session_factory, settings):
    manager.retrain()
    old_model, _ = manager.get()
    assert manager.new_records_since_last_train() == 0

    new_records(session_factory, settings, 7, material="Hemp", country="China")
    with session_factory() as s:
        assert s.query(LCARecord).filter_by(source="api").count() == 7
    assert manager.new_records_since_last_train() == 7

    result = manager.retrain()
    new_model, version = manager.get()
    assert (result["version_id"], version) == ("v2", "v2")
    assert new_model is not old_model
    assert result["record_count"] > 2030
    assert manager.new_records_since_last_train() == 0
    with session_factory() as s:
        assert {r.version_id: r.is_active for r in s.scalars(select(ModelRun))} == {
            "v1": False,
            "v2": True,
        }


def test_failed_gate_keeps_active_model(manager, settings):
    manager.retrain()
    manager.settings = settings.model_copy(update={"MIN_AVG_R2": 1.1})
    with pytest.raises(ValidationGateError):
        manager.retrain()
    assert manager.version == "v1"
    assert manager.registry.read_metadata()["active_version"] == "v1"


def test_insufficient_data_rejected(session_factory, settings):
    mgr = ModelManager(session_factory, settings)
    new_records(session_factory, settings, 3)
    with pytest.raises(InsufficientDataError):
        mgr.retrain()
    assert not mgr.is_loaded


def test_rollback_restores_previous_version(manager, session_factory, settings):
    manager.retrain()
    with pytest.raises(ModelUnavailableError):
        manager.rollback()
    new_records(session_factory, settings, 6)
    manager.retrain()
    assert manager.version == "v2"

    assert manager.rollback() == "v1"
    assert manager.version == "v1"
    assert manager.registry.read_metadata()["active_version"] == "v1"
    with session_factory() as s:
        assert [r.version_id for r in s.scalars(select(ModelRun).where(ModelRun.is_active))] == [
            "v1"
        ]


def test_hot_reload_with_corrupt_file_keeps_old_model(manager, tmp_path):
    manager.retrain()
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    with pytest.raises(Exception):  # noqa: B017
        manager.hot_reload(bad)
    assert manager.version == "v1" and manager.is_loaded


def test_concurrent_retrain_is_refused(manager):
    assert manager._retrain_lock.acquire(blocking=False)
    try:
        with pytest.raises(RetrainInProgressError):
            manager.retrain()
    finally:
        manager._retrain_lock.release()


def test_retrain_if_needed_threshold(manager, session_factory, settings):
    assert retrain_if_needed(manager, min_new_records=5)["retrained"]  # no model yet
    assert not retrain_if_needed(manager, min_new_records=5)["retrained"]
    new_records(session_factory, settings, 4)
    assert not retrain_if_needed(manager, min_new_records=5)["retrained"]
    new_records(session_factory, settings, 1)
    out = retrain_if_needed(manager, min_new_records=5)
    assert out["retrained"] and out["version_id"] == "v2"


def test_scheduler_jobs_sync_then_retrain(manager, session_factory, settings):
    manager.retrain()
    settings = settings.model_copy(update={"EXTERNAL_DATA_API_URL": "http://up/records"})
    payload = [{**make_record(i, material="Silk"), "id": i} for i in range(5)]
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))
    client = DataSyncClient(session_factory, settings, http_client=http)

    out = threshold_job(manager, client, settings)
    assert out["retrained"] and manager.version == "v2"
    # same upstream payload again: all duplicates, nothing new, no retrain
    assert not threshold_job(manager, client, settings)["retrained"]
    assert not cron_job(manager, client)["retrained"]


def test_create_scheduler_registers_jobs(manager, session_factory, settings):
    client = DataSyncClient(session_factory, settings)
    scheduler = create_scheduler(manager, client, settings)
    assert {j.id for j in scheduler.get_jobs()} == {"scheduled_retrain", "threshold_retrain"}
    off = settings.model_copy(update={"RETRAIN_CRON_SCHEDULE": "", "SYNC_POLL_INTERVAL_MINUTES": 0})
    assert create_scheduler(manager, client, off).get_jobs() == []
