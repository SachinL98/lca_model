import pytest
from fastapi.testclient import TestClient

from src.main import create_app
from tests.conftest import make_record


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def test_startup_seeds_trains_and_reports_healthy(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "model_available": True,
        "model_version": "v1",
    }
    status = client.get("/model/status").json()
    assert status["active_version"] == "v1"
    assert status["training_record_count"] == status["total_records"] > 1000
    assert status["metrics"]["avg_r2"] >= 0
    assert status["new_records_since_last_train"] == 0


def test_predict_scales_by_amount_and_clips(client):
    body = {"material": "Cotton", "country": "India", "supply_chain_step": "Dyeing"}
    one = client.post("/predict", json=body).json()
    ten = client.post("/predict", json={**body, "amount_kg": 10}).json()
    assert set(one["predictions"]) == {"co2_emission", "energy_consumption", "water_usage"}
    assert all(v >= 0 for v in one["predictions"].values())
    for k, v in one["predictions"].items():
        assert ten["predictions"][k] == pytest.approx(10 * v)
    assert one["unseen_categories"] == [] and one["model_version"] == "v1"


def test_predict_handles_case_and_unseen_categories(client):
    r = client.post(
        "/predict",
        json={"material": " cotton ", "country": "Atlantis", "supply_chain_step": "Teleporting"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["input"]["material"] == "Cotton"
    assert body["unseen_categories"] == ["country", "supply_chain_step"]
    assert all(v >= 0 for v in body["predictions"].values())


def test_predict_validation(client):
    assert client.post("/predict", json={"material": "Cotton"}).status_code == 422
    bad = {"material": "", "country": "India", "supply_chain_step": "Dyeing"}
    assert client.post("/predict", json=bad).status_code == 422
    ok = {**bad, "material": "Cotton", "amount_kg": -1}
    assert client.post("/predict", json=ok).status_code == 422


def test_records_single_list_and_wrapped(client):
    r = client.post("/records", json=make_record())
    assert r.status_code == 201 and r.json()["inserted"] == 1
    r = client.post("/records", json=[make_record(1), make_record(2)])
    assert r.json()["inserted"] == 2
    r = client.post("/records", json={"records": [make_record(3)]})
    assert r.json()["inserted"] == 1
    assert client.get("/model/status").json()["new_records_since_last_train"] == 4


def test_records_rejects_invalid(client):
    assert client.post("/records", json={**make_record(), "co2_emission": -3}).status_code == 422
    assert client.post("/records", json=[make_record(), {"material": "x"}]).status_code == 422
    assert client.get("/model/status").json()["new_records_since_last_train"] == 0


def test_records_over_threshold_queues_background_retrain(client):
    # threshold is 5 in the test settings; TestClient runs background tasks before returning
    r = client.post("/records", json=[make_record(i, material="Hemp") for i in range(5)])
    assert r.json()["retrain_queued"] is True
    status = client.get("/model/status").json()
    assert status["active_version"] == "v2"
    assert status["new_records_since_last_train"] == 0


def test_below_threshold_does_not_retrain(client):
    r = client.post("/records", json=[make_record(i) for i in range(2)])
    assert r.json()["retrain_queued"] is False
    assert client.get("/health").json()["model_version"] == "v1"


def test_retrain_endpoint_hot_reloads(client):
    client.post("/records", json=[make_record(i, country="Peru") for i in range(3)])
    r = client.post("/retrain")
    assert r.status_code == 200
    body = r.json()
    assert (body["version_id"], body["previous_version"]) == ("v2", "v1")
    assert body["metrics"]["avg_r2"] >= 0
    # the freshly trained model serves immediately, including the newly seen country
    p = client.post(
        "/predict",
        json={"material": "Cotton", "country": "Peru", "supply_chain_step": "Dyeing"},
    ).json()
    assert p["model_version"] == "v2" and p["unseen_categories"] == []


def test_rollback_endpoint(client):
    assert client.post("/model/rollback").status_code == 409
    client.post("/retrain")
    assert client.post("/model/rollback").json()["active_version"] == "v1"
    assert client.get("/health").json()["model_version"] == "v1"


def test_sync_external_requires_configuration(client):
    assert client.post("/sync/external").status_code == 400


def test_sync_external_queued(settings, monkeypatch):
    import httpx

    payload = [{**make_record(i, material="Silk"), "id": i} for i in range(5)]
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: real_client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)), **kw
        ),
    )
    cfg = settings.model_copy(update={"EXTERNAL_DATA_API_URL": "http://upstream/records"})
    with TestClient(create_app(cfg)) as c:
        assert c.post("/sync/external").status_code == 202
        status = c.get("/model/status").json()
        assert status["active_version"] == "v2"  # 5 new upstream records hit the threshold


def test_admin_key_protects_mutating_endpoints(settings):
    cfg = settings.model_copy(update={"ADMIN_API_KEY": "s3cret"})
    with TestClient(create_app(cfg)) as c:
        assert c.post("/retrain").status_code == 401
        assert c.post("/records", json=make_record()).status_code == 401
        assert c.post("/retrain", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200
        assert (
            c.post(
                "/predict",
                json={"material": "Cotton", "country": "India", "supply_chain_step": "Dyeing"},
            ).status_code
            == 200
        )


def test_restart_reuses_persisted_model(settings):
    with TestClient(create_app(settings)) as c:
        c.post("/retrain")
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").json()["model_version"] == "v2"  # loaded, not retrained
