import json

import httpx
import pandas as pd
import pytest
from sqlalchemy import func, select

from src.data_sync import DataSyncClient, SyncError
from src.database import TARGETS, LCARecord
from src.preprocess import (
    fill_missing_targets,
    load_baseline_frame,
    load_training_frame,
    seed_baseline_from_tsv,
    seed_if_empty,
)
from tests.conftest import TSV_PATH, make_record


def count(session):
    return session.scalar(select(func.count()).select_from(LCARecord))


def test_pivot_is_one_row_per_combination_and_complete():
    frame = load_baseline_frame(TSV_PATH)
    assert not frame.duplicated(["material", "country", "supply_chain_step"]).any()
    assert not frame[TARGETS].isna().any().any()
    assert {"Cotton", "Hemp", "Silk", "Polyester", "Wool"} <= set(frame["material"])


def test_pivot_matches_raw_values():
    raw = pd.read_csv(TSV_PATH, sep="\t")
    row = raw[
        (raw.material == "Cotton")
        & (raw.country == "Argentina")
        & (raw.supply_chain_step == "Dyeing")
    ].set_index("compound")["input_or_output_per_kg"]
    frame = load_baseline_frame(TSV_PATH)
    got = frame[
        (frame.material == "Cotton")
        & (frame.country == "Argentina")
        & (frame.supply_chain_step == "Dyeing")
    ].iloc[0]
    assert got.co2_emission == pytest.approx(row["carbon dioxide"])
    assert got.energy_consumption == pytest.approx(row["energy"])
    assert got.water_usage == pytest.approx(row["water"])


def test_missing_targets_filled_with_group_median():
    df = pd.DataFrame(
        {
            "material": ["A", "A", "A"],
            "country": ["X", "Y", "Z"],
            "supply_chain_step": ["S", "S", "S"],
            "co2_emission": [1.0, 3.0, None],
            "energy_consumption": [1.0, 1.0, 1.0],
            "water_usage": [1.0, 1.0, 1.0],
        }
    )
    assert fill_missing_targets(df)["co2_emission"].tolist() == [1.0, 3.0, 2.0]


def test_seed_inserts_baseline_and_is_idempotent_via_seed_if_empty(session):
    n = seed_baseline_from_tsv(TSV_PATH, session)
    assert n > 1000 and count(session) == n
    assert seed_if_empty(session, TSV_PATH) == 0
    assert count(session) == n
    assert session.scalars(select(LCARecord.source)).first() == "baseline"
    assert len(load_training_frame(session)) == n


def test_insert_custom_records_validates(session_factory, settings, session):
    client = DataSyncClient(session_factory, settings)
    res = client.insert_custom_records(
        [make_record(), make_record(material=""), make_record(co2_emission=-1), {"x": 1}]
    )
    assert (res["inserted"], res["invalid"]) == (1, 3)
    assert count(session) == 1


def _transport(payloads):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=json.dumps(payloads.pop(0)))

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_sync_from_external_api_dedupes_and_paginates(session_factory, settings, session):
    settings = settings.model_copy(
        update={"EXTERNAL_DATA_API_URL": "http://up/records", "EXTERNAL_DATA_API_KEY": "k"}
    )
    page1 = {
        "records": [
            {**make_record(1), "id": 7},
            {**make_record(2), "id": 8, "verified": False},
        ],
        "next": "http://up/records?page=2",
    }
    page2 = [{**make_record(3), "id": 9}, {"bad": True}]
    http, calls = _transport([page1, page2])
    client = DataSyncClient(session_factory, settings, http_client=http)
    res = client.sync_from_external_api()
    assert (res["fetched"], res["inserted"], res["unverified"], res["invalid"]) == (4, 2, 1, 1)
    assert calls[0].headers["authorization"] == "Bearer k"
    assert count(session) == 2

    http, _ = _transport([[{**make_record(1), "id": 7}, {**make_record(5), "id": 10}]])
    res = DataSyncClient(session_factory, settings, http_client=http).sync_from_external_api()
    assert (res["inserted"], res["duplicates"]) == (1, 1)
    assert count(session) == 3


def test_sync_skipped_without_url_and_errors_on_failure(session_factory, settings):
    assert DataSyncClient(session_factory, settings).sync_from_external_api()["skipped"]
    settings = settings.model_copy(update={"EXTERNAL_DATA_API_URL": "http://up"})
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(SyncError):
        DataSyncClient(session_factory, settings, http_client=http).sync_from_external_api()
