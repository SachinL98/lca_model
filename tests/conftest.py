from pathlib import Path

import pytest

from src.config import Settings
from src.database import init_db, make_engine, make_session_factory
from src.preprocess import seed_baseline_from_tsv

TSV_PATH = Path(__file__).resolve().parents[1] / "data" / "life_cycle.tsv"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        DATABASE_URL=f"sqlite:///{tmp_path / 'test.db'}",
        ARTIFACTS_DIR=str(tmp_path / "artifacts"),
        LIFE_CYCLE_TSV_PATH=str(TSV_PATH),
        ENABLE_SCHEDULER=False,
        RETRAIN_THRESHOLD_NEW_RECORDS=5,
    )


@pytest.fixture
def session_factory(settings):
    engine = make_engine(settings.DATABASE_URL)
    init_db(engine)
    yield make_session_factory(engine)
    engine.dispose()


@pytest.fixture
def session(session_factory):
    with session_factory() as s:
        yield s


@pytest.fixture
def seeded_session(session):
    seed_baseline_from_tsv(TSV_PATH, session)
    return session


def make_record(i: int = 0, **overrides) -> dict:
    rec = {
        "material": "Cotton",
        "country": "China",
        "supply_chain_step": "Dyeing",
        "co2_emission": 1.0 + i,
        "energy_consumption": 10.0 + i,
        "water_usage": 100.0 + i,
    }
    rec.update(overrides)
    return rec
