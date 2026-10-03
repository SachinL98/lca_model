import joblib
import pandas as pd
import pytest

from src.database import FEATURES, TARGETS, ModelRun
from src.model_manager import ModelRegistry, next_version_id
from src.preprocess import load_training_frame
from src.train import InsufficientDataError, ValidationGateError, train_model


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One shared training run (it is the slowest thing in the suite)."""
    from src.config import Settings
    from src.database import init_db, make_engine, make_session_factory
    from src.preprocess import seed_baseline_from_tsv
    from tests.conftest import TSV_PATH

    tmp = tmp_path_factory.mktemp("model")
    settings = Settings(
        _env_file=None,
        DATABASE_URL=f"sqlite:///{tmp / 'db.sqlite'}",
        ARTIFACTS_DIR=str(tmp / "artifacts"),
    )
    engine = make_engine(settings.DATABASE_URL)
    init_db(engine)
    with make_session_factory(engine)() as s:
        seed_baseline_from_tsv(TSV_PATH, s)
        result = train_model(s, settings)
    return settings, result


def test_train_saves_versioned_model_with_metrics(trained):
    settings, result = trained
    assert result["version_id"] == "v1"
    assert result["path"].endswith("models/model_v1.joblib")
    m = result["metrics"]
    assert set(m["per_target"]) == set(TARGETS)
    for metric in ("r2", "mae", "rmse"):
        assert all(metric in t for t in m["per_target"].values())
    assert m["avg_r2"] >= 0.0
    assert m["n_train"] > m["n_test"] > 0


def test_inference_and_unseen_categories(trained):
    _, result = trained
    model = joblib.load(result["path"])
    X = pd.DataFrame(
        [
            ["Cotton", "China", "Dyeing"],
            ["Unobtainium", "Atlantis", "Teleporting"],  # entirely unseen
        ],
        columns=FEATURES,
    )
    pred = model.predict(X)
    assert pred.shape == (2, 3)
    assert pd.notna(pred).all()


def test_insufficient_data_guard(session, settings):
    with pytest.raises(InsufficientDataError):
        train_model(session, settings)
    assert not settings.models_path.exists() or not list(settings.models_path.glob("*.joblib"))


def test_validation_gate_blocks_degraded_model(seeded_session, settings):
    strict = settings.model_copy(update={"MIN_AVG_R2": 1.1})
    with pytest.raises(ValidationGateError):
        train_model(seeded_session, strict)
    assert not list(settings.models_path.glob("*.joblib"))


def test_regression_vs_baseline_is_rejected(seeded_session, settings):
    class OracleBaseline:
        """A baseline that reproduces the ground truth exactly (R2 == 1)."""

        def predict(self, X):
            truth = load_training_frame(seeded_session).groupby(FEATURES, as_index=False)[TARGETS]
            merged = X.reset_index(drop=True).merge(truth.mean(), on=FEATURES, how="left")
            return merged[TARGETS].to_numpy()

    with pytest.raises(ValidationGateError, match="regresses"):
        train_model(seeded_session, settings, baseline_model=OracleBaseline())
    assert not list(settings.models_path.glob("*.joblib"))


def test_registry_versioning_and_pointer(session, settings):
    reg = ModelRegistry(settings.artifacts_path)
    assert next_version_id(session, reg.models_dir) == "v1"
    for v in ("v1", "v2"):
        reg.models_dir.mkdir(parents=True, exist_ok=True)
        reg.model_path(v).write_bytes(v.encode())
        reg.record_run(session, v, 100, {"avg_r2": 0.5}, reg.model_path(v))
        reg.activate(session, v, {"material": ["Cotton"]})
    assert next_version_id(session, reg.models_dir) == "v3"
    assert reg.current_path.read_bytes() == b"v2"
    assert [r.version_id for r in session.query(ModelRun).filter_by(is_active=True)] == ["v2"]
    meta = reg.read_metadata()
    assert meta["active_version"] == "v2" and meta["history"] == ["v1", "v2"]
    assert reg.previous_version("v2") == "v1"
    assert reg.previous_version("v1") is None
