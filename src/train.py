"""Multi-target regression training with validation guards."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy.orm import Session

from src.config import Settings, get_settings
from src.database import FEATURES, TARGETS
from src.model_manager import next_version_id
from src.preprocess import load_training_frame

logger = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Base class for training failures that must not result in a deployment."""


class InsufficientDataError(TrainingError):
    pass


class ValidationGateError(TrainingError):
    pass


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            (
                "regressor",
                MultiOutputRegressor(
                    LGBMRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=6,
                        random_state=42,
                        verbose=-1,
                    )
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, X, y) -> dict[str, Any]:
    """R^2, MAE and RMSE per target on ``(X, y)``, plus averages (``avg_r2`` etc.)."""
    pred = model.predict(X)
    per_target: dict[str, dict[str, float]] = {}
    for i, name in enumerate(TARGETS):
        truth = y[name].to_numpy()
        per_target[name] = {
            "r2": float(r2_score(truth, pred[:, i])),
            "mae": float(mean_absolute_error(truth, pred[:, i])),
            "rmse": float(np.sqrt(mean_squared_error(truth, pred[:, i]))),
        }
    avg = {
        f"avg_{m}": float(np.mean([t[m] for t in per_target.values()]))
        for m in ("r2", "mae", "rmse")
    }
    return {"per_target": per_target, **avg}


def train_model(
    db_session: Session,
    settings: Settings | None = None,
    baseline_model: Pipeline | None = None,
) -> dict[str, Any]:
    """Train on every clean record, validate, and save a versioned model file.

    Raises ``InsufficientDataError`` / ``ValidationGateError`` (nothing is saved) when the guards
    fail. ``baseline_model`` (the currently active model) is scored on the same held-out split
    so a candidate that is clearly worse than what is deployed is rejected too.

    Returns ``version_id``, ``path``, ``record_count``, ``metrics`` and ``trained_at``. The model
    is *not* activated; see ``ModelManager.promote`` / ``ModelManager.retrain``.
    """
    settings = settings or get_settings()
    raw = load_training_frame(db_session)
    record_count = len(raw)
    if record_count < settings.MIN_TRAIN_SAMPLES:
        raise InsufficientDataError(
            f"Need at least {settings.MIN_TRAIN_SAMPLES} clean records, found {record_count}"
        )

    # Repeated material/country/step observations are averaged so identical feature rows can
    # never straddle the train/test split (which would leak and inflate the test scores).
    data = raw.groupby(FEATURES, as_index=False)[TARGETS].mean()
    if len(data) < 10:
        raise InsufficientDataError(f"Only {len(data)} distinct feature combinations available")

    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGETS], test_size=0.2, random_state=42
    )
    model = build_pipeline().fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    metrics.update(n_train=len(X_train), n_test=len(X_test))

    if not np.isfinite(metrics["avg_r2"]) or metrics["avg_r2"] < settings.MIN_AVG_R2:
        raise ValidationGateError(
            f"Average test R2 {metrics['avg_r2']:.4f} is below the minimum {settings.MIN_AVG_R2}"
        )
    if baseline_model is not None:
        try:
            baseline_r2 = evaluate(baseline_model, X_test, y_test)["avg_r2"]
        except Exception:  # an incompatible/corrupt baseline must not block a good candidate
            logger.warning("Could not score the active model on the new test split", exc_info=True)
        else:
            metrics["baseline_avg_r2"] = baseline_r2
            if metrics["avg_r2"] < baseline_r2 - settings.MAX_R2_REGRESSION:
                raise ValidationGateError(
                    f"Candidate avg R2 {metrics['avg_r2']:.4f} regresses more than "
                    f"{settings.MAX_R2_REGRESSION} below the active model's {baseline_r2:.4f}"
                )

    version_id = next_version_id(db_session, settings.models_path)
    path = settings.models_path / f"model_{version_id}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".joblib.tmp")
    joblib.dump(model, tmp)
    tmp.replace(path)
    logger.info(
        "Trained %s on %d records: avg R2=%.4f", version_id, record_count, metrics["avg_r2"]
    )
    return {
        "version_id": version_id,
        "path": str(path),
        "record_count": record_count,
        "metrics": metrics,
        "trained_at": datetime.now(UTC).isoformat(),
    }
