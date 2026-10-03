"""Model versioning: run history in the DB, ``metadata.json`` and the ``current_model`` pointer."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings
from src.database import FEATURES, TARGETS, LCARecord, ModelRun

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"^v(\d+)$")


def next_version_id(session: Session, models_dir: Path) -> str:
    """``v<N>`` where N is one more than any version known to the DB or present on disk."""
    numbers = [0]
    for (vid,) in session.execute(select(ModelRun.version_id)):
        if m := _VERSION_RE.match(vid):
            numbers.append(int(m.group(1)))
    if models_dir.exists():
        for f in models_dir.glob("model_v*.joblib"):
            if m := _VERSION_RE.match(f.stem.removeprefix("model_")):
                numbers.append(int(m.group(1)))
    return f"v{max(numbers) + 1}"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class ModelRegistry:
    """Persistent bookkeeping of model versions (no in-memory state)."""

    def __init__(self, artifacts_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.current_path = self.artifacts_dir / "current_model.joblib"
        self.metadata_path = self.artifacts_dir / "metadata.json"

    # -- metadata.json
    def read_metadata(self) -> dict[str, Any]:
        try:
            return json.loads(self.metadata_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {"active_version": None, "history": [], "vocabulary": {}}

    def write_metadata(self, meta: dict[str, Any]) -> None:
        _atomic_write_text(self.metadata_path, json.dumps(meta, indent=2, sort_keys=True))

    def model_path(self, version_id: str) -> Path:
        return self.models_dir / f"model_{version_id}.joblib"

    # -- current_model.joblib pointer (symlink, or an atomic copy where symlinks are unavailable)
    def point_current_to(self, version_id: str) -> None:
        target = self.model_path(version_id)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.current_path.with_name(self.current_path.name + ".tmp")
        tmp.unlink(missing_ok=True)
        try:
            os.symlink(os.path.relpath(target, self.artifacts_dir), tmp)
        except (OSError, NotImplementedError):
            shutil.copyfile(target, tmp)
        os.replace(tmp, self.current_path)

    # -- DB
    def record_run(
        self, session: Session, version_id: str, record_count: int, metrics: dict, path: str | Path
    ) -> ModelRun:
        run = ModelRun(
            version_id=version_id,
            record_count=record_count,
            metrics=metrics,
            is_active=False,
            path=str(path),
        )
        session.merge(run)
        session.commit()
        return run

    def activate(
        self, session: Session, version_id: str, vocabulary: dict[str, list[str]] | None = None
    ) -> None:
        """Mark ``version_id`` as the single active run and persist the pointer + metadata."""
        session.execute(update(ModelRun).values(is_active=False))
        session.execute(
            update(ModelRun).where(ModelRun.version_id == version_id).values(is_active=True)
        )
        session.commit()
        self.point_current_to(version_id)
        meta = self.read_metadata()
        history = [v for v in meta.get("history", []) if v != version_id]
        history.append(version_id)
        meta.update(active_version=version_id, history=history)
        if vocabulary is not None:
            meta["vocabulary"] = vocabulary
        self.write_metadata(meta)

    def previous_version(self, active: str | None) -> str | None:
        """Most recent earlier version (still on disk) that was previously active."""
        history = self.read_metadata().get("history", [])
        if active in history:
            history = history[: history.index(active)]
        for version_id in reversed(history):
            if self.model_path(version_id).exists():
                return version_id
        return None

    def forget_active(self, version_id: str) -> None:
        """Drop ``version_id`` from the stable history (used when a version is rolled back)."""
        meta = self.read_metadata()
        meta["history"] = [v for v in meta.get("history", []) if v != version_id]
        self.write_metadata(meta)

    def active_run(self, session: Session) -> ModelRun | None:
        return session.scalars(
            select(ModelRun)
            .where(ModelRun.is_active.is_(True))
            .order_by(ModelRun.created_at.desc())
        ).first()

    def latest_run(self, session: Session) -> ModelRun | None:
        return session.scalars(select(ModelRun).order_by(ModelRun.created_at.desc())).first()


# --------------------------------------------------------------------------------------
# In-memory active model with atomic hot-reload / rollback
# --------------------------------------------------------------------------------------
class ModelUnavailableError(RuntimeError):
    pass


class RetrainInProgressError(RuntimeError):
    pass


def vocabulary_of(model: Any) -> dict[str, list[str]]:
    """Known category values per feature, read from the fitted one-hot encoder."""
    encoder = model.named_steps["encoder"]
    return {
        name: [str(c) for c in cats]
        for name, cats in zip(FEATURES, encoder.categories_, strict=True)
    }


def smoke_test(model: Any) -> None:
    """Raise if ``model`` cannot produce finite predictions for all targets."""
    cats = vocabulary_of(model)
    row = {f: [cats[f][0] if cats[f] else "x"] for f in FEATURES}
    pred = model.predict(pd.DataFrame(row))
    if pred.shape != (1, len(TARGETS)) or not pd.notna(pred).all():
        raise ValueError(f"Model failed smoke test: got {pred!r}")


class ModelManager:
    """Thread-safe holder of the active model.

    Readers call :meth:`get` (a lock-protected reference read); a swap replaces the reference in
    one step after the new model was loaded and smoke-tested *outside* the lock, so inference is
    never blocked and never sees a half-loaded model.
    """

    def __init__(self, session_factory: sessionmaker, settings: Settings):
        self.session_factory = session_factory
        self.settings = settings
        self.registry = ModelRegistry(settings.artifacts_path)
        self._lock = threading.RLock()
        self._retrain_lock = threading.Lock()
        self._model: Any = None
        self._version: str | None = None

    # -- reads
    @property
    def version(self) -> str | None:
        with self._lock:
            return self._version

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    @property
    def retraining(self) -> bool:
        return self._retrain_lock.locked()

    def get(self) -> tuple[Any, str]:
        """The active ``(model, version_id)``; raises ``ModelUnavailableError`` if none."""
        with self._lock:
            if self._model is None:
                raise ModelUnavailableError("No model is loaded yet")
            return self._model, self._version  # type: ignore[return-value]

    def _swap(self, model: Any, version: str | None) -> tuple[Any, str | None]:
        with self._lock:
            previous = (self._model, self._version)
            self._model, self._version = model, version
        return previous

    # -- loading
    def load_active_model(self) -> bool:
        """Load ``current_model.joblib`` (falling back to metadata). False if nothing usable."""
        candidates = [self.registry.current_path]
        active = self.registry.read_metadata().get("active_version")
        if active:
            candidates.append(self.registry.model_path(active))
        for path in candidates:
            if not path.exists():
                continue
            try:
                model = joblib.load(path)
                smoke_test(model)
            except Exception:
                logger.exception("Could not load model from %s", path)
                continue
            version = active or self._version_from_path(path)
            self._swap(model, version)
            logger.info("Loaded active model %s", version)
            return True
        return False

    def _version_from_path(self, path: Path) -> str | None:
        real = path.resolve().stem
        return real.removeprefix("model_") if real.startswith("model_") else None

    def hot_reload(self, new_model_path: str | Path, version_id: str | None = None) -> str | None:
        """Atomically replace the in-memory model. The old model stays active on any failure."""
        path = Path(new_model_path)
        model = joblib.load(path)  # slow part, outside the lock
        smoke_test(model)
        version = version_id or self._version_from_path(path)
        self._swap(model, version)
        logger.info("Hot-reloaded model %s", version)
        return version

    # -- promotion / rollback
    def promote(self, result: dict[str, Any]) -> str:
        """Record a trained model, persist it as the active version and hot-reload it.

        The model is loaded and smoke-tested first, so a corrupt file never replaces a working
        one; the in-memory swap happens last, in a single reference assignment.
        """
        version = result["version_id"]
        model = joblib.load(result["path"])
        smoke_test(model)
        previous = self.version
        with self.session_factory() as session:
            self.registry.record_run(
                session, version, result["record_count"], result["metrics"], result["path"]
            )
            try:
                self.registry.activate(session, version, vocabulary_of(model))
            except Exception:
                session.rollback()
                if previous:  # restore the persisted pointer to the still-loaded model
                    self.registry.activate(session, previous)
                raise
        self._swap(model, version)
        logger.info("Promoted model %s", version)
        return version

    def rollback(self) -> str:
        """Re-activate the previous stable version (e.g. after a bad deployment)."""
        with self._lock:
            current = self._version
        previous = self.registry.previous_version(current)
        if previous is None:
            raise ModelUnavailableError("No previous stable model version to roll back to")
        model = joblib.load(self.registry.model_path(previous))
        smoke_test(model)
        with self.session_factory() as session:
            self.registry.activate(session, previous, vocabulary_of(model))
        self._swap(model, previous)
        if current:
            self.registry.forget_active(current)
        logger.warning("Rolled back from %s to %s", current, previous)
        return previous

    # -- retraining
    def retrain(self) -> dict[str, Any]:
        """Train a candidate on all records; deploy it only if every validation guard passes.

        Raises ``RetrainInProgressError`` if another retrain is running, and the ``TrainingError``
        subclasses from :mod:`src.train` when the candidate is rejected (the active model is kept).
        """
        from src.train import train_model  # local import: train depends on this module

        if not self._retrain_lock.acquire(blocking=False):
            raise RetrainInProgressError("A retrain is already running")
        try:
            baseline = self._model
            with self.session_factory() as session:
                result = train_model(session, self.settings, baseline_model=baseline)
            previous_version = self.version
            self.promote(result)
            result["previous_version"] = previous_version
            return result
        finally:
            self._retrain_lock.release()

    def new_records_since_last_train(self) -> int:
        with self.session_factory() as session:
            total = session.scalar(select(func.count()).select_from(LCARecord)) or 0
            run = self.registry.latest_run(session)
            return total - (run.record_count if run else 0)
