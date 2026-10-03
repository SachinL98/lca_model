"""Model versioning: run history in the DB, ``metadata.json`` and the ``current_model`` pointer."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.database import ModelRun

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
