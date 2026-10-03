"""SQLAlchemy models and session factory."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from src.config import Settings, get_settings

FEATURES = ["material", "country", "supply_chain_step"]
TARGETS = ["co2_emission", "energy_consumption", "water_usage"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class LCARecord(Base):
    __tablename__ = "lca_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    material: Mapped[str] = mapped_column(String(100), index=True)
    country: Mapped[str] = mapped_column(String(100), index=True)
    supply_chain_step: Mapped[str] = mapped_column(String(100), index=True)
    co2_emission: Mapped[float | None] = mapped_column(Float, nullable=True)  # kg CO2e per kg
    energy_consumption: Mapped[float | None] = mapped_column(Float, nullable=True)  # MJ per kg
    water_usage: Mapped[float | None] = mapped_column(Float, nullable=True)  # L per kg
    source: Mapped[str] = mapped_column(String(100), default="api")
    external_id: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelRun(Base):
    __tablename__ = "model_runs"

    version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    record_count: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    path: Mapped[str | None] = mapped_column(String(500), nullable=True)


def make_engine(url: str) -> Engine:
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        return create_engine(url, pool_pre_ping=True)
    if parsed.database and parsed.database != ":memory:":
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"connect_args": {"check_same_thread": False}}
    if not parsed.database or parsed.database == ":memory:":
        kwargs["poolclass"] = StaticPool
    return create_engine(url, **kwargs)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


_default_factory: sessionmaker[Session] | None = None


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    """Process-wide default session factory (created and initialised lazily)."""
    global _default_factory
    if _default_factory is None:
        engine = make_engine((settings or get_settings()).DATABASE_URL)
        init_db(engine)
        _default_factory = make_session_factory(engine)
    return _default_factory


@contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    session = (factory or get_session_factory())()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
