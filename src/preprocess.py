"""Data extraction from the database and baseline seeding from the LCA TSV export."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database import FEATURES, TARGETS, LCARecord, get_session_factory

logger = logging.getLogger(__name__)

COMPOUND_TO_TARGET = {
    "carbon dioxide": "co2_emission",
    "energy": "energy_consumption",
    "water": "water_usage",
}


def load_baseline_frame(tsv_path: str | Path) -> pd.DataFrame:
    """Read the TSV and pivot ``compound`` rows into one row per material/country/step."""
    raw = pd.read_csv(tsv_path, sep="\t")
    raw = raw[raw["compound"].isin(COMPOUND_TO_TARGET)].copy()
    for col in FEATURES:
        raw[col] = raw[col].astype(str).str.strip()
    raw["target"] = raw["compound"].map(COMPOUND_TO_TARGET)
    raw["value"] = pd.to_numeric(raw["input_or_output_per_kg"], errors="coerce")

    wide = raw.pivot_table(index=FEATURES, columns="target", values="value", aggfunc="mean")
    wide = wide.reindex(columns=TARGETS).reset_index()
    wide.columns.name = None
    return fill_missing_targets(wide)


def fill_missing_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing targets with group medians, from the most to the least specific group."""
    df = df.copy()
    for target in TARGETS:
        for group in (["material", "supply_chain_step"], ["supply_chain_step"], ["material"]):
            df[target] = df[target].fillna(df.groupby(group)[target].transform("median"))
        df[target] = df[target].fillna(df[target].median())
    return df.dropna(subset=TARGETS)


def seed_baseline_from_tsv(tsv_path: str | Path, session: Session | None = None) -> int:
    """Insert the baseline TSV into ``lca_records``. Returns the number of rows inserted."""
    frame = load_baseline_frame(tsv_path)
    owns_session = session is None
    session = session or get_session_factory()()
    try:
        session.add_all(
            LCARecord(source="baseline", **{k: row[k] for k in FEATURES + TARGETS})
            for row in frame.to_dict("records")
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        if owns_session:
            session.close()
    logger.info("Seeded %d baseline records from %s", len(frame), tsv_path)
    return len(frame)


def seed_if_empty(session: Session, tsv_path: str | Path) -> int:
    """Seed the baseline only when the table is empty; returns rows inserted (0 if skipped)."""
    if session.scalar(select(func.count()).select_from(LCARecord)):
        return 0
    if not Path(tsv_path).exists():
        logger.warning("Baseline TSV %s not found; skipping seed", tsv_path)
        return 0
    return seed_baseline_from_tsv(tsv_path, session)


def load_training_frame(session: Session) -> pd.DataFrame:
    """All clean (fully populated, finite, non-negative) records as a DataFrame."""
    rows = session.execute(select(*[getattr(LCARecord, c) for c in FEATURES + TARGETS])).all()
    df = pd.DataFrame(rows, columns=FEATURES + TARGETS)
    if df.empty:
        return df
    df = df.dropna(subset=FEATURES + TARGETS)
    for col in FEATURES:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df[FEATURES] != "").all(axis=1)]
    return df[(df[TARGETS] >= 0).all(axis=1)].reset_index(drop=True)
