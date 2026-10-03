"""Record validation/ingestion and synchronisation with an external verified-LCA API.

Expected upstream contract (``GET EXTERNAL_DATA_API_URL`` with ``Authorization: Bearer <key>``):
a JSON list, or an object ``{"records": [...], "next": "<url>"}`` (``next`` optional, for
pagination). Each record has ``material``, ``country``, ``supply_chain_step``, ``co2_emission``,
``energy_consumption``, ``water_usage`` and optionally ``id`` (stable upstream identifier, used
for de-duplication) and ``verified`` (records with ``verified: false`` are skipped).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings, get_settings
from src.database import FEATURES, TARGETS, LCARecord, get_session_factory

logger = logging.getLogger(__name__)

MAX_PAGES = 100


class SyncError(RuntimeError):
    """The external service could not be reached or returned an unusable payload."""


class RecordIn(BaseModel):
    """A verified ground-truth LCA record (all values are per kg of material)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    material: str = Field(min_length=1, max_length=100)
    country: str = Field(min_length=1, max_length=100)
    supply_chain_step: str = Field(min_length=1, max_length=100)
    co2_emission: float = Field(ge=0)
    energy_consumption: float = Field(ge=0)
    water_usage: float = Field(ge=0)
    external_id: str | None = Field(default=None, max_length=200)

    @field_validator("co2_emission", "energy_consumption", "water_usage")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("must be a finite number")
        return v


def _record_key(r: dict[str, Any] | LCARecord) -> tuple:
    get = r.get if isinstance(r, dict) else lambda k: getattr(r, k)
    return tuple(get(k) for k in FEATURES + TARGETS)


class DataSyncClient:
    def __init__(
        self,
        session_factory: sessionmaker[Session] | None = None,
        settings: Settings | None = None,
        http_client: httpx.Client | None = None,
    ):
        self.settings = settings or get_settings()
        self.session_factory = session_factory or get_session_factory(self.settings)
        self._http = http_client

    # ------------------------------------------------------------------ ingestion
    def insert_custom_records(
        self, records: list[dict[str, Any]], source: str = "api", skip_duplicates: bool = False
    ) -> dict[str, Any]:
        """Validate and bulk insert raw records. Invalid records are reported, not inserted."""
        valid: list[RecordIn] = []
        errors: list[dict[str, Any]] = []
        for i, raw in enumerate(records):
            try:
                valid.append(RecordIn.model_validate(raw))
            except ValidationError as exc:
                errors.append(
                    {"index": i, "error": exc.errors(include_url=False, include_input=False)}
                )

        duplicates = 0
        with self.session_factory() as session:
            if skip_duplicates and valid:
                valid, duplicates = self._drop_duplicates(session, valid, source)
            session.add_all(LCARecord(source=source, **r.model_dump()) for r in valid)
            session.commit()
        return {
            "received": len(records),
            "inserted": len(valid),
            "duplicates": duplicates,
            "invalid": len(errors),
            "errors": errors,
        }

    @staticmethod
    def _drop_duplicates(
        session: Session, candidates: list[RecordIn], source: str
    ) -> tuple[list[RecordIn], int]:
        known_ids = {
            e
            for (e,) in session.execute(
                select(LCARecord.external_id).where(LCARecord.external_id.is_not(None))
            )
        }
        known_keys = {_record_key(r) for r in session.scalars(select(LCARecord))}
        kept: list[RecordIn] = []
        for rec in candidates:
            key = _record_key(rec.model_dump())
            if (rec.external_id and rec.external_id in known_ids) or (
                not rec.external_id and key in known_keys
            ):
                continue
            kept.append(rec)
            known_keys.add(key)
            if rec.external_id:
                known_ids.add(rec.external_id)
        return kept, len(candidates) - len(kept)

    # ------------------------------------------------------------------ external sync
    def _fetch_all(self) -> list[dict[str, Any]]:
        url: str | None = self.settings.EXTERNAL_DATA_API_URL
        headers = {"Accept": "application/json"}
        if self.settings.EXTERNAL_DATA_API_KEY:
            headers["Authorization"] = f"Bearer {self.settings.EXTERNAL_DATA_API_KEY}"
        client = self._http or httpx.Client(timeout=self.settings.EXTERNAL_DATA_API_TIMEOUT)
        records: list[dict[str, Any]] = []
        try:
            for _ in range(MAX_PAGES):
                if not url:
                    break
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                if isinstance(payload, list):
                    records.extend(payload)
                    break
                if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
                    raise SyncError("Unexpected payload: expected a list or {'records': [...]}")
                records.extend(payload["records"])
                url = payload.get("next")
        except httpx.HTTPError as exc:
            raise SyncError(f"External API request failed: {exc}") from exc
        except ValueError as exc:
            raise SyncError(f"External API returned invalid JSON: {exc}") from exc
        finally:
            if self._http is None:
                client.close()
        return records

    def sync_from_external_api(self) -> dict[str, Any]:
        """Poll the external API for verified records, de-duplicate and bulk insert them."""
        if not self.settings.EXTERNAL_DATA_API_URL:
            return {"skipped": True, "reason": "EXTERNAL_DATA_API_URL is not configured"}

        fetched = self._fetch_all()
        candidates: list[dict[str, Any]] = []
        unverified = 0
        for item in fetched:
            if not isinstance(item, dict):
                candidates.append({})  # reported as invalid
                continue
            if item.get("verified") is False:
                unverified += 1
                continue
            item = dict(item)
            if item.get("id") is not None and not item.get("external_id"):
                item["external_id"] = str(item["id"])
            candidates.append({k: v for k, v in item.items() if k in RecordIn.model_fields})

        result = self.insert_custom_records(candidates, source="external", skip_duplicates=True)
        result.update(skipped=False, fetched=len(fetched), unverified=unverified)
        logger.info("External sync: %s", {k: v for k, v in result.items() if k != "errors"})
        return result


__all__ = ["DataSyncClient", "RecordIn", "SyncError", "FEATURES", "TARGETS"]
