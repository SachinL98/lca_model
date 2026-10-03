"""Inference engine: robust to casing, whitespace and categories the model has never seen."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.database import FEATURES, TARGETS
from src.model_manager import ModelManager, vocabulary_of


def canonicalise(value: str, known: list[str]) -> tuple[str, bool]:
    """Map ``value`` onto a known category ignoring case/whitespace; flag if unseen."""
    cleaned = " ".join(value.split())
    lookup = {k.casefold(): k for k in known}
    if cleaned.casefold() in lookup:
        return lookup[cleaned.casefold()], False
    return cleaned, True


def predict(
    manager: ModelManager,
    material: str,
    country: str,
    supply_chain_step: str,
    amount_kg: float = 1.0,
) -> dict[str, Any]:
    """Predict LCA metrics for ``amount_kg`` of material (values clipped at 0).

    Categories outside the training vocabulary are encoded as all-zeros by the pipeline's
    ``OneHotEncoder(handle_unknown='ignore')``; they are reported in ``unseen_categories`` so
    callers know the prediction relied on the remaining features only.
    """
    model, version = manager.get()
    vocab = vocabulary_of(model)
    inputs = {"material": material, "country": country, "supply_chain_step": supply_chain_step}
    row: dict[str, str] = {}
    unseen: list[str] = []
    for feature in FEATURES:
        row[feature], is_unseen = canonicalise(inputs[feature], vocab[feature])
        if is_unseen:
            unseen.append(feature)

    per_kg = model.predict(pd.DataFrame([row], columns=FEATURES))[0]
    per_kg = [max(0.0, float(v)) for v in per_kg]
    return {
        "model_version": version,
        "amount_kg": amount_kg,
        "input": row,
        "unseen_categories": unseen,
        "predictions": {t: v * amount_kg for t, v in zip(TARGETS, per_kg, strict=True)},
        "predictions_per_kg": dict(zip(TARGETS, per_kg, strict=True)),
    }
