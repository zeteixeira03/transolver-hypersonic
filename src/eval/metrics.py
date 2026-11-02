"""Relative L2 scoring in physical units, and the names of the evaluation tiers."""

from __future__ import annotations

import numpy as np
import torch

from src.data.su2 import OOD_GROUPS, TARGET_ORDER

EVAL_SPLITS = ("val", "test", "test_interp", *(f"ood_{g}" for g in OOD_GROUPS))
TIER_LABELS = {
    "test_interp": "interpolation",
    "test": "family holdout",
}


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Relative L2 error per channel, (N, C) fields to shape (C,)."""
    return (pred - true).pow(2).sum(dim=0).sqrt() / true.pow(2).sum(dim=0).sqrt()


def channel_mean(values: dict[str, float], prefix: str = "") -> float:
    """Mean over the four predicted primitives of ``values[prefix + channel]``."""
    return float(np.mean([values[prefix + c] for c in TARGET_ORDER]))


def tier_label(split: str) -> str:
    """Name of an evaluation split as the README tables print it."""
    return TIER_LABELS.get(split, split.removeprefix("ood_") + " (extrapolation)")


def tiers_present(records: list[dict]) -> list[str]:
    """Evaluation tiers present in per-case records, in report order, without ``val``."""
    present = {r["split"] for r in records}
    return [s for s in EVAL_SPLITS if s != "val" and s in present]
