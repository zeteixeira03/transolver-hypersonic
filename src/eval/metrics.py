"""
How a prediction is scored, and what the evaluation tiers are called.

The score of one case is the relative L2 error per channel, in physical units:
the norm of the error over all mesh nodes divided by the norm of the true field.
A number of 0.1 means the predicted field is off by 10% of the field itself.
"""

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
    """Relative L2 error per channel over the nodes of one case.

    Parameters
    ----------
    pred, true : torch.Tensor
        Fields of shape (N, C) in physical units.

    Returns
    -------
    torch.Tensor
        Shape (C,).
    """
    return (pred - true).pow(2).sum(dim=0).sqrt() / true.pow(2).sum(dim=0).sqrt()


def channel_mean(values: dict[str, float], prefix: str = "") -> float:
    """Mean over the four predicted primitives of ``values[prefix + channel]``."""
    return float(np.mean([values[prefix + c] for c in TARGET_ORDER]))


def tier_label(split: str) -> str:
    """Name of an evaluation split as the README tables print it."""
    return TIER_LABELS.get(split, split.removeprefix("ood_") + " (extrapolation)")


def tiers_present(records: list[dict]) -> list[str]:
    """Evaluation tiers present in per-case records, in report order.

    The validation split only calibrates thresholds, so it is not a tier.
    """
    present = {r["split"] for r in records}
    return [s for s in EVAL_SPLITS if s != "val" and s in present]
