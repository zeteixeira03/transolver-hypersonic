"""SU2 wrapper: cfg rendering, subprocess driver, and output post-processing."""

from src.cfd.postprocess import (
    extract_axis_line,
    extract_surface,
    extract_training_tensors,
    find_shock_standoff,
    stagnation_values,
)
from src.cfd.runner import Case, render_cfg, run_case, run_su2

__all__ = [
    "Case",
    "render_cfg",
    "run_su2",
    "run_case",
    "extract_surface",
    "extract_axis_line",
    "stagnation_values",
    "find_shock_standoff",
    "extract_training_tensors",
]
