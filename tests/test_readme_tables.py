"""Rebuild the README results tables from ``results/`` and check that every row reproduces.

Artifact: ``data/processed/reproduced.md``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARTIFACT = ROOT / "data/processed/reproduced.md"
NUMBER = re.compile(r"(?<![<\w.])-?\d+\.\d+")


def reports() -> list[tuple[str, list[str]]]:
    """The report commands, each with the title its output gets in the artifact."""
    slice_runs = [str(RESULTS / f"slice_ablation/run_m{m}") for m in (8, 16, 32, 64)]
    return [
        ("Slice-count ablation", ["scripts/slice_ablation.py", *slice_runs]),
        ("Uncertainty and decision rule", ["scripts/ensemble_uq.py", "--per-case",
                                           str(RESULTS / "ensemble/per_case.json")]),
        ("Physics priors", ["scripts/prior_ablation.py", "table",
                            *sorted(str(p) for p in (RESULTS / "prior_ablation").glob("run_*"))]),
    ]


def readme_rows() -> list[tuple[int, list[str]]]:
    """Table rows of the README's Results section that carry decimals, with their line numbers."""
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "## Results")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    rows = [(i + 1, NUMBER.findall(lines[i])) for i in range(start, end) if lines[i].startswith("|")]
    return [(ln, nums) for ln, nums in rows if nums]


def rounds_to(value: str, target: str) -> bool:
    """True when ``value`` rounds half up to ``target`` at the decimals ``target`` shows."""
    places = Decimal(1).scaleb(-len(target.split(".")[1]))
    return abs(Decimal(value)).quantize(places, ROUND_HALF_UP) == abs(Decimal(target))


def row_reproduces(row: list[str], lines: list[list[str]]) -> bool:
    """True when some output line holds the row as an ordered subsequence, rounded half up."""
    for produced in lines:
        it = iter(produced)
        if all(any(rounds_to(v, n) for v in it) for n in row):
            return True
    return False


def test_readme_tables_reproduce() -> None:
    sections = []
    for title, cmd in reports():
        out = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
        sections.append(f"# {title}\n\n{out.strip()}\n")
    text = "\n".join(sections)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(text, encoding="utf-8")

    produced = [NUMBER.findall(ln) for ln in text.splitlines()]
    rows = readme_rows()
    missing = [f"README.md:{ln}  {' '.join(row)}" for ln, row in rows
               if not row_reproduces(row, produced)]
    assert len(rows) >= 15, f"only {len(rows)} result rows found in the README"
    assert not missing, "README rows that do not reproduce:\n" + "\n".join(missing)
