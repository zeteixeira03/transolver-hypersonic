"""
Rebuild every results table of the README from the committed records, and check the README.

The four report scripts run on the per-run records in ``results/``. Their
output is written to ``results/reproduced.md``, which is the artifact of this
check. Every table row of the README's Results section must then match one line
of that output: the row's numbers appear on that line, in the same order, at
the precision the README prints. The script lists each row that does not
match, and exits with an error. Figures go to a temporary directory, so ``data/samples`` stays
untouched.

Usage
-----
    python scripts/check_reproduces.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NUMBER = re.compile(r"(?<![<\w.])-?\d+\.\d+")


def reports(fig_dir: Path) -> list[tuple[str, list[str]]]:
    """The report commands, each with the title its output gets in the artifact."""
    slice_runs = [str(RESULTS / f"slice_ablation/run_m{m}") for m in (8, 16, 32, 64)]
    return [
        ("Slice-count ablation", ["scripts/slice_ablation.py", *slice_runs,
                                  "--fig-out", str(fig_dir / "envelope.png")]),
        ("Uncertainty and decision rule", ["scripts/ensemble_uq.py", "--per-case",
                                           str(RESULTS / "ensemble/v3_per_case.json"),
                                           "--fig-out", str(fig_dir / "spread.png")]),
        ("Active-learning loop", ["scripts/loop_report.py",
                                  "--before", str(RESULTS / "ensemble/v2_per_case.json"),
                                  "--after", str(RESULTS / "ensemble/v3_per_case.json"),
                                  "--ledger", str(RESULTS / "ledger.db"),
                                  "--out", str(fig_dir / "loop"),
                                  "--fig-out", str(fig_dir / "loop.png")]),
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
    """True when some output line holds every number of the row, in order, as a subsequence.

    The README may leave out columns, so the row only has to be a subsequence,
    and it may round further, the way a person does: half up on the printed digits.
    """
    for produced in lines:
        it = iter(produced)
        if all(any(rounds_to(v, n) for v in it) for n in row):
            return True
    return False


def main() -> int:
    sections = []
    with tempfile.TemporaryDirectory() as tmp:
        for title, cmd in reports(Path(tmp)):
            out = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout
            sections.append(f"# {title}\n\n{out.strip()}\n")
    text = "\n".join(sections)
    # temporary paths change on every run and would make the artifact differ
    text = "\n".join(ln for ln in text.splitlines() if tmp not in ln) + "\n"
    (RESULTS / "reproduced.md").write_text(text, encoding="utf-8")

    produced = [NUMBER.findall(ln) for ln in text.splitlines()]
    rows = readme_rows()
    missing = [(ln, row) for ln, row in rows if not row_reproduces(row, produced)]
    if missing:
        print(f"{len(missing)} of {len(rows)} README table rows do not reproduce:")
        for ln, row in missing:
            print(f"  README.md:{ln}  {' '.join(row)}")
        return 1
    print(f"all {len(rows)} README result rows ({sum(len(r) for _, r in rows)} numbers) "
          f"reproduce "
          f"(artifact: results/reproduced.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
