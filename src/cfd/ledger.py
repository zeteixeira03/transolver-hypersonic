"""SQLite work ledger for dataset generation, one row per case.

``ord`` is the nearest-neighbour run order and ``block`` splits it into
contiguous ranges. ``restart_from`` names the previous case in the same block
for a warm start. It never crosses a block boundary, because two blocks can
run on two machines.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from src.cfd.runner import Case
from src.data.sampler import CaseSpec, partition

# ============================================================================================
#                                       schema
# ============================================================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id       INTEGER PRIMARY KEY,
    ord           INTEGER NOT NULL,
    block         INTEGER NOT NULL,
    group_name    TEXT    NOT NULL,
    geom_id       INTEGER NOT NULL,
    restart_from  INTEGER,
    R_n           REAL NOT NULL,
    theta_c_deg   REAL NOT NULL,
    R_b           REAL NOT NULL,
    R_s           REAL NOT NULL,
    mach          REAL NOT NULL,
    T_inf         REAL NOT NULL,
    p_inf         REAL NOT NULL,
    T_w           REAL NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    worker        TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    started_at    REAL,
    finished_at   REAL,
    qw            REAL,
    p02           REAL,
    standoff      REAL,
    checks_passed INTEGER,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_status_ord ON cases (status, ord);
"""

_CASE_COLS = ("R_n", "theta_c_deg", "R_b", "R_s", "mach", "T_inf", "p_inf", "T_w")
# counts claims, so it budgets interruptions (power cycles, closed sessions),
# not solver failures, which go straight to mark_failed
MAX_ATTEMPTS = 6


# ============================================================================================
#                                       connection
# ============================================================================================

def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the ledger in write-ahead log mode with a long busy timeout.

    ``synchronous=FULL`` is required: under ``NORMAL`` a hard restart can drop
    an uncheckpointed write-ahead log and roll the ledger back by tens of
    cases. This only holds on a physical disk. The virtual ext4 disk of the
    Windows Subsystem for Linux ignores write barriers, so the ledger must not
    live there.
    """
    con = sqlite3.connect(str(db_path), timeout=60.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA busy_timeout=60000")
    return con


def case_from_row(row: sqlite3.Row) -> Case:
    """Reconstruct a :class:`Case` from a ledger row."""
    return Case(**{c: row[c] for c in _CASE_COLS})


# ============================================================================================
#                                       initialisation
# ============================================================================================

def init_ledger(
    db_path: str | Path,
    specs: list[CaseSpec],
    n_blocks: int,
) -> None:
    """Create or extend the ledger from ``specs`` in run order.

    Existing ``case_id`` rows are left alone. ``case_id`` and ``ord`` are the
    position in ``specs``. ``restart_from`` is set only when the previous case
    shares both ``geom_id`` and ``block``.
    """
    blocks = partition(len(specs), n_blocks)
    block_of_pos = [0] * len(specs)
    for b, idxs in enumerate(blocks):
        for i in idxs:
            block_of_pos[i] = b

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = connect(db_path)
    try:
        con.executescript(_SCHEMA)
        with con:
            for pos, spec in enumerate(specs):
                block = block_of_pos[pos]
                prev = specs[pos - 1] if pos > 0 else None
                restart_from = (
                    pos - 1
                    if prev is not None
                    and prev.geom_id == spec.geom_id
                    and block_of_pos[pos - 1] == block
                    else None
                )
                c = spec.case
                con.execute(
                    """INSERT OR IGNORE INTO cases
                       (case_id, ord, block, group_name, geom_id, restart_from,
                        R_n, theta_c_deg, R_b, R_s, mach, T_inf, p_inf, T_w)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (pos, pos, block, spec.group, spec.geom_id, restart_from,
                     c.R_n, c.theta_c_deg, c.R_b, c.R_s,
                     c.mach, c.T_inf, c.p_inf, c.T_w),
                )
    finally:
        con.close()


# ============================================================================================
#                                       claim / report
# ============================================================================================

def reclaim_stale(db_path: str | Path, max_age_s: float = 7200.0) -> int:
    """Reclaim ``running`` rows whose lease is older than ``max_age_s``.

    A row that has spent :data:`MAX_ATTEMPTS` goes to ``failed`` instead of
    ``pending``. Returns the number of rows changed.
    """
    now = time.time()
    con = connect(db_path)
    try:
        with con:
            cur = con.execute(
                """UPDATE cases SET status='pending', worker=NULL, started_at=NULL
                   WHERE status='running' AND started_at < ? AND attempts < ?""",
                (now - max_age_s, MAX_ATTEMPTS),
            )
            n = cur.rowcount
            cur = con.execute(
                """UPDATE cases SET status='failed', error='lease expired, attempts exhausted'
                   WHERE status='running' AND started_at < ? AND attempts >= ?""",
                (now - max_age_s, MAX_ATTEMPTS),
            )
            n += cur.rowcount
        return n
    finally:
        con.close()


def claim_next(
    db_path: str | Path,
    worker: str,
    block: int | None = None,
) -> tuple[int, Case, int | None] | None:
    """Atomically claim the lowest-ordinal ``pending`` case, optionally in one block.

    Returns ``(case_id, case, restart_from)``, or None when nothing is pending.
    """
    con = connect(db_path)
    try:
        con.execute("BEGIN IMMEDIATE")
        if block is None:
            row = con.execute(
                "SELECT * FROM cases WHERE status='pending' ORDER BY ord LIMIT 1"
            ).fetchone()
        else:
            row = con.execute(
                "SELECT * FROM cases WHERE status='pending' AND block=? ORDER BY ord LIMIT 1",
                (block,),
            ).fetchone()
        if row is None:
            con.execute("ROLLBACK")
            return None
        con.execute(
            """UPDATE cases SET status='running', worker=?, attempts=attempts+1,
               started_at=?, finished_at=NULL, error=NULL WHERE case_id=?""",
            (worker, time.time(), row["case_id"]),
        )
        con.execute("COMMIT")
        return int(row["case_id"]), case_from_row(row), row["restart_from"]
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def mark_done(
    db_path: str | Path,
    case_id: int,
    *,
    qw: float,
    p02: float,
    standoff: float,
    checks_passed: bool,
) -> None:
    """Record a finished case with the three quantities the sanity checks read."""
    con = connect(db_path)
    try:
        with con:
            con.execute(
                """UPDATE cases SET status='done', finished_at=?, qw=?, p02=?,
                   standoff=?, checks_passed=?, error=NULL WHERE case_id=?""",
                (time.time(), qw, p02, standoff, int(checks_passed), case_id),
            )
    finally:
        con.close()


def mark_failed(db_path: str | Path, case_id: int, error: str) -> None:
    """Record a failed case. The row stays ``failed`` until a person retries it."""
    con = connect(db_path)
    try:
        with con:
            con.execute(
                "UPDATE cases SET status='failed', finished_at=?, error=? WHERE case_id=?",
                (time.time(), error[:2000], case_id),
            )
    finally:
        con.close()


# ============================================================================================
#                                       reporting
# ============================================================================================

def case_row(db_path: str | Path, case_id: int) -> sqlite3.Row | None:
    """Return the full row for one case, or None if it is not in the ledger."""
    con = connect(db_path)
    try:
        return con.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
    finally:
        con.close()


def status_counts(db_path: str | Path) -> dict[str, int]:
    """Return a {status: count} dict over the whole ledger."""
    con = connect(db_path)
    try:
        rows = con.execute("SELECT status, COUNT(*) n FROM cases GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}
    finally:
        con.close()


def recent_done(db_path: str | Path, limit: int = 50) -> list[sqlite3.Row]:
    """Return the cases that finished most recently, newest first."""
    con = connect(db_path)
    try:
        return con.execute(
            "SELECT * FROM cases WHERE status='done' ORDER BY finished_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        con.close()


def all_rows(db_path: str | Path) -> list[sqlite3.Row]:
    """Return every case row in ``ord`` order, for the dataset manifest."""
    con = connect(db_path)
    try:
        return con.execute("SELECT * FROM cases ORDER BY ord").fetchall()
    finally:
        con.close()


def read_cases(db_path: str | Path) -> dict[int, dict]:
    """Return every row as ``{case_id: {column: value}}`` without writing anything.

    Opens read-only and immutable, so it works on a read-only Kaggle input
    mount where ``connect`` fails.
    """
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return {int(r["case_id"]): dict(r) for r in con.execute("SELECT * FROM cases")}
    finally:
        con.close()
