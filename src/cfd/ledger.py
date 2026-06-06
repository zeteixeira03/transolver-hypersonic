"""SQLite work ledger for Phase 3 dataset generation.

One row per attempted case. A worker claims the lowest-ordinal ``pending`` row
in an atomic transaction, runs it, and writes back ``done`` (with the three
postprocessed sanity quantities) or ``failed`` (with the error string). A
janitor pass at worker start-up reclaims ``running`` rows whose lease has
expired -- the standard pattern for resuming after a hard interruption.

The ledger is laid out in nearest-neighbour order (column ``ord``) and split
into ``block`` ranges. ``restart_from`` points at the previous case in the same
block, so a worker can warm-start a solve from its predecessor's converged
solution; it is NULL for the first case of each block and never crosses a block
boundary (blocks may be handed to different machines). A worker can be pinned to
a block or left free to take any pending case.
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
MAX_ATTEMPTS = 2


# ============================================================================================
#                                       connection
# ============================================================================================

def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the ledger with WAL and a generous busy timeout for many workers.

    ``synchronous=FULL`` fsyncs every commit. On a free-tier laptop that can be
    hard-restarted at any moment, ``NORMAL`` lost an un-checkpointed WAL and rolled
    the ledger back ~40 cases; ``FULL`` persists each ``mark_done`` to disk before
    it returns. The cost is one fsync per case, negligible against a multi-hour
    solve, and it only pays off if the ledger lives on a real device (the WSL
    ext4 vhdx ignores barriers, so the workdir must sit on a physical disk).
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
    """Create (or extend) the ledger from an ordered case-spec list.

    Idempotent at the row level: rows whose ``case_id`` already exists are left
    untouched, so re-running only adds anything new. ``case_id`` is the position
    of the spec in ``specs`` (which is already in processing order), ``ord``
    equals ``case_id``, ``block`` comes from splitting the range into
    ``n_blocks`` contiguous parts, and ``restart_from`` is the previous case iff
    it shares this case's ``geom_id`` and ``block`` (same mesh, same machine).

    Parameters
    ----------
    db_path : path
        Ledger file (created if absent).
    specs : list of CaseSpec
        Output of :func:`src.data.sampler.sample_cases`.
    n_blocks : int
        Number of contiguous blocks.
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

    Rows that have already used up :data:`MAX_ATTEMPTS` are moved to ``failed``
    instead of back to ``pending``. Returns the number of rows touched. Call this
    once at worker start-up.
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
    """Atomically claim the lowest-ordinal ``pending`` case.

    Parameters
    ----------
    db_path : path
        Ledger file.
    worker : str
        Identifier recorded on the claimed row (e.g. ``"oracle-3"``).
    block : int, optional
        If given, only claim cases in this block.

    Returns
    -------
    tuple of (case_id, Case, restart_from) or None
        ``None`` when no claimable case remains. ``restart_from`` is the
        predecessor case id to warm-start from, or ``None``.
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
    """Record a finished case and its three postprocessed sanity quantities."""
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
    """Record a failed case. Stays ``failed``; reclaim/retry is a manual choice."""
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

def case_status(db_path: str | Path, case_id: int) -> str | None:
    """Return the ``status`` of one case, or None if it is not in the ledger."""
    con = connect(db_path)
    try:
        row = con.execute("SELECT status FROM cases WHERE case_id=?", (case_id,)).fetchone()
        return row["status"] if row else None
    finally:
        con.close()


def case_row(db_path: str | Path, case_id: int) -> sqlite3.Row | None:
    """Return the full row for one case, or None if it is not in the ledger.

    Used by the warm-start guard to inspect ``status`` and ``checks_passed``
    together: a predecessor warm-starts a successor only if it both finished
    cleanly and cleared the analytical gates.
    """
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
    """Return the most recently finished cases (newest first), for the quality gate."""
    con = connect(db_path)
    try:
        return con.execute(
            "SELECT * FROM cases WHERE status='done' ORDER BY finished_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        con.close()


def all_done(db_path: str | Path) -> list[sqlite3.Row]:
    """Return every finished case ordered by ``ord`` (for the final quality plot)."""
    con = connect(db_path)
    try:
        return con.execute("SELECT * FROM cases WHERE status='done' ORDER BY ord").fetchall()
    finally:
        con.close()


def all_rows(db_path: str | Path) -> list[sqlite3.Row]:
    """Return every case row ordered by ``ord`` (for the dataset manifest)."""
    con = connect(db_path)
    try:
        return con.execute("SELECT * FROM cases ORDER BY ord").fetchall()
    finally:
        con.close()
