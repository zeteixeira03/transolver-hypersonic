"""SQLite work ledger for dataset generation.

The ledger holds one row per attempted case. A worker claims the lowest-ordinal
``pending`` row inside a single transaction, runs it, and writes back either
``done`` with the three quantities the sanity checks need, or ``failed`` with
the error text. At start-up a worker first reclaims every ``running`` row whose
lease has expired, which is what lets the sweep resume after a hard
interruption.

Column ``ord`` holds the nearest-neighbour order, and column ``block`` splits
that order into contiguous ranges. ``restart_from`` points at the previous case
in the same block, so a worker can warm-start a solve from the converged
solution beside it. It is NULL for the first case of a block and never crosses a
block boundary, because two blocks can land on two different machines. A worker
either takes one named block or takes any pending case.
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
# a claim increments attempts, so this budget counts interruptions rather than
# solver failures. a case cut off part way through, by a power cycle or a
# closed session, comes back to pending until it has been claimed this many
# times. a real SU2 failure goes to mark_failed and never spends the budget.
MAX_ATTEMPTS = 6


# ============================================================================================
#                                       connection
# ============================================================================================

def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the ledger in write-ahead log mode with a long busy timeout.

    Several workers share one ledger file, so the timeout must outlast a slow
    commit under contention.

    ``synchronous=FULL`` is a requirement here and not a preference. It flushes
    each commit to the device before the call returns, so ``mark_done`` survives
    a power cycle. Under ``NORMAL`` a hard restart can discard a write-ahead log
    that was never checkpointed and roll the ledger back by tens of cases. One
    flush per case costs nothing against a solve that runs for hours. The
    guarantee only holds on a physical disk: the virtual ext4 disk of the
    Windows Subsystem for Linux ignores write barriers, so the working directory
    must not live there.
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
    """Create or extend the ledger from an ordered list of case specifications.

    A second call adds only what is new, because a row whose ``case_id`` already
    exists is left alone. The columns follow from the list. ``case_id`` is the
    position of the specification in ``specs``, which already carries the run
    order, and ``ord`` repeats it. ``block`` comes from a split of the range
    into ``n_blocks`` contiguous parts. ``restart_from`` names the previous case
    only when that case shares both this ``geom_id`` and this ``block``, which
    means one mesh and one machine.

    Parameters
    ----------
    db_path : path
        Ledger file. Created if absent.
    specs : list of CaseSpec
        The output of :func:`src.data.sampler.sample_cases`.
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

    A row that has spent :data:`MAX_ATTEMPTS` goes to ``failed`` rather than
    back to ``pending``. Returns the number of rows changed. Call it once at
    worker start-up.
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
        Identifier written onto the claimed row, for example ``"oracle-3"``.
    block : int, optional
        If given, only claim cases in this block.

    Returns
    -------
    tuple of (case_id, Case, restart_from) or None
        ``None`` when no case remains to claim. ``restart_from`` holds the case
        id to warm-start from, or ``None``.
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
    """Return the full row for one case, or None if it is not in the ledger.

    The warm-start guard reads ``status`` and ``checks_passed`` together. One
    case may seed the next only when it finished without error and also cleared
    the analytical gates.
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
    """Return the cases that finished most recently, newest first.

    The periodic quality gate reads this window.
    """
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

    The connection opens read-only and immutable, which works on a read-only
    mount such as a Kaggle input directory, where ``connect`` would fail.
    """
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return {int(r["case_id"]): dict(r) for r in con.execute("SELECT * FROM cases")}
    finally:
        con.close()
