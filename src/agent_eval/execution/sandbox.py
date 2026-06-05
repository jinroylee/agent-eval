"""Safety sandbox for executing generated SQL.

Generated queries run against an ephemeral, READ-ONLY connection with op/time budget guards so a
runaway or mutating query can neither change data nor hang the harness. Never point this at a
production database without an explicit read-replica.
"""

from __future__ import annotations

import sqlite3
import time


def readonly_connection(
    db_path: str,
    timeout_s: float = 5.0,
    max_ops: int = 5_000_000,
    op_check_interval: int = 10_000,
) -> sqlite3.Connection:
    """Open a read-only SQLite connection that aborts on op-count or wall-clock budget overrun."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_s)
    start = time.monotonic()
    state = {"ops": 0}

    def _guard() -> int:
        state["ops"] += op_check_interval
        if state["ops"] > max_ops:
            return 1  # nonzero aborts the running statement
        if time.monotonic() - start > timeout_s:
            return 1
        return 0

    con.set_progress_handler(_guard, op_check_interval)
    return con
