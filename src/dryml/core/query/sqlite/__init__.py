from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..model import QueryIndexUnavailable


JournalMode = Literal["auto", "wal", "delete"]
Durability = Literal["normal", "full"]


@dataclass(frozen=True, slots=True)
class SQLiteQueryIndexConfig:
    """Configuration for a Store-owned SQLite query-index sidecar.

    `path` defaults to the owning `DirStore` sidecar path when omitted.
    `journal_mode` selects rollback journal, WAL, or conservative automatic
    choice. `busy_timeout` and `max_write_retries` bound write contention waits.
    """

    path: str | Path | None = None
    journal_mode: JournalMode = "auto"
    durability: Durability = "normal"
    busy_timeout: float = 30.0
    max_write_retries: int = 6


def sqlite_available() -> bool:
    try:
        import sqlite3  # noqa: F401
    except ImportError:
        return False
    return True


def require_sqlite():
    try:
        import sqlite3
    except ImportError as exc:
        raise QueryIndexUnavailable("SQLite query index requires Python's optional sqlite3 module.") from exc
    return sqlite3


__all__ = [
    "Durability",
    "JournalMode",
    "SQLiteQueryIndexConfig",
    "require_sqlite",
    "sqlite_available",
]
