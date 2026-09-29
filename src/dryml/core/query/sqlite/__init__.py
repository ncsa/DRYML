from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..model import QueryIndexUnavailable

if TYPE_CHECKING:
    import sqlite3


JournalMode = Literal["auto", "wal", "delete"]
Durability = Literal["normal", "full"]


@dataclass(frozen=True, slots=True)
class SQLiteQueryIndexConfig:
    """Configuration for a Store-owned SQLite query-index sidecar.

    ``path`` defaults to the owning ``DirStore`` sidecar path when omitted.
    ``journal_mode`` selects rollback journal, WAL, or conservative automatic
    choice. ``busy_timeout`` and ``max_write_retries`` bound write contention
    waits.
    """

    path: str | Path | None = None
    journal_mode: JournalMode = "auto"
    durability: Durability = "normal"
    busy_timeout: float = 30.0
    max_write_retries: int = 6


@contextmanager
def open_connection(
        config: SQLiteQueryIndexConfig,
        *,
        readonly: bool = False) -> Iterator[sqlite3.Connection]:
    """Open and own one configured SQLite query-index connection.

    The yielded connection uses SQLite's deferred transaction behavior. The
    outer context owns only the handle: it never commits, rolls back any
    transaction still open at exit, and always closes the connection. Use
    ``with con:`` inside this context to commit that inner block on success or
    roll it back when its body raises.

    Args:
        config: A :class:`SQLiteQueryIndexConfig` containing database path,
            journal, durability, and contention settings. ``config.path`` must
            not be ``None``.
        readonly: A :class:`bool` selecting SQLite read-only mode.
            Read-only opening does not create the database or its parent
            directory.

    Yields:
        A configured, exclusively owned :class:`sqlite3.Connection`.

    Raises:
        TypeError: If ``config`` is not a :class:`SQLiteQueryIndexConfig` or
            ``readonly`` is not exactly a :class:`bool`.
        QueryIndexError: If no path is configured or required connection
            settings cannot be established.
        QueryIndexUnavailable: If Python's optional SQLite backend is missing.
        sqlite3.Error: If SQLite cannot open, configure, or use the database.

    Side Effects:
        Writable opening creates missing parent directories and may create the
        database or update its journal settings. Exit rolls back uncommitted
        work and closes the handle, including when the body raises or is
        cancelled. A body exception remains the reported exception if cleanup
        also fails. Standalone handles retain the standard-library SQLite
        creator-thread affinity.
    """

    from .connection import _open_connection

    with _open_connection(config, readonly=readonly) as con:
        yield con


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
        raise QueryIndexUnavailable(
            "SQLite query index requires Python's optional sqlite3 module."
        ) from exc
    return sqlite3


__all__ = [
    "Durability",
    "JournalMode",
    "SQLiteQueryIndexConfig",
    "open_connection",
    "require_sqlite",
    "sqlite_available",
]
