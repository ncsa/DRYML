from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING
from weakref import WeakSet

from ....paths import to_file_uri
from ..model import QueryIndexError
from . import SQLiteQueryIndexConfig, require_sqlite
from .utils import wal_runtime_is_known_safe

if TYPE_CHECKING:
    import sqlite3


@contextmanager
def _open_connection(
        config: SQLiteQueryIndexConfig,
        *,
        readonly: bool = False) -> Iterator[sqlite3.Connection]:
    con = _connect(
        config,
        readonly=readonly,
        isolation_level="DEFERRED",
        check_same_thread=True,
    )
    try:
        _initialize_connection(config, con, readonly=readonly)
    except BaseException:
        try:
            con.close()
        except BaseException:
            pass
        raise

    try:
        yield con
    except BaseException:
        _rollback_and_close(con, suppress_errors=True)
        raise
    else:
        _rollback_and_close(con, suppress_errors=False)


def _connect(
        config: SQLiteQueryIndexConfig,
        *,
        readonly: bool,
        isolation_level: str | None,
        check_same_thread: bool):
    _validate_connection_arguments(config, readonly=readonly)
    path = _database_path(config)
    sqlite3 = require_sqlite()
    timeout = float(config.busy_timeout)
    if readonly:
        uri = f"{to_file_uri(path)}?mode=ro"
        return sqlite3.connect(
            uri,
            uri=True,
            timeout=timeout,
            isolation_level=isolation_level,
            check_same_thread=check_same_thread,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(
        str(path),
        timeout=timeout,
        isolation_level=isolation_level,
        check_same_thread=check_same_thread,
    )


def _validate_connection_arguments(
        config: SQLiteQueryIndexConfig,
        *,
        readonly: bool) -> None:
    if not isinstance(config, SQLiteQueryIndexConfig):
        raise TypeError("config must be a SQLiteQueryIndexConfig.")
    if type(readonly) is not bool:
        raise TypeError("readonly must be a bool.")


def _database_path(config: SQLiteQueryIndexConfig) -> Path:
    if not isinstance(config, SQLiteQueryIndexConfig):
        raise TypeError("config must be a SQLiteQueryIndexConfig.")
    if config.path is None:
        raise QueryIndexError(
            "SQLite query index connection requires a database path."
        )
    return Path(config.path)


def _rollback_and_close(con, *, suppress_errors: bool) -> None:
    try:
        if con.in_transaction:
            con.rollback()
    except BaseException:
        if not suppress_errors:
            raise
    finally:
        try:
            con.close()
        except BaseException:
            if not suppress_errors:
                raise


class SQLiteConnectionManager:
    """Own SQLite connections for one index path per process and thread.

    Connections are opened lazily in SQLite autocommit mode, keyed by
    ``(pid, thread_id, readonly)``, and initialized with DRYML's required
    PRAGMAs. If a process forks, the child uses a different PID key and opens
    its own connection instead of reusing the parent's connection object.
    Cached handles are never shared between keys, but disable SQLite's thread
    affinity so a coordinator can close handles after their worker threads end.

    Args:
        config: A ``SQLiteQueryIndexConfig`` containing database path, journal,
            durability, and contention settings.

    Side Effects:
        Registers the manager for same-thread path close barriers. Database
        handles and writable parent directories are created only when
        requested. Callers must not use a cached handle concurrently with
        manager teardown.
    """

    _instances = WeakSet()
    _instances_lock = threading.Lock()

    def __init__(self, config: SQLiteQueryIndexConfig):
        self.config = config
        self._connections = {}
        self._file_identities = {}
        with self._instances_lock:
            self._instances.add(self)

    @property
    def path(self) -> Path:
        """Return the configured database path.

        Returns:
            The configured path without requiring it to exist.

        Raises:
            TypeError: If the manager configuration is not a
                ``SQLiteQueryIndexConfig``.
            QueryIndexError: If the configuration has no database path.

        Side Effects:
            None.
        """

        return _database_path(self.config)

    def connection(self, *, readonly: bool = False) -> sqlite3.Connection:
        """Return this process/thread's live connection to the active sidecar.

        Args:
            readonly: A ``bool`` selecting SQLite read-only mode.

        Returns:
            A configured autocommit ``sqlite3.Connection`` owned by this
            manager. A cached connection outside a transaction is replaced
            when atomic sidecar publication changes the path's device/inode
            identity; active transactions remain pinned.

        Raises:
            TypeError: If the manager configuration is not a
                ``SQLiteQueryIndexConfig`` or ``readonly`` is not exactly a
                ``bool``.
            QueryIndexError: If no sidecar path is configured or required
                SQLite connection settings cannot be established.
            QueryIndexUnavailable: If the optional SQLite backend is
                unavailable.
            sqlite3.Error: If SQLite cannot open or configure the database.

        Side Effects:
            May create a writable database and its parent directory. Caches the
            resulting handle until an explicit close or file-identity change.
            The handle remains assigned exclusively to the current process,
            thread-key, and access mode despite permitting manager-owned close
            from a coordinator thread.
        """

        _validate_connection_arguments(self.config, readonly=readonly)
        key = (os.getpid(), threading.get_ident(), readonly)
        con = self._connections.get(key)
        if con is not None:
            if con.in_transaction:
                return con
            if self._file_identities.get(key) == self._path_identity():
                return con
            self._close_cached(key)

        while True:
            before = self._path_identity()
            con = _connect(
                self.config, readonly=readonly, isolation_level=None,
                check_same_thread=False,
            )
            after = self._path_identity()
            if before == after or before is None:
                break
            con.close()
        try:
            self._initialize_connection(con, readonly=readonly)
        except BaseException:
            con.close()
            raise
        self._connections[key] = con
        self._file_identities[key] = after
        return con

    def close_current(self) -> None:
        """Close cached connections owned by the current process and thread.

        Returns:
            None.

        Raises:
            sqlite3.Error: If SQLite cannot close a cached handle. The handle
                remains cached when close fails.

        Side Effects:
            Rolls back SQLite-managed active transactions as each handle
            closes.
        """

        key_prefix = (os.getpid(), threading.get_ident())
        for key in list(self._connections):
            if key[:2] == key_prefix:
                self._close_cached(key)

    def close_all_current_process(self) -> None:
        """Close every cached connection this manager owns in this process.

        Returns:
            None.

        Raises:
            sqlite3.Error: If SQLite cannot close a cached handle. The handle
                remains cached when close fails.

        Side Effects:
            Closes handles from all thread keys for the current process.
            Callers must ensure those handles are not concurrently in use.
        """

        pid = os.getpid()
        for key in list(self._connections):
            if key[0] == pid:
                self._close_cached(key)

    def _close_cached(self, key) -> None:
        con = self._connections[key]
        con.close()
        self._connections.pop(key, None)
        self._file_identities.pop(key, None)

    @classmethod
    def _close_current_thread_for_path(cls, path: str | Path) -> None:
        """Close same-thread handles for one canonical index path."""

        path_key = os.path.normcase(os.path.abspath(os.fspath(path)))
        with cls._instances_lock:
            managers = tuple(cls._instances)
        for manager in managers:
            manager_path = os.path.normcase(
                os.path.abspath(os.fspath(manager.path))
            )
            if manager_path == path_key:
                manager.close_current()

    def _path_identity(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return stat.st_dev, stat.st_ino

    def _initialize_connection(self, con, *, readonly: bool) -> None:
        _initialize_connection(self.config, con, readonly=readonly)


def _initialize_connection(config, con, *, readonly: bool) -> None:
    con.execute("PRAGMA foreign_keys = ON")
    timeout_ms = int(float(config.busy_timeout) * 1000)
    con.execute(f"PRAGMA busy_timeout = {timeout_ms}")
    _execute_optional_pragma(con, "PRAGMA trusted_schema = OFF")
    if not readonly:
        _configure_journal_and_durability(config, con)

    foreign_keys = con.execute("PRAGMA foreign_keys").fetchone()[0]
    if foreign_keys != 1:
        raise QueryIndexError(
            "SQLite query index connection could not enable foreign keys."
        )


def _configure_journal_and_durability(config, con) -> None:
    requested = config.journal_mode
    if requested == "auto":
        version = require_sqlite().sqlite_version_info
        requested = "wal" if wal_runtime_is_known_safe(version) else "delete"
    journal_mode = con.execute("PRAGMA journal_mode").fetchone()[0].lower()
    if journal_mode != requested:
        result = con.execute(f"PRAGMA journal_mode = {requested.upper()}")
        journal_mode = result.fetchone()[0].lower()
    if requested != "delete" and journal_mode != requested:
        raise QueryIndexError(
            "SQLite query index could not enable "
            f"journal_mode={requested!r}; got {journal_mode!r}."
        )
    synchronous = "NORMAL" if config.durability == "normal" else "FULL"
    con.execute(f"PRAGMA synchronous = {synchronous}")


def _execute_optional_pragma(con, sql: str) -> None:
    try:
        con.execute(sql)
    except Exception:
        return
