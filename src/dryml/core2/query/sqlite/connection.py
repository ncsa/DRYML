from __future__ import annotations

import os
from pathlib import Path
import threading

from ..model import QueryIndexError
from . import SQLiteQueryIndexConfig, require_sqlite
from .utils import wal_runtime_is_known_safe


class SQLiteConnectionManager:
    """Own SQLite connections for one index path per process and thread.

    Connections are opened lazily, keyed by `(pid, thread_id, readonly)`, and
    initialized with DRYML's required PRAGMAs. If a process forks, the child uses
    a different PID key and opens its own connection instead of reusing the
    parent's connection object.
    """

    def __init__(self, config: SQLiteQueryIndexConfig):
        self.config = config
        self._connections = {}
        self._file_identities = {}

    @property
    def path(self) -> Path:
        if self.config.path is None:
            raise QueryIndexError("SQLite query index connection requires a database path.")
        return Path(self.config.path)

    def connection(self, *, readonly: bool = False):
        """Return this process/thread's live connection to the active sidecar.

        Args:
            readonly: Open in SQLite read-only mode when ``True``.

        Returns:
            A configured ``sqlite3.Connection``. A cached connection outside a
            transaction is replaced when atomic sidecar publication changes the
            path's device/inode identity; active transactions remain pinned.

        Raises:
            QueryIndexError: If no sidecar path is configured or required SQLite
                connection settings cannot be established.
            QueryIndexUnavailable: If the optional SQLite backend is unavailable.
        """

        key = (os.getpid(), threading.get_ident(), readonly)
        con = self._connections.get(key)
        if con is not None:
            if con.in_transaction or self._file_identities.get(key) == self._path_identity():
                return con
            self._connections.pop(key).close()
            self._file_identities.pop(key, None)

        sqlite3 = require_sqlite()
        path = self.path
        timeout = float(self.config.busy_timeout)
        while True:
            before = self._path_identity()
            if readonly:
                uri = f"file:{path.as_posix()}?mode=ro"
                con = sqlite3.connect(uri, uri=True, timeout=timeout, isolation_level=None, check_same_thread=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                con = sqlite3.connect(str(path), timeout=timeout, isolation_level=None, check_same_thread=True)
            after = self._path_identity()
            if before == after or before is None:
                break
            con.close()
        self._initialize_connection(con, readonly=readonly)
        self._connections[key] = con
        self._file_identities[key] = after
        return con

    def close_current(self) -> None:
        key_prefix = (os.getpid(), threading.get_ident())
        for key in list(self._connections):
            if key[:2] == key_prefix:
                self._connections.pop(key).close()
                self._file_identities.pop(key, None)

    def close_all_current_process(self) -> None:
        pid = os.getpid()
        for key in list(self._connections):
            if key[0] == pid:
                self._connections.pop(key).close()
                self._file_identities.pop(key, None)

    def _path_identity(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return stat.st_dev, stat.st_ino

    def _initialize_connection(self, con, *, readonly: bool) -> None:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute(f"PRAGMA busy_timeout = {int(float(self.config.busy_timeout) * 1000)}")
        _execute_optional_pragma(con, "PRAGMA trusted_schema = OFF")
        if not readonly:
            self._configure_journal_and_durability(con)

        foreign_keys = con.execute("PRAGMA foreign_keys").fetchone()[0]
        if foreign_keys != 1:
            raise QueryIndexError("SQLite query index connection could not enable foreign keys.")

    def _configure_journal_and_durability(self, con) -> None:
        requested = self.config.journal_mode
        if requested == "auto":
            version = require_sqlite().sqlite_version_info
            requested = "wal" if wal_runtime_is_known_safe(version) else "delete"
        journal_mode = con.execute(f"PRAGMA journal_mode = {requested.upper()}").fetchone()[0].lower()
        if requested != "delete" and journal_mode != requested:
            raise QueryIndexError(f"SQLite query index could not enable journal_mode={requested!r}; got {journal_mode!r}.")
        synchronous = "NORMAL" if self.config.durability == "normal" else "FULL"
        con.execute(f"PRAGMA synchronous = {synchronous}")


def _execute_optional_pragma(con, sql: str) -> None:
    try:
        con.execute(sql)
    except Exception:
        return
