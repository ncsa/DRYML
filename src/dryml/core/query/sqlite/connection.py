from __future__ import annotations

import os
from pathlib import Path
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from ..model import QueryIndexError
from . import SQLiteQueryIndexConfig, require_sqlite
from .utils import wal_runtime_is_known_safe


_REGISTRY_LOCK = threading.RLock()
_CONNECTION_REGISTRY = {}


@dataclass(slots=True)
class _SharedConnectionEntry:
    con: object
    file_identity: tuple[int, int]
    owner_count: int = 0
    lease_count: int = 0


class SQLiteConnectionManager:
    """Own SQLite connections for one index path per process and thread.

    Connections are opened lazily, keyed by `(pid, thread_id, readonly)`, and
    initialized with DRYML's required PRAGMAs. Cached connections are reopened
    when another process replaces the database path. If a process forks, the
    child uses a different PID key and opens its own connection instead of
    reusing the parent's connection object.
    """

    def __init__(self, config: SQLiteQueryIndexConfig):
        self.config = config
        self._owned_keys = {}

    @property
    def path(self) -> Path:
        if self.config.path is None:
            raise QueryIndexError("SQLite query index connection requires a database path.")
        return Path(self.config.path)

    def connection(self, *, readonly: bool = False):
        _, con = self._acquire_connection(readonly=readonly, reserve_lease=False)
        return con

    def _acquire_connection(self, *, readonly: bool, reserve_lease: bool):
        """Return a registry key and connection.

        Args:
            readonly: Open or reuse a read-only connection when true.
            reserve_lease: Increment the shared lease count before returning.

        Returns:
            The registry key and SQLite connection. Lease reservation, when
            requested, is atomic with registry acquisition.
        """

        key = self._key(readonly=readonly)
        path = self.path
        with _REGISTRY_LOCK:
            entry = _CONNECTION_REGISTRY.get(key)
            if entry is not None:
                if not _connection_is_open(entry.con):
                    _discard_entry_locked(key, entry)
                    self._owned_keys.pop(key, None)
                elif entry.file_identity != _file_identity(path):
                    if entry.lease_count:
                        raise QueryIndexError(
                            "SQLite query index database changed while a connection is actively leased."
                        )
                    _discard_entry_locked(key, entry)
                    self._owned_keys.pop(key, None)
                else:
                    self._claim_key_locked(key, entry)
                    if reserve_lease:
                        entry.lease_count += 1
                    return key, entry.con
            else:
                self._owned_keys.pop(key, None)

        sqlite3 = require_sqlite()
        timeout = float(self.config.busy_timeout)
        identity_before_open = _file_identity(path)
        if readonly:
            uri = f"file:{path.as_posix()}?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=timeout, isolation_level=None, check_same_thread=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(str(path), timeout=timeout, isolation_level=None, check_same_thread=True)
        try:
            self._initialize_connection(con, readonly=readonly)
        except Exception:
            con.close()
            raise
        file_identity = _file_identity(path)
        if file_identity is None:
            con.close()
            raise QueryIndexError("SQLite query index database disappeared while opening a connection.")
        if identity_before_open is not None and file_identity != identity_before_open:
            con.close()
            raise QueryIndexError("SQLite query index database changed while opening a connection.")
        with _REGISTRY_LOCK:
            if _file_identity(path) != file_identity:
                con.close()
                raise QueryIndexError("SQLite query index database changed while opening a connection.")
            entry = _CONNECTION_REGISTRY.get(key)
            if entry is not None:
                if not _connection_is_open(entry.con):
                    _discard_entry_locked(key, entry)
                    self._owned_keys.pop(key, None)
                elif entry.file_identity != _file_identity(path):
                    if entry.lease_count:
                        con.close()
                        raise QueryIndexError(
                            "SQLite query index database changed while a connection is actively leased."
                        )
                    _discard_entry_locked(key, entry)
                    self._owned_keys.pop(key, None)
                else:
                    con.close()
                    self._claim_key_locked(key, entry)
                    if reserve_lease:
                        entry.lease_count += 1
                    return key, entry.con
            else:
                self._owned_keys.pop(key, None)
            entry = _SharedConnectionEntry(con, file_identity)
            _CONNECTION_REGISTRY[key] = entry
            self._claim_key_locked(key, entry)
            if reserve_lease:
                entry.lease_count += 1
        return key, con

    @contextmanager
    def lease(self, *, readonly: bool = False):
        """Yield a shared connection and defer physical close while in use."""

        key, con = self._acquire_connection(
            readonly=readonly,
            reserve_lease=True,
        )
        try:
            yield con
        finally:
            with _REGISTRY_LOCK:
                entry = _CONNECTION_REGISTRY.get(key)
                if entry is not None and entry.con is con:
                    entry.lease_count = max(0, entry.lease_count - 1)
                    _close_entry_if_unused_locked(key, entry)

    def close_current(self) -> None:
        key_prefix = (os.getpid(), threading.get_ident())
        self._release_owned_keys(lambda key: key[:2] == key_prefix)

    def close_all_current_process(self) -> None:
        pid = os.getpid()
        self._release_owned_keys(lambda key: key[0] == pid)

    def active_lease_count(self, *, readonly: bool = False) -> int:
        key = self._key(readonly=readonly)
        with _REGISTRY_LOCK:
            entry = _CONNECTION_REGISTRY.get(key)
            return 0 if entry is None else entry.lease_count

    def close_path_current_process(self) -> None:
        """Close unused current-process connections for this database path."""

        pid = os.getpid()
        path = os.path.abspath(os.fspath(self.path))
        with _REGISTRY_LOCK:
            for key in tuple(_CONNECTION_REGISTRY):
                if key[0] != pid or key[2] != path:
                    continue
                entry = _CONNECTION_REGISTRY[key]
                if entry.lease_count > 0:
                    entry.owner_count = 0
                    continue
                _discard_entry_locked(key, entry)
                self._owned_keys.pop(key, None)

    def __del__(self):
        try:
            self.close_all_current_process()
        except Exception:
            pass

    def _initialize_connection(self, con, *, readonly: bool) -> None:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute(f"PRAGMA busy_timeout = {int(float(self.config.busy_timeout) * 1000)}")
        _execute_optional_pragma(con, "PRAGMA trusted_schema = OFF")
        if not readonly:
            self._configure_journal_and_durability(con)

        foreign_keys = con.execute("PRAGMA foreign_keys").fetchone()[0]
        if foreign_keys != 1:
            raise QueryIndexError("SQLite query index connection could not enable foreign keys.")

    def _key(self, *, readonly: bool):
        return (os.getpid(), threading.get_ident(), os.path.abspath(os.fspath(self.path)), readonly)

    def _claim_key_locked(self, key, entry: _SharedConnectionEntry) -> None:
        if self._owned_keys.get(key) is entry:
            return
        entry.owner_count += 1
        self._owned_keys[key] = entry

    def _release_owned_keys(self, predicate) -> None:
        with _REGISTRY_LOCK:
            for key in tuple(self._owned_keys):
                if not predicate(key):
                    continue
                owned_entry = self._owned_keys.pop(key)
                entry = _CONNECTION_REGISTRY.get(key)
                if entry is None or entry is not owned_entry:
                    continue
                entry.owner_count = max(0, entry.owner_count - 1)
                _close_entry_if_unused_locked(key, entry)

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


def _connection_is_open(con) -> bool:
    try:
        con.execute("SELECT 1").fetchone()
        return True
    except Exception:
        return False


def _file_identity(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino


def _discard_entry_locked(key, entry: _SharedConnectionEntry) -> None:
    _CONNECTION_REGISTRY.pop(key, None)
    try:
        entry.con.close()
    except Exception:
        pass


def _close_entry_if_unused_locked(key, entry: _SharedConnectionEntry) -> None:
    if entry.owner_count > 0 or entry.lease_count > 0:
        return
    _discard_entry_locked(key, entry)
