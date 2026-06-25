from __future__ import annotations

import os
from pathlib import Path
import threading

from ..model import QueryIndexError
from . import SQLiteQueryIndexConfig, require_sqlite
from .utils import wal_runtime_is_known_safe


class SQLiteConnectionManager:
    def __init__(self, config: SQLiteQueryIndexConfig):
        self.config = config
        self._connections = {}

    @property
    def path(self) -> Path:
        if self.config.path is None:
            raise QueryIndexError("SQLite query index connection requires a database path.")
        return Path(self.config.path)

    def connection(self, *, readonly: bool = False):
        key = (os.getpid(), threading.get_ident(), readonly)
        con = self._connections.get(key)
        if con is not None:
            return con

        sqlite3 = require_sqlite()
        path = self.path
        timeout = float(self.config.busy_timeout)
        if readonly:
            uri = f"file:{path.as_posix()}?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=timeout, isolation_level=None, check_same_thread=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(str(path), timeout=timeout, isolation_level=None, check_same_thread=True)
        self._initialize_connection(con, readonly=readonly)
        self._connections[key] = con
        return con

    def close_current(self) -> None:
        key_prefix = (os.getpid(), threading.get_ident())
        for key in list(self._connections):
            if key[:2] == key_prefix:
                self._connections.pop(key).close()

    def close_all_current_process(self) -> None:
        pid = os.getpid()
        for key in list(self._connections):
            if key[0] == pid:
                self._connections.pop(key).close()

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
