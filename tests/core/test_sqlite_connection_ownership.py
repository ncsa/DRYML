import threading

import pytest

from dryml.core.query.model import QueryIndexError, QueryIndexUnavailable
from dryml.core.query.sqlite import (
    SQLiteQueryIndexConfig,
    open_connection,
    sqlite_available,
)
import dryml.core.query.sqlite.connection as connection_module
from dryml.core.query.sqlite.connection import SQLiteConnectionManager


pytestmark = pytest.mark.skipif(
    not sqlite_available(), reason="sqlite3 is unavailable"
)


def _config(path):
    return SQLiteQueryIndexConfig(
        path, journal_mode="delete", busy_timeout=1.0
    )


def _assert_closed(con):
    with pytest.raises(connection_module.require_sqlite().ProgrammingError):
        con.execute("SELECT 1")


def test_connection_rejects_invalid_config_without_filesystem_effects(
        tmp_path):
    parent = tmp_path / "must-not-exist"

    class InvalidConfig:
        path = parent / "index.sqlite"

    for config in (None, {}, InvalidConfig()):
        with pytest.raises(
                TypeError, match="config must be a SQLiteQueryIndexConfig"):
            with open_connection(config):
                pass

    assert not parent.exists()


@pytest.mark.parametrize("readonly", (None, 0, 1, "true"))
def test_connection_rejects_non_bool_readonly_without_filesystem_effects(
        tmp_path, readonly):
    path = tmp_path / "must-not-exist" / "index.sqlite"

    with pytest.raises(TypeError, match="readonly must be a bool"):
        with open_connection(_config(path), readonly=readonly):
            pass

    assert not path.parent.exists()


def test_connection_requires_configured_path_before_sqlite_import(monkeypatch):
    def unexpected_import():
        raise AssertionError("SQLite import must follow argument validation")

    monkeypatch.setattr(connection_module, "require_sqlite", unexpected_import)

    with pytest.raises(QueryIndexError, match="requires a database path"):
        with open_connection(SQLiteQueryIndexConfig()):
            pass


@pytest.mark.parametrize(
    "relative_path",
    (
        "hash # directory/index.sqlite",
        "percent %23 directory/index.sqlite",
        "spaces directory/index name.sqlite",
        "unicode-\N{SNOWMAN}/index-\N{GREEK SMALL LETTER LAMDA}.sqlite",
    ),
)
def test_readonly_connections_escape_reserved_and_unicode_paths(
        tmp_path, relative_path):
    path = tmp_path / relative_path
    config = _config(path)
    with open_connection(config) as con:
        with con:
            con.execute("CREATE TABLE values_table (value TEXT NOT NULL)")
            con.execute("INSERT INTO values_table VALUES ('expected')")

    with open_connection(config, readonly=True) as con:
        row = con.execute("SELECT value FROM values_table").fetchone()
        assert row == ("expected",)

    manager = SQLiteConnectionManager(config)
    try:
        managed = manager.connection(readonly=True)
        assert managed.isolation_level is None
        assert managed.execute(
            "SELECT value FROM values_table"
        ).fetchone() == ("expected",)
    finally:
        manager.close_all_current_process()


def test_readonly_connection_does_not_create_target_or_parent(tmp_path):
    path = tmp_path / "missing parent # %23" / "missing database.sqlite"

    with pytest.raises(connection_module.require_sqlite().OperationalError):
        with open_connection(_config(path), readonly=True):
            pass

    assert not path.exists()
    assert not path.parent.exists()


def test_readonly_connection_rejects_mutation(tmp_path):
    config = _config(tmp_path / "readonly.sqlite")
    with open_connection(config) as con:
        with con:
            con.execute("CREATE TABLE values_table (value INTEGER NOT NULL)")

    with open_connection(config, readonly=True) as con:
        with pytest.raises(
                connection_module.require_sqlite().OperationalError,
                match="readonly"):
            con.execute("INSERT INTO values_table VALUES (1)")


def test_inner_connection_context_commits_and_rolls_back(tmp_path):
    config = _config(tmp_path / "transactions.sqlite")
    with open_connection(config) as con:
        assert con.isolation_level == "DEFERRED"
        assert con.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert con.execute("PRAGMA busy_timeout").fetchone() == (1000,)
        with con:
            con.execute("CREATE TABLE values_table (value TEXT NOT NULL)")
            con.execute("INSERT INTO values_table VALUES ('committed')")

        with pytest.raises(RuntimeError, match="rollback"):
            with con:
                con.execute("INSERT INTO values_table VALUES ('rolled back')")
                raise RuntimeError("rollback")

        assert con.execute("SELECT value FROM values_table").fetchall() == [
            ("committed",)
        ]

    with open_connection(config, readonly=True) as con:
        assert con.execute("SELECT value FROM values_table").fetchall() == [
            ("committed",)
        ]


def test_outer_connection_context_rolls_back_uncommitted_work(tmp_path):
    config = _config(tmp_path / "outer-rollback.sqlite")
    with open_connection(config) as con:
        with con:
            con.execute("CREATE TABLE values_table (value TEXT NOT NULL)")
        con.execute("INSERT INTO values_table VALUES ('uncommitted')")

    with open_connection(config, readonly=True) as con:
        assert con.execute("SELECT value FROM values_table").fetchall() == []


class _Cancellation(BaseException):
    pass


@pytest.mark.parametrize(
    "error", (RuntimeError("body"), _Cancellation("cancelled"))
)
def test_connection_always_closes_and_preserves_body_exception(
        tmp_path, error):
    config = _config(tmp_path / "body-exception.sqlite")
    con = None

    with pytest.raises(type(error)) as caught:
        with open_connection(config) as con:
            raise error

    assert caught.value is error
    _assert_closed(con)


def test_body_exception_survives_rollback_and_close_failures(
        tmp_path, monkeypatch):
    events = []
    body_error = RuntimeError("body failed")

    class Connection:
        in_transaction = True

        def rollback(self):
            events.append("rollback")
            raise OSError("rollback failed")

        def close(self):
            events.append("close")
            raise OSError("close failed")

    monkeypatch.setattr(
        connection_module, "_connect", lambda *args, **kwargs: Connection()
    )
    monkeypatch.setattr(
        connection_module,
        "_initialize_connection",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError) as caught:
        with open_connection(_config(tmp_path / "unused.sqlite")):
            raise body_error

    assert caught.value is body_error
    assert events == ["rollback", "close"]


def test_standalone_connection_retains_thread_affinity(tmp_path):
    errors = []
    with open_connection(_config(tmp_path / "thread-affinity.sqlite")) as con:
        thread = threading.Thread(
            target=lambda: _capture_execute_error(con, errors)
        )
        thread.start()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(
        errors[0], connection_module.require_sqlite().ProgrammingError
    )


def test_connection_closes_after_normal_exit(tmp_path):
    with open_connection(_config(tmp_path / "normal-exit.sqlite")) as con:
        assert con.execute("SELECT 1").fetchone() == (1,)

    _assert_closed(con)


def test_initialization_failure_closes_connection(tmp_path, monkeypatch):
    opened = []
    original_connect = connection_module._connect
    failure = QueryIndexError("initialization failed")

    def capture_connection(*args, **kwargs):
        con = original_connect(*args, **kwargs)
        opened.append(con)
        return con

    def fail_initialization(*args, **kwargs):
        raise failure

    monkeypatch.setattr(connection_module, "_connect", capture_connection)
    monkeypatch.setattr(
        connection_module, "_initialize_connection", fail_initialization
    )

    with pytest.raises(QueryIndexError) as caught:
        with open_connection(_config(tmp_path / "initialization.sqlite")):
            pass

    assert caught.value is failure
    assert len(opened) == 1
    _assert_closed(opened[0])


def test_optional_sqlite_backend_failure_remains_explicit(
        tmp_path, monkeypatch):
    failure = QueryIndexUnavailable("sqlite unavailable")

    def unavailable():
        raise failure

    monkeypatch.setattr(connection_module, "require_sqlite", unavailable)

    with pytest.raises(QueryIndexUnavailable) as caught:
        with open_connection(_config(tmp_path / "unavailable.sqlite")):
            pass

    assert caught.value is failure
    assert not (tmp_path / "unavailable.sqlite").exists()


def _capture_execute_error(con, errors):
    try:
        con.execute("SELECT 1")
    except Exception as error:
        errors.append(error)
