from __future__ import annotations


def wal_runtime_is_known_safe(version: tuple[int, int, int]) -> bool:
    major, minor, patch = version
    if (major, minor, patch) >= (3, 51, 3):
        return True
    if (major, minor) == (3, 50) and patch >= 7:
        return True
    if (major, minor) == (3, 44) and patch >= 6:
        return True
    return False


def is_sqlite_busy_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name not in {"OperationalError", "DatabaseError"}:
        return False
    code_name = getattr(exc, "sqlite_errorname", "")
    if code_name in {"SQLITE_BUSY", "SQLITE_BUSY_SNAPSHOT", "SQLITE_LOCKED", "SQLITE_LOCKED_SHAREDCACHE"}:
        return True
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message or "database is busy" in message
