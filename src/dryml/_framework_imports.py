"""Passive standard-library import coordination primitives.

This module deliberately holds no runtime generation, adapter plan, or session
state.  It exists below runtime initialization so a later framework interceptor
can coordinate imports without importing DRYML's optional framework packages.
"""

from __future__ import annotations

import importlib.abc
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


class ImportEpochBusyError(RuntimeError):
    """Raised when a non-upgradable import epoch cannot be admitted."""


class ImportEpochReentryError(RuntimeError):
    """Raised when a writer owner tries to re-enter the coordinator."""


@dataclass(frozen=True, slots=True)
class ImportEpochToken:
    """Opaque reader or writer admission token."""

    kind: str
    owner: int


class ImportEpochCoordinator:
    """Coordinate overlapping import readers with short non-waiting writers."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer: int | None = None
        self._reader_owners: dict[int, int] = {}

    @contextmanager
    def reader(self) -> Iterator[ImportEpochToken]:
        """Admit a reader, waiting only while a different writer is active."""

        owner = threading.get_ident()
        with self._condition:
            if self._writer == owner:
                raise ImportEpochReentryError("transition writer cannot enter import reader admission")
            while self._writer is not None:
                self._condition.wait()
            self._readers += 1
            self._reader_owners[owner] = self._reader_owners.get(owner, 0) + 1
        token = ImportEpochToken("reader", owner)
        try:
            yield token
        finally:
            with self._condition:
                self._readers -= 1
                count = self._reader_owners[owner] - 1
                if count:
                    self._reader_owners[owner] = count
                else:
                    del self._reader_owners[owner]
                self._condition.notify_all()

    @contextmanager
    def writer(self) -> Iterator[ImportEpochToken]:
        """Admit an exclusive writer or fail immediately for any active epoch."""

        owner = threading.get_ident()
        with self._condition:
            if self._writer == owner:
                raise ImportEpochReentryError("transition writer re-entry is not allowed")
            if self._reader_owners.get(owner):
                raise ImportEpochBusyError("read-to-write import epoch upgrade is not allowed")
            if self._writer is not None or self._readers:
                raise ImportEpochBusyError("import-busy: active reader or writer prevents transition")
            self._writer = owner
        token = ImportEpochToken("writer", owner)
        try:
            yield token
        finally:
            with self._condition:
                if self._writer == owner:
                    self._writer = None
                    self._condition.notify_all()

    @property
    def writer_owner(self) -> int | None:
        """Return the active writer owner while holding no token."""

        with self._condition:
            return self._writer


class PassiveFrameworkFinder(importlib.abc.MetaPathFinder):
    """Record watched-root observations without changing normal import lookup."""

    def __init__(self, coordinator: ImportEpochCoordinator) -> None:
        self.coordinator = coordinator
        self._roots: set[str] = set()
        self._observed: set[str] = set()
        self._lock = threading.Lock()

    def watch(self, root: str) -> None:
        """Register one top-level module root for passive observation."""

        if not root or "." in root:
            raise ValueError("watched framework root must be one top-level module name")
        with self._lock:
            self._roots.add(root)

    def observed(self, root: str) -> bool:
        """Return whether *root* or one of its submodules was observed."""

        with self._lock:
            return any(name == root or name.startswith(root + ".") for name in self._observed)

    def find_spec(self, fullname, path=None, target=None):
        """Passively record a watched lookup and delegate to the next finder."""

        root = fullname.partition(".")[0]
        with self._lock:
            if root in self._roots:
                self._observed.add(fullname)
        return None


coordinator = ImportEpochCoordinator()
finder = PassiveFrameworkFinder(coordinator)


def install_passive_finder() -> PassiveFrameworkFinder:
    """Install the singleton passive finder exactly once without runtime effects."""

    if not any(item is finder for item in sys.meta_path):
        sys.meta_path.insert(0, finder)
    return finder


__all__ = [
    "ImportEpochBusyError",
    "ImportEpochCoordinator",
    "ImportEpochReentryError",
    "ImportEpochToken",
    "PassiveFrameworkFinder",
    "coordinator",
    "finder",
    "install_passive_finder",
]
