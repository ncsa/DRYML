"""Passive PEP 451 interception primitives for watched framework roots.

This module is deliberately dependency-light: importing :mod:`dryml` installs
the finder and reserves built-in names without importing the runtime package or
an optional framework.  Loader callbacks import the runtime half only after a
watched module's original loader has been selected.
"""

from __future__ import annotations

import importlib.abc
import os
import sys
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


class ImportEpochBusyError(RuntimeError):
    """Raised when registration cannot safely overlap an import callback."""


class ImportEpochReentryError(RuntimeError):
    """Raised when an import callback attempts an unsupported writer reentry."""


@dataclass(frozen=True, slots=True)
class ImportEpochToken:
    """Opaque reader or writer admission token for one coordinator epoch."""

    kind: str
    owner: int


class ImportEpochCoordinator:
    """Coordinate registration writers with watched-loader readers.

    The coordinator is intentionally not a general import lock.  Reader to
    writer upgrades fail rather than waiting, which prevents normal callback
    reentry from deadlocking a process-global publication transition.
    """

    def __init__(self) -> None:
        """Initialize empty reader and writer ownership state."""
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer: int | None = None
        self._owners: dict[int, int] = {}

    @contextmanager
    def reader(self) -> Iterator[ImportEpochToken]:
        """Admit one loader callback and yield its immutable ownership token."""
        owner = threading.get_ident()
        with self._condition:
            if self._writer == owner:
                raise ImportEpochReentryError("import writer cannot reenter a framework reader")
            while self._writer is not None:
                self._condition.wait()
            self._readers += 1
            self._owners[owner] = self._owners.get(owner, 0) + 1
        try:
            yield ImportEpochToken("reader", owner)
        finally:
            with self._condition:
                self._readers -= 1
                count = self._owners[owner] - 1
                if count:
                    self._owners[owner] = count
                else:
                    del self._owners[owner]
                self._condition.notify_all()

    @contextmanager
    def writer(self) -> Iterator[ImportEpochToken]:
        """Admit a registration writer or fail boundedly while imports overlap."""
        owner = threading.get_ident()
        with self._condition:
            if self._writer == owner:
                raise ImportEpochReentryError("framework registration writer reentry is not allowed")
            if self._owners.get(owner):
                raise ImportEpochBusyError("import-busy: reader-to-writer upgrade is not allowed")
            if self._writer is not None or self._readers:
                raise ImportEpochBusyError("import-busy: active framework import prevents registration")
            self._writer = owner
        try:
            yield ImportEpochToken("writer", owner)
        finally:
            with self._condition:
                self._writer = None
                self._condition.notify_all()

    @contextmanager
    def observation(self) -> Iterator[None]:
        """Serialize a finder observation with a registration writer."""
        with self._condition:
            while self._writer is not None:
                self._condition.wait()
            yield

    @property
    def reader_count(self) -> int:
        """Return the number of active watched-loader readers."""
        with self._condition:
            return self._readers


class PassiveFrameworkFinder(importlib.abc.MetaPathFinder):
    """Record first observations and wrap delegated loaders for watched roots."""

    observation_limit = 4096

    def __init__(self, coordinator: ImportEpochCoordinator) -> None:
        """Create a finder using *coordinator* to linearize registry changes."""
        self.coordinator = coordinator
        self._roots: set[str] = set()
        self._observed: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._registration_frozen = False
        self._observation_callback = None

    def _check_pid(self) -> None:
        """Reject inherited finder state before acquiring its local lock."""
        if os.getpid() != self._pid:
            raise RuntimeError("framework import state was inherited after fork; use spawn or a fresh interpreter")

    def install_builtin_roots(self, roots: tuple[str, ...]) -> None:
        """Reserve built-in roots before the finder is installed on ``meta_path``."""
        self._check_pid()
        with self.coordinator.writer():
            with self._lock:
                self._check_roots(roots, allow_existing=True)
                self._roots.update(roots)

    def roots(self) -> frozenset[str]:
        """Return the immutable watched-root snapshot."""
        self._check_pid()
        with self._lock:
            return frozenset(self._roots)

    @property
    def registration_frozen(self) -> bool:
        """Return whether the bounded observation ledger closed registration."""
        self._check_pid()
        with self._lock:
            return self._registration_frozen

    def can_register(self, roots: tuple[str, ...], *, allow_existing: bool = False) -> None:
        """Validate roots without mutating the watched-root set."""
        self._check_pid()
        with self._lock:
            self._check_roots(roots, allow_existing=allow_existing)

    def register(self, roots: tuple[str, ...]) -> None:
        """Reserve validated roots while the caller owns the writer epoch."""
        self._check_pid()
        with self._lock:
            self._check_roots(roots)
            self._roots.update(roots)

    def observed(self, root: str) -> bool:
        """Return whether *root* overlaps a retained first observation."""
        self._check_pid()
        with self._lock:
            return any(self._overlaps(root, name) for name in self._observed.values())

    def set_observation_callback(self, callback) -> None:
        """Install the registry freeze callback invoked for watched roots.

        Args:
            callback: Dependency-light callable receiving an observed fullname.

        The callback runs after finder locks are released while the observation
        epoch remains held, so registration cannot race the freeze boundary.
        """
        self._check_pid()
        with self._lock:
            self._observation_callback = callback
            observed = tuple(
                fullname
                for fullname in self._observed.values()
                if any(self._matches(root, fullname) for root in self._roots)
            )
        for fullname in observed:
            callback(fullname)

    def find_spec(self, fullname, path=None, target=None):
        """Record *fullname* and return a wrapper around its delegated spec loader."""
        self._check_pid()
        with self.coordinator.observation():
            with self._lock:
                root = fullname.partition(".")[0]
                if root not in self._observed and not self._registration_frozen:
                    self._observed[root] = fullname
                    self._registration_frozen = len(self._observed) >= self.observation_limit
                watched = any(self._matches(root_name, fullname) for root_name in self._roots)
                callback = self._observation_callback
            if watched and callback is not None:
                callback(fullname)
        if not watched:
            return None
        try:
            index = sys.meta_path.index(self)
        except ValueError:
            return None
        for delegate in tuple(sys.meta_path[index + 1:]):
            method = getattr(delegate, "find_spec", None)
            if method is None:
                continue
            spec = method(fullname, path, target)
            if spec is not None:
                if spec.loader is not None:
                    spec.loader = _DelegatingLoader(fullname, spec, spec.loader)
                return spec
        return None

    @staticmethod
    def _matches(root: str, fullname: str) -> bool:
        return fullname == root or fullname.startswith(root + ".")

    @staticmethod
    def _overlaps(left: str, right: str) -> bool:
        return left == right or left.startswith(right + ".") or right.startswith(left + ".")

    @staticmethod
    def _validate_root(root: str) -> None:
        if not isinstance(root, str) or not root or any(not part.isidentifier() for part in root.split(".")):
            raise ValueError("watched framework root must be a dotted Python module name")

    def _check_roots(self, roots: tuple[str, ...], *, allow_existing: bool = False) -> None:
        if self._registration_frozen:
            raise RuntimeError("framework registration is frozen after observation-ledger overflow")
        if not roots or len(set(roots)) != len(roots):
            raise ValueError("framework registration requires non-empty unique roots")
        for root in roots:
            self._validate_root(root)
            existing = allow_existing and root in self._roots
            if not existing and any(self._overlaps(root, name) for name in self._observed.values()):
                raise RuntimeError(f"framework root {root!r} was already observed")
            if not existing and any(self._overlaps(root, name) for name in sys.modules):
                raise RuntimeError(f"framework root {root!r} was already loaded")
            if not existing and any(self._overlaps(root, other) for other in self._roots):
                raise ValueError(f"framework root {root!r} overlaps an existing watched root")


class _DelegatingLoader:
    """PEP 451 loader wrapper retaining the original loader and specification."""

    def __init__(self, fullname, spec, loader) -> None:
        """Retain delegated loader facts for one intercepted module specification."""
        self._fullname = fullname
        self._spec = spec
        self._loader = loader

    def create_module(self, spec):
        """Delegate module creation through the lazy runtime lifecycle."""
        from dryml.runtime.imports import create_module
        return create_module(self._fullname, self._spec, self._loader, spec)

    def exec_module(self, module):
        """Delegate module execution through the lazy runtime lifecycle."""
        from dryml.runtime.imports import exec_module
        return exec_module(self._fullname, self._spec, self._loader, module)

    def __getattr__(self, name):
        """Expose optional original loader protocols unchanged."""
        return getattr(self._loader, name)


coordinator = ImportEpochCoordinator()
finder = PassiveFrameworkFinder(coordinator)
_BUILTIN_ROOTS = ("tensorflow", "torch", "jax", "jaxlib")


def install_passive_finder() -> PassiveFrameworkFinder:
    """Install the singleton finder once and return it without importing runtime."""
    if not any(item is finder for item in sys.meta_path):
        sys.meta_path.insert(0, finder)
    return finder


def install_builtin_roots() -> PassiveFrameworkFinder:
    """Reserve built-in watched roots before enabling passive interception."""
    finder.install_builtin_roots(_BUILTIN_ROOTS)
    return finder


__all__ = ["ImportEpochBusyError", "ImportEpochCoordinator", "ImportEpochReentryError", "ImportEpochToken", "PassiveFrameworkFinder", "coordinator", "finder", "install_builtin_roots", "install_passive_finder"]
