"""Passive standard-library import coordination primitives.

This module deliberately holds no runtime generation, adapter plan, or session
state.  It exists below runtime initialization so a later framework interceptor
can coordinate imports without importing DRYML's optional framework packages.
"""

from __future__ import annotations

import importlib.abc
import sys
import threading
from collections import OrderedDict
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
                raise ImportEpochBusyError("import-busy: read-to-write import epoch upgrade is not allowed")
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

    @contextmanager
    def observation(self) -> Iterator[None]:
        """Serialize finder observations with a registration writer.

        Finder probes are not import readers, but their first-observation
        ledger must linearize with a writer reserving a root.  The mutex is
        released before delegated finder or loader code can run.
        """

        with self._condition:
            while self._writer is not None:
                self._condition.wait()
            yield

    @property
    def writer_owner(self) -> int | None:
        """Return the active writer owner while holding no token."""

        with self._condition:
            return self._writer

    @property
    def reader_count(self) -> int:
        """Return the current admission count for deterministic diagnostics."""

        with self._condition:
            return self._readers


class PassiveFrameworkFinder(importlib.abc.MetaPathFinder):
    """Record and delegate watched framework imports without runtime state.

    The finder deliberately knows only roots and observations.  It resolves the
    next finder itself so that a wrapped loader can retain the original spec and
    loader; the runtime is imported lazily only from loader callbacks.
    """

    observation_limit = 4096

    def __init__(self, coordinator: ImportEpochCoordinator) -> None:
        self.coordinator = coordinator
        self._roots: set[str] = set()
        # This is intentionally a fixed-size first-observation ledger, not an
        # ever-growing import history.  Keys bound top-level candidates while
        # values retain the exact first fullname for deterministic diagnostics.
        self._observed: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._registration_frozen = False

    def watch(self, root: str) -> None:
        """Register one top-level module root for passive observation."""

        self._validate_root(root)
        with self._lock:
            self._roots.add(root)

    def install_builtin_roots(self, roots: tuple[str, ...]) -> None:
        """Reserve built-in roots before this finder is installed.

        Built-in metadata is owned by this bootstrap module so base ``dryml``
        need not import ``dryml.runtime`` just to close the startup race.
        """

        with self.coordinator.writer():
            with self._lock:
                self._check_roots(roots, allow_existing=True)
                self._roots.update(roots)

    def roots(self) -> frozenset[str]:
        """Return the immutable current watched-root view."""

        with self._lock:
            return frozenset(self._roots)

    @property
    def observation_count(self) -> int:
        """Return the exact number of retained first observations."""

        with self._lock:
            return len(self._observed)

    @property
    def registration_frozen(self) -> bool:
        """Report whether the bounded ledger has closed registration."""

        with self._lock:
            return self._registration_frozen

    def can_register(self, roots: tuple[str, ...], *, allow_existing: bool = False) -> None:
        """Reject roots that cannot be registered linearly with observations."""

        with self._lock:
            self._check_roots(roots, allow_existing=allow_existing)

    def register(self, roots: tuple[str, ...]) -> None:
        """Reserve roots after ``can_register`` succeeds under a writer epoch."""

        with self._lock:
            self._check_roots(roots)
            self._roots.update(roots)

    def observed(self, root: str) -> bool:
        """Return whether *root* or one of its submodules was observed."""

        with self._lock:
            return any(self._overlaps(name, root) for name in self._observed.values())

    def first_observation(self, root: str) -> str | None:
        """Return the retained exact fullname for one observed root candidate."""

        with self._lock:
            for observed in self._observed.values():
                if self._overlaps(observed, root):
                    return observed
            return None

    def find_spec(self, fullname, path=None, target=None):
        """Record watched lookups and wrap the original delegated loader."""

        # Retain the exact first fullname for each candidate root.  This keeps
        # late descendant diagnostics exact without allowing a deep import
        # graph below one root to exhaust the bounded ledger.  It is deliberately
        # over before any delegated finder can populate ``sys.modules``.
        with self.coordinator.observation():
            with self._lock:
                candidate = fullname.partition(".")[0]
                if candidate not in self._observed and not self._registration_frozen:
                    self._observed[candidate] = fullname
                    if len(self._observed) == self.observation_limit:
                        self._registration_frozen = True
                if not any(self._matches(root, fullname) for root in self._roots):
                    return None

        # Preserve normal meta-path ordering while avoiding recursive entry into
        # this singleton finder.  No coordinator lock survives this delegation.
        try:
            index = sys.meta_path.index(self)
        except ValueError:
            return None
        for delegate in tuple(sys.meta_path[index + 1:]):
            find_spec = getattr(delegate, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
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
        if not root or any(not part.isidentifier() for part in root.split(".")):
            raise ValueError("watched framework root must be a dotted Python module name")

    def _check_roots(self, roots: tuple[str, ...], *, allow_existing: bool = False) -> None:
        if self._registration_frozen:
            raise RuntimeError("framework registration is frozen after 4096 observations")
        if not roots or len(set(roots)) != len(roots):
            raise ValueError("framework registration requires unique roots")
        for root in roots:
            self._validate_root(root)
            for name in self._observed.values():
                if self._overlaps(name, root):
                    raise RuntimeError(f"framework root {root!r} was already observed as {name!r}")
            if any(self._overlaps(name, root) for name in sys.modules):
                raise RuntimeError(f"framework root {root!r} was already loaded")
            if not allow_existing and any(self._overlaps(root, existing) for existing in self._roots):
                raise ValueError(f"framework root {root!r} overlaps an existing watched root")


class _DelegatingLoader:
    """PEP-451 loader wrapper that keeps reader lifetimes callback-bounded."""

    def __init__(self, fullname, spec, loader) -> None:
        self._fullname = fullname
        self._spec = spec
        self._loader = loader

    def create_module(self, spec):
        from_runtime = __import__("dryml.runtime.imports", fromlist=["create_module"])
        return from_runtime.create_module(self._fullname, self._spec, self._loader, spec)

    def exec_module(self, module):
        from_runtime = __import__("dryml.runtime.imports", fromlist=["exec_module"])
        return from_runtime.exec_module(self._fullname, self._spec, self._loader, module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


coordinator = ImportEpochCoordinator()
finder = PassiveFrameworkFinder(coordinator)

_BUILTIN_ROOTS = ("tensorflow", "torch", "jax", "jaxlib")


def install_passive_finder() -> PassiveFrameworkFinder:
    """Install the singleton passive finder exactly once without runtime effects."""

    if not any(item is finder for item in sys.meta_path):
        sys.meta_path.insert(0, finder)
    return finder


def install_builtin_roots() -> PassiveFrameworkFinder:
    """Install built-in root metadata before passive interception begins."""

    finder.install_builtin_roots(_BUILTIN_ROOTS)
    return finder


__all__ = [
    "ImportEpochBusyError",
    "ImportEpochCoordinator",
    "ImportEpochReentryError",
    "ImportEpochToken",
    "PassiveFrameworkFinder",
    "coordinator",
    "finder",
    "install_builtin_roots",
    "install_passive_finder",
]
