from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
import os
import shutil
import tempfile
from uuid import uuid4

from .locking import interprocess_lock, interprocess_read_lock
from ..utils.general import atomic_pickle_save, is_regular_file, pickle_load
from ..query.model import QueryIndexStatus, QueryIndexUnavailable, ReconcileReport, ValidationReport

class StoreAuthorityError(RuntimeError):
    """Raised when an authoritative Store definition is malformed or misplaced."""


class StoreAliasConflictError(StoreAuthorityError):
    """Raised when concurrent alias updates modify the same alias differently."""


_STATE_GENERATIONS_DIR = ".state-generations"
_STATE_POINTER_FILE = ".state-current.pkl"


class Store(ABC):
    """Abstract authoritative persistence backend for definitions and object state.

    Object roots are addressed by concrete-definition identity. New roots are
    published by atomic directory replacement; updates to existing roots publish
    immutable state generations through one atomic pointer so readers observe a
    complete old or new state. Per-root advisory reader leases defer reclamation
    until a supported reader has completed, retaining only the active generation
    after successful updates. Concrete backends own durability and commit timing.
    """

    @property
    def base_dir(self) -> str:
        """Base directory"""
        ...

    @property
    def object_root_dir(self) -> str:
        """Base directory"""
        ...

    @abstractmethod
    def has(self, cdef: "ConcreteDefinition") -> bool:
        """Lightweight membership test: do you have data for this cdef?"""
        ...

    @abstractmethod
    def hydrate_index(self) -> Iterable["ConcreteDefinition"]:
        """
        Full hydration: scan underlying storage and yield all cdefs
        that have data here.
        """
        ...

    @abstractmethod
    def _object_dir(self, cdef: "ConcreteDefinition") -> str:
        """
        Method to get the object directory for this cdef
        """

    def object_dir(self, cdef: "ConcreteDefinition") -> str:
        """Return the storage directory selected by a concrete definition.

        Args:
            cdef: ``ConcreteDefinition`` whose stable identity selects the path.

        Returns:
            The Store-local directory path. Calling this method does not create
            a directory or publish any authoritative state.
        """
        return self._object_dir(cdef)

    def _def_file(self, cdef: "ConcreteDefinition") -> str:
        return os.path.join(self.object_dir(cdef), "def.pkl")

    def save_object(self, obj: Object, *, revision: str|None = None) -> None:
        """Persist one object's complete state under its definition identity.

        New roots become discoverable only after staging and identity validation.
        Existing roots retain their authoritative ``def.pkl`` and publish a
        complete replacement state generation through an atomic pointer. A
        Store-scoped writer lease serializes supported saves of the same root;
        after publication it reclaims inactive generations only after readers
        using the matching reader lease have completed.

        Args:
            obj: Materialized object whose definition and state are persisted.
            revision: Optional object-state revision selected by the object.

        Raises:
            ValueError: If ``revision`` is not a string or ``None``.
            StoreAuthorityError: If a staged or existing root has a mismatched
                identity, malformed definition, or invalid state pointer.
            OSError: If staging or publication fails. The prior published state
                remains active when replacement publication fails.
        """

        if revision is not None:
            if not isinstance(revision, str):
                raise ValueError("revision must be a string or None at the Store.")
        obj_dir = self.object_dir(obj.definition)
        with interprocess_lock(self._state_lock_path(obj_dir)):
            self._save_object_locked(obj, obj_dir, revision=revision)

    def _save_object_locked(self, obj: Object, obj_dir: str, *, revision: str | None) -> None:
        """Stage and publish one object while its root's writer lease is held.

        Args:
            obj: Materialized object whose complete state will be persisted.
            obj_dir: Stable Store path selected by ``obj.definition``.
            revision: Optional object-state revision selected by the object.

        Raises:
            StoreAuthorityError: If authoritative or staged definitions disagree.
            OSError: If staging, publication, or inactive-generation reclamation
                fails. A published old or new complete state remains recoverable.
        """

        parent = os.path.dirname(obj_dir)
        os.makedirs(parent, exist_ok=True)
        existing_root = os.path.exists(obj_dir)
        previous_state_dir = None
        if existing_root:
            existing_def = self._validate_root_definition_path(os.path.join(obj_dir, "def.pkl"))
            if existing_def != obj.definition:
                raise StoreAuthorityError(
                    "Existing Store root identity does not match the object being saved."
                )
            previous_state_dir = self._active_state_dir(obj_dir)
        stage_dir = tempfile.mkdtemp(prefix=f".{obj.definition.stable_hash()}-", dir=parent)
        try:
            if existing_root:
                self._copy_active_state(obj_dir, stage_dir)
            # A root is undiscoverable until its complete staged definition has
            # been verified and the directory is atomically renamed.
            obj.save_state_to_dir(stage_dir, revision=revision)
            staged_def = self._read_concrete_definition(os.path.join(stage_dir, "def.pkl"))
            if staged_def != obj.definition:
                raise StoreAuthorityError("Staged object definition does not match the object identity.")
            dirty_token = self._mark_authority_dirty(obj.definition)
            try:
                if existing_root:
                    self._publish_existing_root(stage_dir, obj_dir)
                else:
                    os.replace(stage_dir, obj_dir)
                    stage_dir = None
            except BaseException:
                if self._publication_may_be_visible(
                        obj_dir,
                        existing_root=existing_root,
                        previous_state_dir=previous_state_dir):
                    stage_dir = None
                else:
                    self._discard_authority_dirty(dirty_token)
                raise
        finally:
            if stage_dir is not None:
                shutil.rmtree(stage_dir, ignore_errors=True)

    @classmethod
    def _publish_existing_root(cls, stage_dir: str, object_dir: str) -> None:
        """Publish a complete state generation and reclaim inactive predecessors.

        The caller must retain the root's writer lease. Pointer replacement makes
        the new complete generation active before reclamation; reader leases
        prevent deletion of a generation being restored by a supported reader.

        Args:
            stage_dir: Complete staged state directory to publish.
            object_dir: Existing authoritative object root.

        Raises:
            OSError: If generation movement, pointer publication, or reclamation
                fails. Before pointer publication the prior state remains active;
                afterwards the new state remains recoverable.
        """

        generation = uuid4().hex
        generations_dir = os.path.join(object_dir, _STATE_GENERATIONS_DIR)
        generation_dir = os.path.join(generations_dir, generation)
        os.makedirs(generations_dir, exist_ok=True)
        os.replace(stage_dir, generation_dir)
        try:
            atomic_pickle_save(generation, os.path.join(object_dir, _STATE_POINTER_FILE))
        except BaseException:
            try:
                generation_is_active = cls._active_state_dir(object_dir) == generation_dir
            except Exception:
                generation_is_active = True
            if not generation_is_active:
                shutil.rmtree(generation_dir, ignore_errors=True)
            raise
        cls._reclaim_inactive_state_generations(object_dir, generation)

    @staticmethod
    def _state_lock_path(object_dir: str) -> str:
        """Return the durable advisory lease path for one object root.

        Args:
            object_dir: Stable Store path for an object root.

        Returns:
            A sibling lock-file path. Keeping it outside the root lets writers
            coordinate initial root publication without making a root visible.
        """

        parent, name = os.path.split(object_dir)
        return os.path.join(parent, f".{name}.state.lock")

    @classmethod
    def _reclaim_inactive_state_generations(cls, object_dir: str, active_generation: str) -> None:
        """Delete inactive complete state directories while holding a writer lease.

        Args:
            object_dir: Existing authoritative object root.
            active_generation: Generation named by the current state pointer.

        Raises:
            OSError: If an inactive generation cannot be removed.

        Side Effects:
            Removes only directories below ``.state-generations`` other than the
            active generation. The caller's exclusive lease excludes supported
            readers that may still be restoring an older generation.
        """

        generations_dir = os.path.join(object_dir, _STATE_GENERATIONS_DIR)
        for entry in os.scandir(generations_dir):
            if entry.name != active_generation and entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)

    @classmethod
    def _publication_may_be_visible(
            cls,
            object_dir: str, *,
            existing_root: bool,
            previous_state_dir: str | None) -> bool:
        if not existing_root:
            return os.path.exists(object_dir)
        try:
            return cls._active_state_dir(object_dir) != previous_state_dir
        except Exception:
            return True

    @classmethod
    def _copy_active_state(cls, object_dir: str, stage_dir: str) -> None:
        source_dir = cls._active_state_dir(object_dir)
        for name in os.listdir(source_dir):
            if source_dir == object_dir and name in {_STATE_GENERATIONS_DIR, _STATE_POINTER_FILE}:
                continue
            source = os.path.join(source_dir, name)
            destination = os.path.join(stage_dir, name)
            if os.path.isdir(source):
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

    @staticmethod
    def _active_state_dir(object_dir: str) -> str:
        pointer_path = os.path.join(object_dir, _STATE_POINTER_FILE)
        if not os.path.exists(pointer_path):
            return object_dir
        generation = pickle_load(pointer_path)
        if (
                not isinstance(generation, str)
                or len(generation) != 32
                or any(char not in "0123456789abcdef" for char in generation)):
            raise StoreAuthorityError(f"Store state pointer is malformed: {pointer_path!r}.")
        state_dir = os.path.join(object_dir, _STATE_GENERATIONS_DIR, generation)
        if not os.path.isdir(state_dir):
            raise StoreAuthorityError(f"Store state generation is missing: {state_dir!r}.")
        return state_dir

    def _mark_authority_dirty(self, cdef: "ConcreteDefinition | None" = None):
        """Record an explicit authority mutation for buffered Store backends."""

    def _discard_authority_dirty(self, token) -> None:
        """Discard this operation's marker after publication definitively fails."""

    @staticmethod
    def _read_concrete_definition(path: str) -> "ConcreteDefinition":
        """Decode one regular persisted definition without class resolution."""

        if not is_regular_file(path):
            raise StoreAuthorityError(f"Store definition is not a regular file: {path!r}.")
        value = pickle_load(path)
        from ..definition import ConcreteDefinition

        if not isinstance(value, ConcreteDefinition):
            raise StoreAuthorityError(
                f"Store definition is {type(value).__name__}, not a ConcreteDefinition: {path!r}."
            )
        return value

    def _validate_root_definition_path(self, path: str) -> "ConcreteDefinition":
        """Decode and validate a complete object-root relative path and digest."""

        relative = os.path.relpath(path, self.object_root_dir)
        parts = relative.split(os.sep)
        if len(parts) != 3 or parts[2] != "def.pkl":
            raise StoreAuthorityError(f"Store root must be objects/<fanout>/<digest>/def.pkl: {path!r}.")
        fanout, digest, _ = parts
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise StoreAuthorityError(f"Store root has an invalid stable-hash digest: {path!r}.")
        if fanout != digest[:2]:
            raise StoreAuthorityError(f"Store root fanout does not match its digest: {path!r}.")
        cdef = self._read_concrete_definition(path)
        actual_digest = cdef.stable_hash()
        if actual_digest != digest:
            raise StoreAuthorityError(
                "Store def.pkl is stored under the wrong stable-hash directory. "
                f"path={path!r}, expected={digest!r}, actual={actual_digest!r}"
            )
        return cdef

    def restore_object(self, obj: Object, *, revision: str|None = None) -> None:
        """Restore an object's requested revision from the active state generation.

        Args:
            obj: Materialized object whose definition selects the Store root.
            revision: Optional revision forwarded to the object's state loader.

        Returns:
            ``None``. A missing root leaves ``obj`` unchanged.

        Raises:
            StoreAuthorityError: If the active state pointer or generation is
                malformed or missing.
            OSError: If persisted state cannot be read.

        Concurrency:
            Holds the root's shared reader lease from pointer resolution until
            object restoration completes. A cooperating writer waits before
            replacing the pointer and reclaiming an inactive generation, so this
            call observes one complete old or new state rather than a mixture.
        """

        cdef = obj.definition
        def_path = self._def_file(cdef)
        if not os.path.exists(def_path):
            return

        obj_dir = self.object_dir(cdef)
        with interprocess_read_lock(self._state_lock_path(obj_dir)):
            obj.restore_state_from_dir(self._active_state_dir(obj_dir), revision=revision)

    def read_definition(self, cdef: "ConcreteDefinition") -> "ConcreteDefinition | None":
        """Read and validate the authoritative root matching ``cdef``.

        Args:
            cdef: Expected concrete definition and storage identity.

        Returns:
            The persisted definition, or ``None`` when the root is absent or a
            different valid definition occupies the same stable-hash path.

        Raises:
            StoreAuthorityError: If the root path, digest, or payload type is
                malformed or inconsistent.
        """

        def_path = self._def_file(cdef)
        if not os.path.exists(def_path):
            return None
        stored = self._validate_root_definition_path(def_path)
        if stored != cdef:
            return None
        return stored

    def read_main_def(self) -> "ConcreteDefinition" | None:
        """Read the Store's main-definition reference without changing it.

        Returns:
            The stored ``ConcreteDefinition``, or ``None`` if no main reference
            has been published.

        Raises:
            StoreAuthorityError: If the persisted reference is not a complete
                ``ConcreteDefinition`` payload.
        """
        return None

    def write_main_def(self, main_def: "ConcreteDefinition") -> None:
        """Atomically publish a validated main-definition reference.

        Args:
            main_def: ``ConcreteDefinition`` to make the Store's main reference.

        Raises:
            StoreAuthorityError: If ``main_def`` is not a concrete definition.
            OSError: If publication fails; an existing reference remains
                authoritative until replacement succeeds.

        Side Effects:
            Concrete Stores may update a buffered reference cache and mark a
            deferred archive or other persistence target dirty.
        """
        pass

    def set_main_def(self, main_def: "ConcreteDefinition") -> None:
        """Stage a validated main-definition reference for a later commit.

        Args:
            main_def: ``ConcreteDefinition`` to use as the main reference.

        Raises:
            StoreAuthorityError: If ``main_def`` is not a concrete definition.

        Side Effects:
            Buffered Stores cache the reference and mark it for explicit
            publication; this method itself need not publish bytes.
        """
        pass

    def read_aliases(self) -> dict[str, "ConcreteDefinition"]:
        """Read validated named object references without publishing changes.

        Returns:
            A new mapping from non-empty string aliases to concrete definitions,
            or an empty mapping when no aliases are stored or supported.

        Raises:
            StoreAuthorityError: If the persisted mapping, an alias key, or a
                definition payload is malformed.
        """
        return {}

    def write_aliases(self, aliases: dict[str, "ConcreteDefinition"]) -> dict[str, "ConcreteDefinition"]:
        """Validate and publish named concrete-definition references.

        Args:
            aliases: Complete mapping whose keys are non-empty strings and whose
                values are ``ConcreteDefinition`` instances.

        Returns:
            The mapping published by the Store. Stores without alias support may
            return the supplied mapping without persistence.

        Raises:
            StoreAuthorityError: If any key or payload is malformed.
            OSError: If replacement publication fails; prior alias bytes remain
                authoritative.

        Side Effects:
            Implementations may atomically replace reference bytes or mark a
            buffered archive dirty for a later commit.
        """
        pass

    @staticmethod
    def _validate_main_definition(main_def) -> None:
        Store._validate_reference_definition(main_def, "main definition")

    @staticmethod
    def _validate_reference_definition(cdef, reference: str) -> None:
        from ..definition import ConcreteDefinition

        if not isinstance(cdef, ConcreteDefinition):
            raise StoreAuthorityError(
                f"Store {reference} is {type(cdef).__name__}, not a ConcreteDefinition."
            )

    @classmethod
    def _validate_aliases(cls, aliases) -> None:
        if not isinstance(aliases, dict):
            raise StoreAuthorityError("Store aliases payload is not a dictionary.")
        for alias, cdef in aliases.items():
            if not isinstance(alias, str):
                raise StoreAuthorityError("Store aliases contain a non-string alias.")
            if alias == "":
                raise StoreAuthorityError("Store aliases contain an empty alias.")
            cls._validate_reference_definition(cdef, f"alias {alias!r}")

    def commit(self) -> None:
        """Optional; useful for zips, S3, HDF5, etc."""
        ...

    def catalog_key(self) -> str:
        """Stable logical identity for query-catalog replica deduplication."""
        try:
            base_dir = getattr(self, "base_dir", None)
        except Exception:
            base_dir = None
        if base_dir is not None and base_dir is not Ellipsis:
            return f"{type(self).__module__}.{type(self).__qualname__}:{os.path.abspath(os.fspath(base_dir))}"
        return f"{type(self).__module__}.{type(self).__qualname__}:id:{id(self)}"

    def open_query_index(self):
        """Return this Store's optional query index, or None for memory/no index modes."""
        return None

    def query_index_status(self) -> QueryIndexStatus:
        """Return backend-neutral status for this Store's own query index.

        Stores that do not own a persistent query index report a disabled index.
        Concrete Store implementations may return richer backend-specific status.
        """
        return QueryIndexStatus(
            backend="none",
            store_key=self.catalog_key(),
            generation=None,
            schema_version=None,
            semantic_versions={},
            state="disabled",
        )

    def rebuild_query_index(self) -> ReconcileReport:
        """Rebuild this Store's query index from authoritative object state.

        The base Store has no persistent query index, so callers receive a clear
        unavailability error instead of a silent no-op.
        """
        raise QueryIndexUnavailable(f"Store {self!r} does not own a rebuildable query index.")

    def reconcile_query_index(self) -> ReconcileReport:
        """Repair this Store's query index against authoritative object state."""
        return self.rebuild_query_index()

    def validate_query_index(self, *, thorough: bool = False) -> ValidationReport:
        """Validate this Store's query index without exposing backend internals."""
        return ValidationReport("none", self.catalog_key(), True)

    def close(self) -> None:
        """Cleanup (temp dirs, handles, etc.)"""
        ...
