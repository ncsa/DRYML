"""Concrete resumable Dataset Artifact backed by logical cache codecs."""

from __future__ import annotations

import json
import os
import shutil
import stat
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from dryml.core import AutoRef, ConcreteDefinition, Definition, ObjectRef, Ref, StateRef
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import SpecTree
from dryml.core.utils.graph.path import GraphPath
from dryml.data import Dataset, DatasetCursor
from dryml.locking import FileLock, LockError
from dryml.managed import ManagedContext, managed_operation

from ._cache_model import (
    CacheIntegrityError, _FORMAT, _VERSION, flatten_spec, normalize_value,
    numpy_spec, rebuild_value, spec_from_data, spec_to_data,
)
from ._cache_numpy import normalize_leaf, read_chunk, validate_chunk_descriptor, write_chunk
from . import _cache_parquet
from .base import Artifact
from .value import ArtifactNotReadyError


T = TypeVar("T")
CacheCodec: TypeAlias = Literal["numpy", "parquet", "netcdf"]
_MANIFEST_FILENAME = "cache-manifest.json"
_DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
_MAX_YIELDS_PER_CHUNK = 65_536
_WORK_MARKER = "cache-work.json"
_WORK_FORMAT = "dryml.cached-dataset-work"


@dataclass(slots=True)
class _CacheGeneration:
    """One private immutable completed payload and its active iterator leases."""

    payload: dict[str, Any]
    leases: int = 0
    retired: bool = False


class CachedDataset(Artifact, Dataset[T], Generic[T]):
    """Persist a finite referenced Dataset as re-iterable cache content.

    Args:
        src: A ``Ref[AutoRef]`` source binding. Construction retains the reference
            without traversal, source materialization, or implicit publication.

    ``compute`` admits only finite sources and writes self-contained completed
    state. Intermediate checkpoints contain only progress metadata and a private
    Store work token; completed reads validate metadata eagerly and chunks lazily.
    """

    state_codec = "cachedataset1"

    def __init__(self, src: Ref[AutoRef]) -> None:
        """Retain one non-materializing Dataset source reference.

        Args:
            src: Existing CDef, ObjectRef, StateRef, or live Dataset accepted by
                DRYML's ``Ref[AutoRef]`` binding rules.

        Raises:
            TypeError: If DRYML cannot represent ``src`` at the reference boundary.
        """

        self.src = src

    @property
    def ready(self) -> bool:
        """Return whether a complete cache payload is installed without I/O."""

        return self._read_generation() is not None

    @property
    def spec(self) -> SpecTree:
        """Return completed codec output spec, or raise before computation."""

        if not self.ready:
            raise ArtifactNotReadyError("CachedDataset is not ready.")
        return self._read_generation().payload["spec"]

    @property
    def processed_count(self) -> int:
        """Return the sealed progress count, including completed cache content."""

        generation = self._read_generation()
        if generation is not None:
            return int(generation.payload["count"])
        return int(getattr(self, "_cache_payload", {}).get("count", 0))

    def __len__(self) -> int:
        """Return exact cached yield cardinality once completed."""

        if not self.ready:
            raise ArtifactNotReadyError("CachedDataset is not ready.")
        return self._read_generation().payload["count"]

    def __iter__(self) -> Iterator[T]:
        """Create a fresh lazy iterator over completed cache chunks.

        Raises:
            ArtifactNotReadyError: If no complete payload is installed.
            CacheIntegrityError: If a chunk fails validation before it yields.
        """

        return self.iterator()

    def iterator(self) -> DatasetCursor[T]:
        """Create an independent cursor that owns one cache-content traversal."""

        if not self.ready:
            raise ArtifactNotReadyError("CachedDataset is not ready.")
        generation = self._read_generation()
        assert generation is not None
        with self._generation_guard():
            generation.leases += 1
        return DatasetCursor(self._iter_generation(generation))

    def peek(self) -> T:
        """Return one cached yield and always close the temporary reader."""

        iterator = self.iterator()
        try:
            return next(iterator)
        except StopIteration as error:
            raise ValueError("Cannot peek an empty cached Dataset.") from error
        finally:
            iterator.close()

    @managed_operation(resumable=True, return_state_ref=True, store_parameter="store")
    def compute(
            self, *, codec: CacheCodec, store=None, target_chunk_bytes: int | None = None,
            managed: ManagedContext) -> None:
        """Compute or republish one selected cache codec through managed lifecycle.

        Args:
            codec: Built-in physical codec, currently ``"numpy"`` or optional
                ``"parquet"``.
            store: Optional managed publication Store override.
            target_chunk_bytes: Positive soft target for retained logical chunks.
            managed: Runtime-created lifecycle authority.

        Raises:
            ValueError: If arguments, source cardinality, source values, or source
                specification violate the cache contract.
            TypeError: If the source is not a Dataset or leaves are unsupported.
            CacheIntegrityError: If retained progress or payload bytes are invalid.

        Side Effects:
            Materializes the source only when a new codec computation is needed,
            creates metadata-only checkpoints, and installs complete content only
            after exact source exhaustion succeeds.
        """

        if codec not in {"numpy", "parquet", "netcdf"}:
            raise ValueError("CachedDataset codec must be one of numpy, parquet, or netcdf.")
        if codec == "netcdf":
            raise ValueError(f"CachedDataset codec {codec!r} is not implemented in this build.")
        if codec == "parquet":
            # Dependency admission intentionally precedes source loading, including
            # same-codec republication that otherwise skips the source.
            _cache_parquet.load_pyarrow()
        target = _validate_target(target_chunk_bytes)
        current = getattr(self, "_cache_payload", None)
        if current is not None and current.get("mode") == "completed" and current.get("codec") == codec:
            return

        # A saved prefix is authenticated before source materialization.  A broken
        # work token must not silently restart a stochastic or one-shot source.
        resumed = None
        if managed.is_resuming and current is not None and current.get("mode") == "working":
            resumed = self._resume_or_start_work(
                managed, codec, current["spec"], current["expected"],
            )
            if resumed[2] == current["expected"]:
                # Full progress is checkpointed only after EOF was observed below.
                # A resumed one-shot source consequently need not be reopened.
                workdir, chunks, _ = resumed
                self._cache_payload = {
                    "mode": "completed", "codec": codec, "spec": current["spec"],
                    "count": current["expected"], "chunks": chunks,
                    "source_dir": _work_data_directory(workdir), "_workdir": workdir,
                }
                return
        source = self._load_source(managed)
        source_spec = source.spec
        leaves = flatten_spec(source_spec)
        output_spec = numpy_spec(source_spec)
        expected = _finite_cardinality(source)
        if resumed is None:
            workdir, chunks, count = self._resume_or_start_work(
                managed, codec, output_spec, expected,
            )
        else:
            workdir, chunks, count = resumed
            if expected != current["expected"] or spec_to_data(output_spec) != spec_to_data(current["spec"]):
                raise ValueError("CachedDataset checkpoint source is incompatible with this invocation.")
        cursor = source.iterator()
        try:
            if count:
                cursor.skip(count)
            if not leaves:
                chunk_index = len(chunks)
                pending = 0
                while count < expected:
                    try:
                        item = next(cursor)
                    except StopIteration as error:
                        raise ValueError("CachedDataset source exhausted before its declared cardinality.") from error
                    normalize_value(item, source_spec)
                    count += 1
                    pending += 1
                    if pending == _MAX_YIELDS_PER_CHUNK:
                        chunks.append(self._flush_chunk(workdir, chunk_index, (tuple() for _ in range(pending)), target, codec))
                        chunk_index += 1
                        pending = 0
                        if count < expected:
                            self._install_working(codec, output_spec, expected, count, workdir, chunks, managed)
                            managed.checkpoint()
                if pending:
                    chunks.append(self._flush_chunk(workdir, chunk_index, (tuple() for _ in range(pending)), target, codec))
                try:
                    next(cursor)
                except StopIteration:
                    pass
                else:
                    raise ValueError("CachedDataset source yielded more than its declared cardinality.")
                self._install_working(codec, output_spec, expected, count, workdir, chunks, managed)
                managed.checkpoint()
                self._cache_payload = {
                    "mode": "completed", "codec": codec, "spec": output_spec,
                    "count": expected, "chunks": chunks, "source_dir": _work_data_directory(workdir),
                    "_workdir": workdir,
                }
                return
            buffered: list[tuple[Any, ...]] = []
            buffered_bytes = 0
            chunk_index = len(chunks)
            while count < expected:
                try:
                    item = next(cursor)
                except StopIteration as error:
                    raise ValueError("CachedDataset source exhausted before its declared cardinality.") from error
                _, raw_leaves = normalize_value(item, source_spec)
                normalized = tuple(normalize_leaf(value, spec) for value, spec in zip(raw_leaves, leaves))
                size = sum(value.nbytes for value in normalized)
                if buffered and (buffered_bytes + size > target or len(buffered) == _MAX_YIELDS_PER_CHUNK):
                    chunks.append(self._flush_chunk(workdir, chunk_index, buffered, target, codec))
                    chunk_index += 1
                    buffered = []
                    buffered_bytes = 0
                    if count < expected:
                        self._install_working(codec, output_spec, expected, count, workdir, chunks, managed)
                        managed.checkpoint()
                buffered.append(normalized)
                buffered_bytes += size
                count += 1
            if buffered:
                chunks.append(self._flush_chunk(workdir, chunk_index, buffered, target, codec))
            try:
                next(cursor)
            except StopIteration:
                pass
            else:
                raise ValueError("CachedDataset source yielded more than its declared cardinality.")
        finally:
            cursor.close()
        self._install_working(codec, output_spec, expected, count, workdir, chunks, managed)
        managed.checkpoint()
        self._cache_payload = {
            "mode": "completed", "codec": codec, "spec": output_spec,
            "count": expected, "chunks": chunks, "source_dir": _work_data_directory(workdir),
            "_workdir": workdir,
        }

    def _load_source(self, managed: ManagedContext) -> Dataset:
        """Materialize the inert source using this invocation's selected Repo."""

        reference = self.src
        repo = managed.state_repo
        if isinstance(reference, StateRef):
            source = repo.load_state_ref(reference)
        elif isinstance(reference, ObjectRef):
            source = repo.build_object_ref(reference)
        elif isinstance(reference, ConcreteDefinition):
            source = repo._load_structural(reference, require_store=False)
        elif isinstance(reference, Definition):
            source = repo._load_structural(reference.concretize(repo=repo), require_store=False)
        else:
            raise TypeError("CachedDataset source reference is not a supported DRYML reference.")
        if not isinstance(source, Dataset):
            raise TypeError("CachedDataset source reference must materialize as a Dataset.")
        return source

    def _resume_or_start_work(self, managed, codec, spec, expected):
        """Return validated compatible progress or allocate a private new work directory."""

        payload = getattr(self, "_cache_payload", None)
        if managed.is_resuming and payload is not None and payload.get("mode") == "working":
            if payload.get("codec") != codec or payload.get("expected") != expected:
                raise ValueError("CachedDataset checkpoint is incompatible with this invocation.")
            workdir = _work_directory(managed.state_repo, payload.get("work_token"))
            _validate_work_marker(workdir, payload, managed, self.object_ref.digest())
            chunks = payload.get("chunks")
            if not isinstance(chunks, list):
                raise CacheIntegrityError("CachedDataset working chunk inventory is malformed.")
            for chunk in chunks:
                _read_chunk(
                    codec, os.path.join(_work_data_directory(workdir), "chunks", chunk["file"]),
                    chunk, flatten_spec(spec),
                )
            _discard_unassociated_tail(workdir, chunks, codec)
            return workdir, chunks, payload["count"]
        if payload is not None and payload.get("mode") == "working":
            # Managed has already fenced this fresh rerun with a new attempt.  Do
            # not reuse or delete the old prefix, but make the now-superseded
            # allocation eligible for a later bounded Store-local sweep.
            old_workdir = _work_directory(managed.state_repo, payload.get("work_token"))
            _validate_stored_work_marker(old_workdir, payload)
            _mark_work_reclaimable(managed.state_repo, old_workdir)
        workdir = _create_work_directory(managed.state_repo, self)
        _write_work_marker(workdir, managed, self.object_ref.digest())
        return workdir, [], 0

    def _install_working(self, codec, spec, expected, count, workdir, chunks, managed) -> None:
        """Atomically replace live metadata with a checkpointable working prefix."""

        current = getattr(self, "_cache_payload", None)
        prior_ready_state_digest = None
        if isinstance(current, dict):
            prior_ready_state_digest = current.get("prior_ready_state_digest")
            if current.get("mode") == "completed":
                prior = self.last_state_ref
                prior_ready_state_digest = None if prior is None else prior.digest()
        if prior_ready_state_digest is not None and not _is_digest(prior_ready_state_digest):
            raise CacheIntegrityError("CachedDataset prior ready reference is invalid.")
        self._cache_payload = {
            "mode": "working", "codec": codec, "spec": spec, "expected": expected,
            "count": count, "work_token": _work_token(managed.state_repo, workdir), "chunks": list(chunks),
            "object_ref_digest": self.object_ref.digest(), "operation_id": managed.operation_id,
            "attempt_id": managed.attempt_id,
        }
        if prior_ready_state_digest is not None:
            # StateRef digests name complete immutable StateRef authority through
            # the selected Repo.  Persisting the digest avoids embedding a CDef,
            # source payload, or recursive working snapshot in this manifest.
            self._cache_payload["prior_ready_state_digest"] = prior_ready_state_digest

    @staticmethod
    def _flush_chunk(workdir, index, values, target, codec):
        """Encode one sealed logical chunk into the retained working directory."""

        suffix = "npz" if codec == "numpy" else "parquet"
        writer = write_chunk if codec == "numpy" else _cache_parquet.write_chunk
        return writer(
            os.path.join(_work_data_directory(workdir), "chunks", f"chunk-{index:08d}.{suffix}"), values,
            segment_bytes=min(target, 8 * 1024 * 1024),
        )

    def _iter_payload(self, payload):
        """Yield fully verified logical chunks from one immutable completed payload."""

        source_dir = payload.get("source_dir")
        if not isinstance(source_dir, str):
            raise CacheIntegrityError("CachedDataset completed payload has no local source directory.")
        leaves = flatten_spec(payload["spec"])
        for descriptor in payload["chunks"]:
            path = os.path.join(source_dir, "chunks", descriptor["file"])
            for values in _read_chunk(payload["codec"], path, descriptor, leaves):
                yield rebuild_value(payload["spec"], values)

    def _generation_guard(self):
        """Return the private lock that serializes generation replacement and leases."""

        guard = getattr(self, "_cache_generation_guard", None)
        if guard is None:
            guard = threading.RLock()
            self._cache_generation_guard = guard
        return guard

    def _read_generation(self) -> _CacheGeneration | None:
        """Return the installed completed generation without touching cache bytes."""

        with self._generation_guard():
            generation = getattr(self, "_cache_generation", None)
            if generation is not None:
                return generation
            payload = getattr(self, "_cache_payload", None)
            if payload is None or payload.get("mode") != "completed":
                return None
            # Restored completed payloads already name an immutable snapshot-local
            # directory. Live computed payloads only become a generation in the hook.
            if "_workdir" in payload:
                return None
            generation = _CacheGeneration(payload)
            self._cache_generation = generation
            return generation

    def _iter_generation(self, generation: _CacheGeneration):
        """Traverse one leased generation and reclaim it after its last reader closes."""

        try:
            yield from self._iter_payload(generation.payload)
        except CacheIntegrityError:
            with self._generation_guard():
                if getattr(self, "_cache_generation", None) is generation:
                    self._cache_generation = None
                    self._cache_payload = {"mode": "uncomputed"}
            raise
        finally:
            with self._generation_guard():
                generation.leases -= 1
                if generation.retired and generation.leases == 0:
                    generation.payload.clear()
                    retired = getattr(self, "_retired_cache_generations", ())
                    self._retired_cache_generations = tuple(
                        item for item in retired if item is not generation
                    )

    def _managed_post_publication(self, final_state, state_repo, report) -> None:
        """Bind this invocation's completed cache to its exact immutable snapshot.

        Args:
            final_state: Exact completed StateRef already associated by managed
                control authority.
            state_repo: Connected Repo used for this invocation's publication.
            report: Immutable publication evidence naming the selected payload Store.

        Raises:
            CacheIntegrityError: If report authority cannot expose this object's
                exact completed snapshot-local payload.

        The managed runtime calls this private hook only after final association.
        It never consults ``last_state_ref``: the supplied StateRef and report are
        the sole authority for the installed generation.
        """

        payload = _snapshot_payload(final_state, state_repo, report)
        old_payload = getattr(self, "_cache_payload", None)
        old_workdir = old_payload.get("_workdir") if isinstance(old_payload, dict) else None
        with self._generation_guard():
            old = getattr(self, "_cache_generation", None)
            generation = _CacheGeneration(payload)
            self._cache_payload = payload
            self._cache_generation = generation
            if old is not None:
                old.retired = True
                if old.leases:
                    self._retired_cache_generations = (
                        *getattr(self, "_retired_cache_generations", ()), old,
                    )
                else:
                    old.payload.clear()
        if isinstance(old_workdir, str):
            _mark_work_reclaimable(state_repo, old_workdir)
            _reclaim_work(state_repo, old_workdir)

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Write cache metadata eagerly and copy completed chunks for deferred hashing."""

        payload = getattr(self, "_cache_payload", None)
        record = _payload_record(payload)
        Path(dest_dir, _MANIFEST_FILENAME).write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        if payload is not None and payload.get("mode") == "completed":
            source_dir = payload.get("source_dir")
            if not isinstance(source_dir, str):
                raise CacheIntegrityError("CachedDataset completed payload has no source directory.")
            for descriptor in payload["chunks"]:
                name = descriptor["file"]
                target = Path(dest_dir, "chunks", name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(source_dir, "chunks", name), target)

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Restore and validate only eager cache metadata, leaving chunks deferred."""

        try:
            record = json.loads(Path(src_dir, _MANIFEST_FILENAME).read_text(encoding="utf-8"))
            payload = _payload_from_record(record, src_dir)
        except CacheIntegrityError:
            raise
        except Exception as error:
            raise CacheIntegrityError("CachedDataset manifest cannot be restored.") from error
        self._cache_payload = payload
        with self._generation_guard():
            self._cache_generation = (
                _CacheGeneration(payload) if payload.get("mode") == "completed" else None
            )

    def deferred_state_payload_paths(self, data_dir: str, *, codec: str) -> tuple[str, ...]:
        """Declare complete cache chunks for lazy Store validation and iteration."""

        payload = getattr(self, "_cache_payload", None)
        if payload is None or payload.get("mode") != "completed":
            return ()
        return tuple(f"chunks/{item['file']}" for item in payload["chunks"])


def _validate_target(value: int | None) -> int:
    """Return the documented soft target after strict caller validation."""

    if value is None:
        return _DEFAULT_CHUNK_BYTES
    if type(value) is not int:
        raise TypeError("target_chunk_bytes must be a positive exact int or None.")
    if value <= 0:
        raise ValueError("target_chunk_bytes must be positive.")
    return value


def _finite_cardinality(source: Dataset) -> int:
    """Require a supported nonnegative finite cardinality before opening a cursor.

    ``Dataset`` implementations historically return either a finite
    :class:`~dryml.core.cardinality.Cardinality` or an exact ``int`` (notably
    ``ArrayDataset``).  Booleans, negative integers, unknown, and infinite
    values are deliberately not cardinalities for cache admission.
    """

    cardinality = source.__len__()
    if type(cardinality) is int:
        if cardinality < 0:
            raise ValueError("CachedDataset source cardinality must be non-negative.")
        return cardinality
    if not isinstance(cardinality, Cardinality) or not cardinality.is_finite:
        raise ValueError("CachedDataset source must have known finite cardinality.")
    assert cardinality.value is not None
    return cardinality.value


def _read_chunk(codec: str, path: str, descriptor: dict[str, object], leaves):
    """Decode one physical chunk through the selected private codec.

    Args:
        codec: Persisted built-in codec name.
        path: Snapshot-local chunk path.
        descriptor: Authenticated manifest inventory entry.
        leaves: Validated output logical leaves.

    Returns:
        Fully validated logical leaf tuples for the chunk.

    Raises:
        CacheIntegrityError: If the persisted codec is unknown or its physical
            content is structurally invalid.
        ImportError: If the selected optional codec dependency is unavailable.
    """

    if codec == "numpy":
        return read_chunk(path, descriptor, leaves)
    if codec == "parquet":
        return _cache_parquet.read_chunk(path, descriptor, leaves)
    raise CacheIntegrityError("CachedDataset chunk uses an unsupported codec.")


def _create_work_directory(repo, obj) -> str:
    """Allocate work from a Store selected by this invocation's save routing.

    The private dry-run keeps cache scratch aligned with the same immutable routing
    view the managed final save will use. It performs no state capture or mutation.
    """

    from dryml.core.repo_plan import build_routed_save_plan

    plan, _, _ = repo._state_graph_evidence(obj)
    with repo._retain_save_context() as context:
        routed = build_routed_save_plan(repo, plan, context)
    for store in routed.destinations.get(GraphPath(), ()):
        create = getattr(store, "create_local_state_staging", None)
        staging_id = getattr(store, "_local_state_staging_id", None)
        if create is None or staging_id is None:
            continue
        workdir = create()
        staging_id(workdir)
        return workdir
    raise ValueError("CachedDataset requires a selected Store with local-state staging support.")


def _work_directory(repo, token: object) -> str:
    """Resolve one work token through exactly one already connected Store.

    Resolution probes only the private direct-child namespace of connected
    Stores.  It neither creates a Store nor searches arbitrary filesystem paths,
    so a changed default route cannot redirect a retained checkpoint.
    """

    candidates = []
    for store in getattr(repo, "stores", ()):
        resolve = getattr(store, "_resolve_local_state_staging_id", None)
        if resolve is None:
            continue
        try:
            candidate = resolve(token)
            if candidate is not None:
                candidates.append(candidate)
        except Exception:
            continue
    if len(candidates) != 1:
        raise CacheIntegrityError("CachedDataset work token is missing, foreign, or ambiguous.")
    return candidates[0]


def _write_work_marker(workdir: str, managed: ManagedContext, object_ref_digest: str) -> None:
    """Bind a new Store-owned work directory to this cache operation and attempt."""

    marker = {
        "format": _WORK_FORMAT, "version": _VERSION,
        "token": _work_token(managed.state_repo, workdir), "object_ref_digest": object_ref_digest,
        "operation_id": managed.operation_id,
        "attempt_id": managed.attempt_id, "state": "retained",
    }
    path = Path(workdir, _WORK_MARKER)
    temporary = path.with_name(f".{_WORK_MARKER}.tmp")
    try:
        temporary.write_text(json.dumps(marker, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    except Exception as error:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise CacheIntegrityError("CachedDataset work marker could not be installed.") from error


def _validate_work_marker(
        workdir: str, payload: dict[str, Any], managed: ManagedContext, object_ref_digest: str,
) -> None:
    """Require the exact retained work marker before reading a saved prefix."""

    try:
        marker = json.loads(Path(workdir, _WORK_MARKER).read_text(encoding="utf-8"))
    except Exception as error:
        raise CacheIntegrityError("CachedDataset work marker is missing or malformed.") from error
    expected = {
        "format": _WORK_FORMAT, "version": _VERSION,
        "token": payload.get("work_token"), "object_ref_digest": payload.get("object_ref_digest"),
        "operation_id": payload.get("operation_id"),
        "attempt_id": payload.get("attempt_id"), "state": "retained",
    }
    if (marker != expected or marker["object_ref_digest"] != object_ref_digest
            or marker["operation_id"] != managed.operation_id
            or marker["attempt_id"] != managed.attempt_id):
        raise CacheIntegrityError("CachedDataset work marker belongs to another cache attempt.")


def _validate_stored_work_marker(workdir: str, payload: dict[str, Any]) -> None:
    """Validate persisted marker identity without accepting a live invocation.

    A rerun uses this narrower check only after managed has replaced the attempt
    authority.  The old marker is deliberately not compared with the new attempt.
    """

    try:
        marker = json.loads(Path(workdir, _WORK_MARKER).read_text(encoding="utf-8"))
    except Exception as error:
        raise CacheIntegrityError("CachedDataset work marker is missing or malformed.") from error
    expected = {
        "format": _WORK_FORMAT, "version": _VERSION,
        "token": payload.get("work_token"), "object_ref_digest": payload.get("object_ref_digest"),
        "operation_id": payload.get("operation_id"), "attempt_id": payload.get("attempt_id"),
        "state": "retained",
    }
    if marker != expected:
        raise CacheIntegrityError("CachedDataset work marker belongs to another cache attempt.")


def _discard_unassociated_tail(workdir: str, chunks: list[dict[str, Any]], codec: str) -> None:
    """Remove only contiguous sealed chunk names absent from an associated prefix.

    A checkpoint association is the sole durable inventory authority.  A crash
    after sealing a chunk but before that association may leave the next numbered
    files in work storage; no other work-tree entries are inferred disposable.
    """

    suffix = "npz" if codec == "numpy" else "parquet" if codec == "parquet" else None
    if suffix is None:
        raise CacheIntegrityError("CachedDataset work codec is unsupported.")
    directory = Path(_work_data_directory(workdir), "chunks")
    index = len(chunks)
    while True:
        candidate = directory / f"chunk-{index:08d}.{suffix}"
        try:
            mode = os.lstat(candidate).st_mode
        except FileNotFoundError:
            return
        except OSError:
            return
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            return
        try:
            candidate.unlink()
        except OSError:
            return
        index += 1


def _snapshot_payload(final_state, state_repo, report) -> dict[str, Any]:
    """Open only the exact root payload named by completed report authority.

    This deliberately uses the managed invocation's report rather than mutable
    receiver receipts or a Repo-wide search. Deferred chunk bytes remain unopened;
    the Store validates the exact snapshot-local manifest and payload inventory.
    """

    try:
        stores = tuple(report.state_stores[GraphPath()])
    except Exception as error:
        raise CacheIntegrityError("CachedDataset completed report has no root payload authority.") from error
    connected = {id(store) for store in getattr(state_repo, "stores", ())}
    for store in stores:
        if id(store) not in connected:
            continue
        open_local = getattr(store, "_open_local_state_for_exact_load", None)
        if open_local is None:
            continue
        try:
            source = open_local(final_state, GraphPath())
            data_dir = os.path.join(os.fspath(source.handle), "data")
            record = json.loads(Path(data_dir, _MANIFEST_FILENAME).read_text(encoding="utf-8"))
            return _payload_from_record(record, data_dir)
        except CacheIntegrityError:
            raise
        except Exception as error:
            raise CacheIntegrityError("CachedDataset completed snapshot payload is unavailable.") from error
    raise CacheIntegrityError("CachedDataset completed report selected no supported payload Store.")


def _mark_work_reclaimable(repo, workdir: str) -> None:
    """Mark only a validated completed work directory reclaimable under its lock.

    Marker failures preserve retained data and surface as cache integrity failures;
    no owner-loss heuristic or broad scan is used to delete work.
    """

    token, _ = _owned_work_store(repo, workdir)
    path = Path(workdir, _WORK_MARKER)
    lock = FileLock(_work_lock_path(workdir))
    try:
        lock.acquire()
        marker = json.loads(path.read_text(encoding="utf-8"))
        if not _is_work_marker(marker, token, "retained"):
            raise CacheIntegrityError("CachedDataset work marker cannot be reclaimed safely.")
        marker["state"] = "reclaimable"
        temporary = path.with_name(f".{_WORK_MARKER}.tmp")
        temporary.write_text(json.dumps(marker, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    except CacheIntegrityError:
        raise
    except (LockError, OSError, ValueError, TypeError) as error:
        raise CacheIntegrityError("CachedDataset work marker could not be marked reclaimable.") from error
    finally:
        try:
            lock.release()
        except LockError as error:
            raise CacheIntegrityError("CachedDataset work marker lease could not be released.") from error


def _reclaim_work(repo, workdir: str) -> None:
    """Discard one exact reclaimable work allocation while holding its own lease.

    The allocation is selected by its opaque token across connected Stores. A
    malformed marker, missing Store ownership, locking error, or competing sweep
    leaves all work untouched rather than guessing that owner loss means abandon.
    """

    token, store = _owned_work_store(repo, workdir)
    lock = FileLock(_work_lock_path(workdir))
    try:
        if not lock.acquire(blocking=False):
            return
        marker = json.loads(Path(workdir, _WORK_MARKER).read_text(encoding="utf-8"))
        if not _is_work_marker(marker, token, "reclaimable"):
            raise CacheIntegrityError("CachedDataset work marker cannot be reclaimed safely.")
        store.discard_local_state_staging(workdir)
    except CacheIntegrityError:
        raise
    except (LockError, OSError, ValueError, TypeError) as error:
        raise CacheIntegrityError("CachedDataset retained work cleanup failed safely.") from error
    finally:
        # Closing the caller-owned descriptor after removal is supported; FileLock
        # never owns directory cleanup and is always released by this same thread.
        try:
            lock.release()
        except LockError as error:
            raise CacheIntegrityError("CachedDataset work cleanup lease could not be released.") from error
        _remove_released_work_lock(lock, workdir)


def _sweep_reclaimable_work(repo) -> None:
    """Boundedly reclaim explicitly reclaimable cache work in connected Stores.

    The sweep intentionally scans only direct ``.staging`` children of Stores
    already connected to ``repo``.  Every malformed, unknown, contended, or
    inaccessible entry is left intact, including retained work after owner exit.
    """

    for store in getattr(repo, "stores", ()):
        root = getattr(store, "_staging_root", None)
        resolve = getattr(store, "_resolve_local_state_staging_id", None)
        discard = getattr(store, "discard_local_state_staging", None)
        if not isinstance(root, str) or resolve is None or discard is None:
            continue
        try:
            entries = tuple(os.scandir(root))
        except OSError:
            continue
        for entry in entries:
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                token = entry.name
                if resolve(token) != entry.path:
                    continue
                lock = FileLock(_work_lock_path(entry.path))
                if not lock.acquire(blocking=False):
                    continue
            except (LockError, OSError, ValueError, TypeError):
                continue
            try:
                try:
                    marker = json.loads(Path(entry.path, _WORK_MARKER).read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not _is_reclaimable_marker(marker, token):
                    continue
                discard(entry.path)
            except (LockError, OSError, ValueError, TypeError):
                continue
            finally:
                try:
                    lock.release()
                except LockError:
                    # The work directory remains authoritative until another
                    # successful sweep observes explicit reclaim evidence.
                    pass
                else:
                    _remove_released_work_lock(lock, entry.path)


def _is_reclaimable_marker(marker: object, token: str) -> bool:
    """Return whether one closed marker explicitly authorizes a sweep deletion."""

    return _is_work_marker(marker, token, "reclaimable")


def _is_work_marker(marker: object, token: str, state: str) -> bool:
    """Return whether a closed marker names one Store-owned work lifecycle state."""

    return (
        isinstance(marker, dict)
        and set(marker) == {
            "format", "version", "token", "object_ref_digest", "operation_id", "attempt_id", "state",
        }
        and marker.get("format") == _WORK_FORMAT
        and marker.get("version") == _VERSION
        and marker.get("token") == token
        and marker.get("state") == state
        and all(type(marker.get(key)) is str and marker[key] for key in (
            "object_ref_digest", "operation_id", "attempt_id",
        ))
    )


def _owned_work_store(repo, workdir: str) -> tuple[str, object]:
    """Return one exact Store owner after rejecting missing or ambiguous work paths."""

    token = _work_token(repo, workdir)
    candidates = []
    for store in getattr(repo, "stores", ()):
        resolve = getattr(store, "_resolve_local_state_staging_id", None)
        if resolve is None:
            continue
        try:
            if resolve(token) == workdir:
                candidates.append(store)
        except Exception:
            continue
    if len(candidates) != 1:
        raise CacheIntegrityError("CachedDataset reclaimable work ownership is ambiguous.")
    return token, candidates[0]


def _work_lock_path(workdir: str) -> Path:
    """Return a sibling lease path so reclamation never removes its held lock file."""

    path = Path(workdir)
    return path.parent / f".{path.name}.cache-work.lock"


def _remove_released_work_lock(lock: FileLock, workdir: str) -> None:
    """Remove a sibling lease file only after its work directory is gone.

    Retaining the lock outside its directory keeps Windows-compatible removal
    possible.  Removing the now-unreferenced sibling after release also keeps
    the Store staging namespace limited to actual work allocations.
    """

    if Path(workdir).exists():
        return
    try:
        Path(lock.path).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # Cleanup can retry from the explicit marker path only while work exists;
        # a stale sibling lock is never authority and must not hide completion.
        pass


def _work_token(repo, workdir: str) -> str:
    """Return the validated opaque staging ID for an allocated work directory."""

    for store in getattr(repo, "stores", ()):
        staging_id = getattr(store, "_local_state_staging_id", None)
        if staging_id is None:
            continue
        try:
            return staging_id(workdir)
        except Exception:
            continue
    raise CacheIntegrityError("CachedDataset work directory is not Store-owned.")


def _work_data_directory(workdir: str) -> str:
    """Return the allocated local-state payload directory used for work chunks."""

    return os.path.join(workdir, "data")


def _payload_record(payload: dict[str, Any] | None) -> dict[str, object]:
    """Create the closed persisted manifest without local paths or chunk bytes."""

    if payload is None:
        return {"format": _FORMAT, "version": _VERSION, "mode": "uncomputed"}
    mode = payload.get("mode")
    if mode == "working":
        prior_ready_state_digest = payload.get("prior_ready_state_digest")
        if prior_ready_state_digest is not None and not _is_digest(prior_ready_state_digest):
            raise CacheIntegrityError("CachedDataset prior ready reference is invalid.")
        return {
            "format": _FORMAT, "version": _VERSION, "mode": "working", "codec": payload["codec"],
            "spec": spec_to_data(payload["spec"]), "expected": payload["expected"],
            "count": payload["count"], "work_token": payload["work_token"], "chunks": payload["chunks"],
            "object_ref_digest": payload["object_ref_digest"], "operation_id": payload["operation_id"],
            "attempt_id": payload["attempt_id"],
            **({"prior_ready_state_digest": prior_ready_state_digest} if prior_ready_state_digest is not None else {}),
        }
    if mode == "completed":
        return {
            "format": _FORMAT, "version": _VERSION, "mode": "completed", "codec": payload["codec"],
            "spec": spec_to_data(payload["spec"]), "count": payload["count"], "chunks": payload["chunks"],
        }
    raise CacheIntegrityError("CachedDataset payload mode is invalid.")


def _payload_from_record(record: object, src_dir: str) -> dict[str, Any]:
    """Validate a persisted cache manifest without reading deferred chunk bytes."""

    if not isinstance(record, dict) or record.get("format") != _FORMAT or record.get("version") != _VERSION:
        raise CacheIntegrityError("CachedDataset manifest format is unsupported.")
    mode = record.get("mode")
    if mode == "uncomputed" and set(record) == {"format", "version", "mode"}:
        return {"mode": "uncomputed"}
    common = {"format", "version", "mode", "codec", "spec", "count", "chunks"}
    if mode == "working":
        common |= {"expected", "work_token", "object_ref_digest", "operation_id", "attempt_id"}
        if "prior_ready_state_digest" in record:
            common |= {"prior_ready_state_digest"}
    if mode not in {"working", "completed"} or set(record) != common or record["codec"] not in {"numpy", "parquet"}:
        raise CacheIntegrityError("CachedDataset manifest fields are invalid.")
    if type(record["count"]) is not int or record["count"] < 0 or not isinstance(record["chunks"], list):
        raise CacheIntegrityError("CachedDataset manifest count or chunks are invalid.")
    spec = spec_from_data(record["spec"])
    leaves = flatten_spec(spec)
    chunks = record["chunks"]
    if sum(item.get("count", -1) for item in chunks if isinstance(item, dict)) != record["count"]:
        raise CacheIntegrityError("CachedDataset manifest chunk counts are invalid.")
    for index, item in enumerate(chunks):
        suffix = "npz" if record["codec"] == "numpy" else "parquet"
        if not isinstance(item, dict) or item.get("file") != f"chunk-{index:08d}.{suffix}":
            raise CacheIntegrityError("CachedDataset manifest chunk names are invalid.")
        if record["codec"] == "numpy":
            validate_chunk_descriptor(item, leaves)
        else:
            _cache_parquet.validate_chunk_descriptor(item, leaves)
    if mode == "working":
        if (type(record["expected"]) is not int or record["expected"] < record["count"]
                or type(record["work_token"]) is not str
                or not all(type(record[key]) is str and record[key] for key in ("object_ref_digest", "operation_id", "attempt_id"))):
            raise CacheIntegrityError("CachedDataset working manifest is invalid.")
        result = {"mode": mode, "codec": record["codec"], "spec": spec, "expected": record["expected"], "count": record["count"], "work_token": record["work_token"], "object_ref_digest": record["object_ref_digest"], "operation_id": record["operation_id"], "attempt_id": record["attempt_id"], "chunks": chunks}
        if "prior_ready_state_digest" in record:
            if not _is_digest(record["prior_ready_state_digest"]):
                raise CacheIntegrityError("CachedDataset prior ready reference is invalid.")
            result["prior_ready_state_digest"] = record["prior_ready_state_digest"]
        return result
    for item in chunks:
        if not Path(src_dir, "chunks", item["file"]).is_file():
            raise CacheIntegrityError("CachedDataset manifest names a missing chunk.")
    return {"mode": mode, "codec": record["codec"], "spec": spec, "count": record["count"], "chunks": chunks, "source_dir": src_dir}


def _is_digest(value: object) -> bool:
    """Return whether ``value`` is one canonical immutable StateRef digest."""

    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _prior_ready_state_ref(repo, payload: dict[str, Any]) -> StateRef:
    """Resolve a working replacement's exact prior completed StateRef on demand.

    Working metadata is intentionally inspectable without this lookup.  Consumers
    that explicitly request prior-ready content must resolve the recorded digest
    through current connected Store authority; missing or corrupt prior snapshots
    are errors and never turn the partial working prefix into ready content.
    """

    from dryml.managed.storage import state_ref_for_digest, validate_state_ref

    digest = payload.get("prior_ready_state_digest")
    if not _is_digest(digest):
        raise CacheIntegrityError("CachedDataset working state has no prior ready reference.")
    try:
        reference = state_ref_for_digest(repo, digest)
        validate_state_ref(repo, reference)
    except Exception as error:
        raise CacheIntegrityError("CachedDataset prior ready state is unavailable.") from error
    return reference


CachedDataset.__module__ = "dryml.artifacts"

__all__ = ["CacheCodec", "CacheIntegrityError", "CachedDataset"]
