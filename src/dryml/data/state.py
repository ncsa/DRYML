"""Detached DatasetState v1 records and cooperating no-clobber directory IO.

Records own immutable codec bytes, not Dataset objects, invokers, Repo handles,
frames, tokens, or retention roots. Reading and validation are inert. Only opening
a supplied equivalent target authorizes lazy execution/source restoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Generic, TypeVar, TYPE_CHECKING

import numpy as np

from dryml.formats.canonical import canonical_json_bytes, canonical_json_load_bytes, json_ready
from dryml.locking import FileLock
from .continuation import _BOUNDS, _MAX_BYTES, decode_value, encode_value
from .errors import ContinuationFormatError, ContinuationMismatchError, UnsupportedContinuationError
from .recovery import _capture_execution, _evidence, _validate_execution
from .traversal import TraversalPosition

if TYPE_CHECKING:
    from dryml.core.definition import ConcreteNode, SavedNode
    from .dataset import Dataset
    from .native import PreparedDataset
    from .recovery import ReplayRecoveryPolicy

T = TypeVar("T")
StatePath = str | os.PathLike[str]
_FORMAT = "dryml-dataset-state"
_VERSION = 1
_PROTOCOL = dict(python=list(sys.version_info[:2]), numpy=np.__version__, replay=1, codec=1, kernel=1)


def _fields(value, fields, name):
    if type(value) is not dict or set(value) != set(fields):
        raise ContinuationFormatError(f"Invalid {name} fields.")


def _validate(data):
    from .recovery import (_initial_value, _validate_selection, _decode_node,
                           _metadata_value, _spec_value, _symbol_value, _controls_value, _validate_tree)
    _fields(data, ("format", "version", "protocol", "target", "state_ref", "snapshots", "position", "policy", "active_replay",
                   "explicit", "pass_nonempty", "initialized", "staged", "staged_boundary", "boundary_pending", "execution"), "DatasetState")
    if data["format"] != _FORMAT or type(data["version"]) is not int or data["version"] != _VERSION or canonical_json_bytes(data["protocol"], **_BOUNDS) != canonical_json_bytes(_PROTOCOL, **_BOUNDS):
        raise ContinuationFormatError("Unsupported DatasetState format/protocol version.")
    if data["policy"] not in {"disabled", "enabled", "if_needed"}:
        raise ContinuationFormatError("Invalid replay policy.")
    for flag in ("active_replay", "explicit", "pass_nonempty", "initialized", "staged_boundary", "boundary_pending"):
        if type(data[flag]) is not bool:
            raise ContinuationFormatError("Invalid traversal flag.")
    try:
        _fields(data["position"], ("epoch", "start", "sequence", "section_complete", "root_complete"), "position")
        TraversalPosition(**data["position"])
    except (TypeError, ValueError) as error:
        raise ContinuationFormatError("Invalid traversal position.") from error
    rows = data["target"]
    if data["state_ref"] is not None:
        from dryml.core.node import require_saved
        require_saved(_decode_node(data["state_ref"]))
    if type(rows) is not list or not rows or type(data["snapshots"]) is not dict:
        raise ContinuationFormatError("Missing target occurrences/snapshots.")
    by_path, children_by_path, aliases, arrays = {}, {}, set(), set()
    for row in rows:
        _fields(row, ("path", "kind", "alias", "controls", "spec", "cardinality", "sectioned", "sections",
                      "section_bound", "recipe", "method", "state", "tree", "source", "selection"), "target occurrence")
        if type(row["path"]) is not list or any(type(n) is not int or n < 0 for n in row["path"]):
            raise ContinuationFormatError("Invalid target occurrence path.")
        path = tuple(row["path"])
        if path in by_path or path and path[:-1] not in by_path:
            raise ContinuationFormatError("Duplicate or missing target occurrence.")
        by_path[path] = row
        if path:
            children_by_path.setdefault(path[:-1], []).append(path[-1])
        if row["kind"] not in {"source", "map", "batch", "unbatch", "take", "skip", "repeat", "shuffle", "chain", "zip", "epoch"}:
            raise ContinuationFormatError("Unknown target operator tag.")
        if type(row["alias"]) is not int or row["alias"] < 0 or row["alias"] > len(aliases):
            raise ContinuationFormatError("Invalid target sharing partition.")
        aliases.add(row["alias"])
        if type(row["sectioned"]) is not bool or row["section_bound"] is not None and (type(row["section_bound"]) is not int or row["section_bound"] < 0):
            raise ContinuationFormatError("Invalid target section metadata.")
        _controls_value(row)
        spec = _spec_value(row["spec"])
        from dryml.core.tensor_spec import is_spec_tree
        if spec is not None and not is_spec_tree(spec):
            raise ContinuationFormatError("Invalid target specification tree.")
        _metadata_value(row["cardinality"], {"cardinality"})
        _validate_selection(row["selection"])
        if (row["kind"] in {"map", "batch", "unbatch"}) != (row["selection"] is not None):
            raise ContinuationFormatError("Missing or unexpected implementation selection.")
        for key in ("recipe", "method", "state"):
            if row[key] is not None:
                try:
                    _decode_node(row[key])
                except (TypeError, ValueError) as error:
                    raise ContinuationFormatError("Invalid target Node metadata.") from error
        if row["kind"] == "source":
            source = row["source"]
            if type(source) is not dict or source.get("kind") not in {"array", "generator"}:
                raise ContinuationFormatError("Invalid source evidence.")
            if source["kind"] == "array":
                _fields(source, ("kind", "length"), "array source")
                if type(source["length"]) is not int or source["length"] < 0:
                    raise ContinuationFormatError("Invalid array length.")
                arrays.add(str(path))
            else:
                _fields(source, ("kind", "factory", "initial", "seed", "seed_aware", "seed_version", "declared_direct"), "generator source")
                from dryml.core.symbol import ImportRef, SourceSpec
                if not isinstance(_symbol_value(source["factory"]), (ImportRef, SourceSpec)):
                    raise ContinuationFormatError("Invalid source factory evidence.")
                if (type(source["seed_aware"]) is not bool or type(source["declared_direct"]) is not bool
                        or type(source["seed_version"]) is not int or source["seed_version"] < 1
                        or source["seed"] is not None and type(source["seed"]) is not int):
                    raise ContinuationFormatError("Invalid source seed/protocol evidence.")
                initial = _initial_value(source["initial"])
                if type(initial) is not tuple or len(initial) != 2 or type(initial[0]) is not tuple or type(initial[1]) is not dict:
                    raise ContinuationFormatError("Invalid initial source declaration.")
        elif row["source"] is not None:
            raise ContinuationFormatError("Source evidence on non-source target.")
    if set(data["snapshots"]) != arrays:
        raise ContinuationFormatError("Array snapshot inventory mismatch.")
    if () not in by_path or data["explicit"] != by_path[()]["sectioned"]:
        raise ContinuationFormatError("Invalid root/boundary mode.")
    for path, row in by_path.items():
        if not path and row["recipe"] is None:
            raise ContinuationFormatError("Missing frozen target CDef.")
        if (row["kind"] == "map") != (row["method"] is not None):
            raise ContinuationFormatError("Invalid Method recipe association.")
        sections = row["sections"]
        if not row["sectioned"] and (sections is not None or row["section_bound"] is not None):
            raise ContinuationFormatError("Section metadata on an implicit target.")
        if sections is not None:
            if type(sections) is not tuple:
                raise ContinuationFormatError("Invalid target section runs.")
            for index, run in enumerate(sections):
                if (type(run) is not tuple or len(run) != 2 or type(run[0]) is not int or run[0] < 0
                        or run[1] is not None and (type(run[1]) is not int or run[1] < 0)
                        or run[1] is None and index != len(sections) - 1):
                    raise ContinuationFormatError("Invalid target section run values.")
        children = children_by_path.get(path, [])
        if children != list(range(len(children))):
            raise ContinuationFormatError("Invalid target child ordering.")
        count = len(children)
        if (row["kind"] == "source" and count or row["kind"] in {"chain", "zip"} and count == 0
                or row["kind"] not in {"source", "chain", "zip"} and count != 1):
            raise ContinuationFormatError("Invalid target arity.")
        if row["kind"] == "zip":
            if _validate_tree(row["tree"]) != count:
                raise ContinuationFormatError("Zip structure/occurrence count mismatch.")
        elif row["tree"] is not None:
            raise ContinuationFormatError("Zip structure on a different operator.")
        if row["kind"] == "source" and row["source"]["kind"] == "array":
            from dryml.core.utils.recurse import iter_leaves
            leaves = list(iter_leaves(data["snapshots"][str(path)]))
            from .continuation import HostBuffer
            if not leaves or any(type(v) not in (np.ndarray, HostBuffer) or not v.shape or v.shape[0] != row["source"]["length"] for v in leaves):
                raise ContinuationFormatError("Array snapshot shape/length mismatch.")
    _validate_execution(data["execution"], by_path)
    executions = tuple(_record_executions(data["execution"]))
    replay = any(e["source"] is not None and e["source"]["strategy"] == "replay" for e in executions)
    active = data["policy"] == "enabled" or data["policy"] == "if_needed" and replay
    if data["active_replay"] != active or data["policy"] == "disabled" and replay:
        raise ContinuationFormatError("Inconsistent replay policy/strategies.")
    if data["execution"] is not None and data["execution"]["count"] and not data["pass_nonempty"]:
        raise ContinuationFormatError("Missing physical pass nonempty evidence.")
    if data["execution"] is not None and data["execution"]["path"] != []:
        raise ContinuationFormatError("Execution must start at the root occurrence.")
    if data["staged"] is not None and data["execution"] is None or data["staged_boundary"] and data["staged"] is not None:
        raise ContinuationFormatError("Inconsistent staged frontier.")
    if data["staged"] is not None and (type(data["staged"]) is not tuple or len(data["staged"]) != 3
            or any(type(v) is not bool for v in data["staged"][1:])):
        raise ContinuationFormatError("Invalid staged unexposed delivery.")


@dataclass(frozen=True, slots=True, init=False)
class DatasetState(Generic[T]):
    """Immutable detached continuation created by capture_state() or load().

    There is no public Dataset-taking constructor, ObjectId, graph edge, or live
    Dataset property. Metadata names the frozen prepared recipe and required state
    selections. Owned snapshots/buffers survive the original traversal; external
    Store selections remain the caller's separate retention responsibility.
    Schema v1 rejects unknown fields/protocols rather than silently converting.
    """

    _metadata: bytes
    _payloads: tuple[bytes, ...]
    _saved_position: TraversalPosition
    _saved_policy: str
    _saved_replay: bool
    _recipe_packet: tuple[bytes, tuple[bytes, ...]] = field(repr=False, compare=False)
    _state_ref_packet: tuple[bytes, tuple[bytes, ...]] | None = field(
        repr=False, compare=False)

    def __new__(cls, *args, **kwargs):
        """Reject direct construction; use traversal.capture_state() or load()."""
        raise TypeError("DatasetState is created through capture_state() or load().")

    @classmethod
    def _from_bytes(cls, metadata, payloads):
        try:
            decoded = decode_value(metadata, payloads)
            _validate(decoded)
        except ContinuationFormatError:
            raise
        except (TypeError, ValueError, KeyError, IndexError, OverflowError, RecursionError) as error:
            raise ContinuationFormatError("Malformed DatasetState metadata/frontier.") from error
        record = object.__new__(cls)
        object.__setattr__(record, "_metadata", metadata)
        object.__setattr__(record, "_payloads", payloads)
        object.__setattr__(record, "_saved_position", TraversalPosition(**decoded["position"]))
        object.__setattr__(record, "_saved_policy", decoded["policy"])
        object.__setattr__(record, "_saved_replay", decoded["active_replay"])
        object.__setattr__(record, "_recipe_packet",
                           encode_value(decoded["target"][0]["recipe"]))
        object.__setattr__(record, "_state_ref_packet",
                           None if decoded["state_ref"] is None else
                           encode_value(decoded["state_ref"]))
        return record

    def _data(self):
        return decode_value(self._metadata, self._payloads)

    @property
    def cdef(self) -> ConcreteNode:
        """Decode independent target ConcreteNode metadata, not the frontier.

        Resolves no target and acquires no source; returned mutable literals cannot
        alter this record or metadata returned by another property access.
        """
        from .recovery import _decode_node
        return _decode_node(decode_value(*self._recipe_packet))

    @property
    def state_ref(self) -> SavedNode | None:
        """Decode independent exact target SavedNode metadata, or return None.

        Decodes no frontier and grants no retention or restoration authority;
        returned mutable literals cannot change this record or later reads.
        """
        from .recovery import _decode_node
        return (None if self._state_ref_packet is None else
                _decode_node(decode_value(*self._state_ref_packet)))

    @property
    def position(self) -> TraversalPosition:
        """Return the saved accepted position, including boundary/terminal facts."""
        return self._saved_position

    @property
    def replay_recovery_policy(self) -> ReplayRecoveryPolicy:
        """Return disabled, enabled, or pending/active conditional if_needed policy."""
        return self._saved_policy

    @property
    def replay_recovery_enabled(self) -> bool:
        """Return True for manual enablement or an actually retained replay source."""
        return self._saved_replay

    def validate(self, target: Dataset[T] | PreparedDataset[T]) -> None:
        """Compare complete frozen recipe, sharing, source/spec and protocol evidence.

        Args:
            target: Caller-supplied Dataset or already prepared equivalent recipe.
        Returns:
            None on compatible metadata; no source is advanced or reconstructed.
        Raises:
            TypeError: Target is neither a Dataset nor a PreparedDataset.
            ContinuationMismatchError: Target differs from the saved execution.
            UnsupportedContinuationError: Target lacks qualified exact evidence.
        Side Effects:
            May prepare inert metadata/selections and read supplied array bytes.
            Never constructs a Dataset, calls factories/hooks, opens Stores, or
            restores referenced Objects. Does not certify future availability.
        """
        self._validated_data(target)

    def _validated_data(self, target):
        """Return fresh host-decoded data after complete inert target validation."""
        from .native import PreparedDataset
        from .dataset import Dataset
        if isinstance(target, Dataset):
            target = target.prepare()
        if not isinstance(target, PreparedDataset):
            raise TypeError("Continuation target must be a Dataset or PreparedDataset.")
        rows, snapshots = _evidence(target)
        data = self._data()
        if data["state_ref"] is not None:
            from .recovery import _node_packet
            selected = target._continuation_target_state
            if selected is None or _node_packet(selected) != data["state_ref"]:
                raise ContinuationMismatchError("Required target SavedNode selection differs.")
        if encode_value((rows, snapshots)) != encode_value((data["target"], data["snapshots"])):
            raise ContinuationMismatchError("Frozen Dataset recipe/source/spec/sharing evidence differs.")
        return data

    def save(self, path: StatePath) -> Path:
        """Publish a new complete inventoried directory without replacing any name.

        Args:
            path: Caller-selected new directory; its parent must already exist.
        Returns:
            Committed Path. No Repo or managed checkpoint/retention is created.
        Raises:
            FileExistsError: Destination already exists (including a competing save).
            ContinuationFormatError: Invalid record or staged inventory.
            OSError: Filesystem/locking/short-write failure. Interruptions propagate;
                around rename publication may have committed, never retry overwrite.
        Side Effects:
            Writes/fsyncs private same-filesystem sibling staging, validates it,
            then holds a stable sibling advisory name lock across the final absence
            check and atomic rename. Cleanup owns only this invocation's staging;
            secondary cleanup errors never mask a primary failure. Requires local
            advisory-lock and atomic directory rename filesystem semantics.
        """
        self._from_bytes(self._metadata, self._payloads)
        path = Path(path).absolute()
        staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent))
        renamed = False
        try:
            files = {"record.json": self._metadata}
            files.update({f"{index:06d}.bin": raw for index, raw in enumerate(self._payloads)})
            inventory = []
            for name, raw in files.items():
                _write(staging / name, raw)
                inventory.append(dict(name=name, size=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            manifest = dict(format=_FORMAT, version=_VERSION, files=inventory)
            _write(staging / "manifest.json", canonical_json_bytes(manifest, **_BOUNDS))
            self.load(staging)
            lock = FileLock(path.with_name(f".{path.name}.dataset-state.lock"))
            with _name_lock(lock):
                if os.path.lexists(path):
                    raise FileExistsError(path)
                renamed = True
                os.rename(staging, path)
            return path
        except BaseException as error:
            if renamed:
                error.add_note("DatasetState publication may have committed around rename; inspect the destination before retrying.")
            try:
                if staging.exists():
                    shutil.rmtree(staging)
            except BaseException as cleanup:
                error.add_note(f"DatasetState staging cleanup failed: {cleanup}")
            raise

    @classmethod
    def load(cls, path: StatePath) -> DatasetState[T]:
        """Read only validated record-owned metadata/payload bytes from a directory.

        Args:
            path: Complete DatasetState v1 directory.
        Returns:
            Independent immutable record with host-only decoded native evidence.
        Raises:
            ContinuationFormatError: Incomplete/corrupt/extra inventory, malformed
                tags/counters, unsupported schema/protocol, or integrity mismatch.
            OSError: Missing/unreadable paths and filesystem failures.
        Side Effects:
            Reads owned files only. Never resolves targets, constructs a Dataset,
            calls a factory/state hook, imports optional backends, opens referenced
            Stores, restores Objects, or establishes a retention root.
        """
        path = Path(path)
        path.stat()
        if not path.is_dir():
            raise NotADirectoryError(path)
        try:
            manifest = json_ready(canonical_json_load_bytes(_read(path / "manifest.json", 16 * 1024 * 1024), **_BOUNDS), **_BOUNDS)
            _fields(manifest, ("format", "version", "files"), "manifest")
            if manifest["format"] != _FORMAT or type(manifest["version"]) is not int or manifest["version"] != _VERSION:
                raise ContinuationFormatError("Unsupported DatasetState manifest.")
            inventory = manifest["files"]
            if type(inventory) is not list or not inventory or len(inventory) > 100000:
                raise ContinuationFormatError("Invalid payload inventory.")
            files, total = {}, 0
            for index, entry in enumerate(inventory):
                _fields(entry, ("name", "size", "sha256"), "inventory")
                name = "record.json" if index == 0 else f"{index - 1:06d}.bin"
                if entry["name"] != name or type(entry["size"]) is not int or not 0 <= entry["size"] <= _MAX_BYTES or type(entry["sha256"]) is not str:
                    raise ContinuationFormatError("Invalid inventory name/size/integrity.")
                total += entry["size"]
                if total > _MAX_BYTES:
                    raise ContinuationFormatError("DatasetState inventory exceeds bounds.")
                raw = _read(path / name, entry["size"])
                if len(raw) != entry["size"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise ContinuationFormatError("DatasetState payload integrity mismatch.")
                files[name] = raw
            if set(p.name for p in path.iterdir()) != set(files) | {"manifest.json"}:
                raise ContinuationFormatError("DatasetState directory inventory mismatch.")
            return cls._from_bytes(files.pop("record.json"), tuple(files.values()))
        except FileNotFoundError as error:
            if path.is_dir():
                raise ContinuationFormatError("Incomplete DatasetState directory.") from error
            raise
        except ContinuationFormatError:
            raise
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise ContinuationFormatError("Malformed DatasetState directory.") from error


def _write(path, raw):
    with path.open("xb") as file:
        if file.write(raw) != len(raw):
            raise OSError("Short DatasetState write.")
        file.flush()
        os.fsync(file.fileno())


def _read(path, limit):
    size = path.stat().st_size
    if not path.is_file() or size > limit:
        raise ContinuationFormatError("Invalid DatasetState inventory file/size.")
    with path.open("rb") as file:
        raw = file.read(limit + 1)
    if len(raw) > limit:
        raise ContinuationFormatError("DatasetState file exceeds bound.")
    return raw


@contextmanager
def _name_lock(lock):
    if not lock.acquire():
        raise OSError("Blocking DatasetState name lock unexpectedly failed to acquire.")
    try:
        yield
    except BaseException as primary:
        try:
            lock.release()
        except BaseException as cleanup:
            primary.add_note(f"DatasetState name-lock release failed: {cleanup}")
        raise
    else:
        lock.release()


def _capture(traversal):
    from dataclasses import asdict
    rows, snapshots = _evidence(traversal._prepared, traversal._root)
    execution = _capture_execution(traversal._execution, traversal._replay_policy)
    from .recovery import _node_packet
    selected = traversal._prepared._continuation_target_state
    data = dict(format=_FORMAT, version=_VERSION, protocol=_PROTOCOL, target=rows,
                state_ref=None if selected is None else _node_packet(selected),
                snapshots=snapshots, position=asdict(traversal.position), policy=traversal._replay_policy,
                active_replay=traversal.replay_recovery_enabled, explicit=traversal._explicit,
                pass_nonempty=traversal._pass_nonempty, initialized=traversal._initialized,
                staged=traversal._staged, staged_boundary=traversal._staged_boundary,
                boundary_pending=traversal._boundary_pending, execution=execution)
    # Unresolved factories require an explicit trusted replay decision unless
    # their declared class provides inert direct capability.
    if traversal._replay_policy == "disabled":
        active = {tuple(e["path"]): e for e in _record_executions(execution)}
        for row in rows:
            if row["kind"] == "source" and row["source"]["kind"] == "generator" and not row["source"]["declared_direct"]:
                current = active.get(tuple(row["path"]))
                if current is None or current["source"] is None or current["source"]["strategy"] != "direct":
                    raise UnsupportedContinuationError("Unresolved ordinary generator requires enabled or conditional replay.")
    return DatasetState._from_bytes(*encode_value(data))


def _record_executions(record):
    if record is not None:
        yield record
        for child in record["children"]:
            yield from _record_executions(child)
