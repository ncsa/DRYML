"""Persisted, reference-associated Experiment checkpoint history.

The module deliberately does not import pandas.  Reference discovery and payload
codec work remain available in lightweight processes; :attr:`ExperimentData.data`
imports pandas only when callers request tabular analysis.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from dryml.core import ObjectRef, Ref, StateRef
from dryml.core.cardinality import Cardinality, CardinalityKind
from dryml.core.object import Serializable
from dryml.core.repo import RepoLoadError, RepoSaveError
from dryml.core.store.records import StateAliasRecord
from dryml.core.store.store import StoreAliasConflictError, StoreAuthorityError


_FORMAT = "dryml-experiment-data"
_VERSION = 1
_FILENAME = "experiment_data.json"
_ALIAS = "experiment_data_current_v1"
_MAX_BYTES = 64 * 1024 * 1024
_MAX_ROWS = 100_000
_MAX_COLUMNS = 1_024
_MAX_ARTIFACTS = 4_096
_MAX_NAME_BYTES = 4_096
_MISSING = object()

_FACT_NAMES = (
    "row_key", "prev_row_key", "state_ref", "prev_state_ref", "time",
    "examples_seen", "total_parameters", "trainable_parameters",
    "dataset_size_kind", "dataset_size", "training_loss",
)
_RESERVED_COLUMNS = frozenset(_FACT_NAMES + (
    "expected_artifacts", "artifact_inputs", "eval_artifacts",
    "evaluation_status", "failed_artifact", "scalars",
))
_COLUMN_KINDS = {
    **{name: "fact" for name in _FACT_NAMES},
    "expected_artifacts": "artifacts",
    "artifact_inputs": "reference_mapping",
    "eval_artifacts": "reference_mapping",
    "evaluation_status": "status",
    "failed_artifact": "status",
}


class ExperimentDataError(ValueError):
    """Raised when history rows or their closed v1 payload are invalid."""


def _copy(value):
    """Deep-copy history data without replacing the private missing sentinel."""

    return copy.deepcopy(value, {id(_MISSING): _MISSING})


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentDataError(f"Duplicate JSON object key {key!r}.")
        result[key] = value
    return result


def _json_loads(raw: bytes):
    if len(raw) > _MAX_BYTES:
        raise ExperimentDataError("ExperimentData payload exceeds the 64 MiB v1 limit.")
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ExperimentDataError(f"Non-finite JSON constant {value!r} is invalid.")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExperimentDataError("ExperimentData payload is not valid UTF-8 JSON.") from error


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentDataError(f"{what} must be a non-empty string.")
    if len(value.encode("utf-8")) > _MAX_NAME_BYTES:
        raise ExperimentDataError(f"{what} exceeds the 4,096-byte UTF-8 limit.")
    return value


def _counter(value: object, what: str) -> int:
    if type(value) is not int or value < 0:
        raise ExperimentDataError(f"{what} must be a non-negative integer, not bool.")
    return value


def _scalar(value: object, what: str = "scalar") -> object:
    if value is _MISSING or value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ExperimentDataError(f"{what} must be missing, null, bool, int, finite float, or str.")


def _encode_cell(value: object) -> dict[str, object]:
    value = _scalar(value)
    if value is _MISSING:
        return {"tag": "missing"}
    if value is None:
        return {"tag": "null"}
    if type(value) is bool:
        return {"tag": "bool", "value": value}
    if type(value) is int:
        return {"tag": "int", "value": str(value)}
    if type(value) is float:
        return {"tag": "float", "value": value}
    return {"tag": "str", "value": value}


def _decode_cell(value: object) -> object:
    if not isinstance(value, Mapping) or "tag" not in value:
        raise ExperimentDataError("Scalar cells require a tag.")
    tag = value["tag"]
    expected = {"tag"} if tag in {"missing", "null"} else {"tag", "value"}
    if set(value) != expected:
        raise ExperimentDataError(f"Scalar cell {tag!r} has invalid fields.")
    if tag == "missing":
        return _MISSING
    if tag == "null":
        return None
    cell = value["value"]
    if tag == "bool" and type(cell) is bool:
        return cell
    if tag == "str" and type(cell) is str:
        return cell
    if tag == "float" and type(cell) is float and math.isfinite(cell):
        return cell
    if tag == "int" and type(cell) is str:
        if not cell or cell == "-0" or cell.startswith("+"):
            raise ExperimentDataError("Integer scalar is not canonical decimal text.")
        digits = cell[1:] if cell.startswith("-") else cell
        if not digits.isdigit() or (len(digits) > 1 and digits.startswith("0")):
            raise ExperimentDataError("Integer scalar is not canonical decimal text.")
        return int(cell)
    raise ExperimentDataError(f"Unsupported or malformed scalar tag {tag!r}.")


def _encode_ref(value: StateRef) -> dict[str, object]:
    if not isinstance(value, StateRef):
        raise ExperimentDataError("History reference values must be exact StateRefs.")
    return {"tag": "state_ref", "value": _reference_to_json(value.to_data())}


def _decode_ref(value: object, *, nullable: bool = False) -> StateRef | None:
    if value is None and nullable:
        return None
    if not isinstance(value, Mapping) or set(value) != {"tag", "value"} or value["tag"] != "state_ref":
        raise ExperimentDataError("History references require an exact state_ref record.")
    try:
        return StateRef.from_data(_reference_from_json(value["value"]))
    except (TypeError, ValueError) as error:
        raise ExperimentDataError("History reference record is malformed.") from error


def _reference_to_json(value: object) -> dict[str, object]:
    """Encode a strict StateRef.to_data tree without relying on JSON inference."""

    from dryml.core.symbol import ImportRef

    if isinstance(value, ImportRef):
        return {"type": "import_ref", "module": value.module, "qualname": value.qualname}
    if value is None:
        return {"type": "null"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": str(value)}
    if type(value) is float and math.isfinite(value):
        return {"type": "float", "value": value}
    if type(value) is str:
        return {"type": "str", "value": value}
    if isinstance(value, list):
        return {"type": "list", "items": [_reference_to_json(item) for item in value]}
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise ExperimentDataError("StateRef record has a non-string JSON mapping key.")
        return {
            "type": "dict",
            "items": [[key, _reference_to_json(item)] for key, item in value.items()],
        }
    raise ExperimentDataError(
        f"StateRef.to_data() contains unsupported non-JSON value {type(value).__name__}."
    )


def _reference_from_json(value: object) -> object:
    """Rebuild the exact StateRef.to_data tree before strict reference decoding."""

    from dryml.core.symbol import ImportRef

    if not isinstance(value, Mapping) or not isinstance(value.get("type"), str):
        raise ExperimentDataError("StateRef record is not a closed JSON value tree.")
    tag = value["type"]
    if tag == "null" and set(value) == {"type"}:
        return None
    if tag == "bool" and set(value) == {"type", "value"} and type(value["value"]) is bool:
        return value["value"]
    if tag == "str" and set(value) == {"type", "value"} and type(value["value"]) is str:
        return value["value"]
    if tag == "float" and set(value) == {"type", "value"} and type(value["value"]) is float and math.isfinite(value["value"]):
        return value["value"]
    if tag == "int" and set(value) == {"type", "value"}:
        return _decode_cell({"tag": "int", "value": value["value"]})
    if tag == "import_ref" and set(value) == {"type", "module", "qualname"}:
        try:
            return ImportRef(value["module"], value["qualname"])
        except (TypeError, ValueError) as error:
            raise ExperimentDataError("StateRef record has an invalid import reference.") from error
    if tag == "list" and set(value) == {"type", "items"} and isinstance(value["items"], list):
        return [_reference_from_json(item) for item in value["items"]]
    if tag == "dict" and set(value) == {"type", "items"} and isinstance(value["items"], list):
        result = {}
        for item in value["items"]:
            if not isinstance(item, list) or len(item) != 2 or type(item[0]) is not str or item[0] in result:
                raise ExperimentDataError("StateRef record has an invalid mapping entry.")
            result[item[0]] = _reference_from_json(item[1])
        return result
    raise ExperimentDataError("StateRef record has an unknown JSON value tag.")


def _mapping(value: object, expected: tuple[str, ...], what: str) -> tuple[tuple[str, StateRef], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise ExperimentDataError(f"{what} must be a mapping of Artifact names to StateRefs.")
    names = tuple(value)
    if len(names) > _MAX_ARTIFACTS:
        raise ExperimentDataError("A row exceeds the Artifact-name limit.")
    if any(name not in expected for name in names):
        raise ExperimentDataError(f"{what} contains a name not declared by expected_artifacts.")
    if names != tuple(name for name in expected if name in value):
        raise ExperimentDataError(f"{what} names must retain expected_artifacts declaration order.")
    return tuple((name, _require_state_ref(value[name], what)) for name in names)


def _require_state_ref(value: object, what: str) -> StateRef:
    if not isinstance(value, StateRef):
        raise ExperimentDataError(f"{what} values must be exact StateRefs.")
    return value


class ExperimentData(Serializable):
    """Persist a checkpoint history table for one projected Experiment subject.

    Args:
        experiment: Default-policy :class:`~dryml.core.ObjectRef` projection of
            the Experiment graph. It is retained as a non-materializing ``Ref``.

    Raises:
        TypeError: If ``experiment`` is not an ObjectRef projection.

    Side Effects:
        Construction creates only an empty in-memory history. It does not import
        pandas, load the referenced Experiment, or publish Store authority.

    Rows retain exact checkpoint and Artifact references without loading their
    payloads. ``add_row`` and ``update_row`` change only local history; call
    :meth:`publish` to reapply those idempotent changes to the Store-local current
    history snapshot through U2 compare-and-set authority.
    """

    def __init__(self, experiment: Ref[ObjectRef]) -> None:
        if not isinstance(experiment, ObjectRef):
            raise TypeError("ExperimentData requires an Experiment ObjectRef projection.")
        self.experiment = experiment
        self._columns: tuple[str, ...] = ()
        self._rows: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._pending_operations: list[tuple[str, object]] = []

    @staticmethod
    def scalar_column(artifact: str, field: str | None = None) -> str:
        """Return a collision-safe scalar column name for one Artifact result.

        Args:
            artifact: Non-empty configured Artifact name.
            field: Optional top-level public result-map field.

        Returns:
            The escaped scalar column name, escaping backslashes and dots in each
            component before joining components with one dot.

        Raises:
            ExperimentDataError: If a component is invalid or collides with a
            built-in history column.
        """

        def escape(component: str) -> str:
            return _name(component, "Artifact scalar component").replace("\\", "\\\\").replace(".", "\\.")

        result = escape(artifact) if field is None else f"{escape(artifact)}.{escape(field)}"
        if result in _RESERVED_COLUMNS:
            raise ExperimentDataError(f"Artifact scalar column {result!r} is reserved.")
        return result

    @property
    def data(self):
        """Return a recursively detached pandas DataFrame of the retained rows.

        Missing scalar values become ``pd.NA`` while explicit null values remain
        ``None``. Importing pandas happens only when this property is accessed.
        Mutating returned nested mappings or containers cannot alter this Object.

        Returns:
            A newly constructed pandas ``DataFrame`` with exact StateRef cells.

        Raises:
            ImportError: If the required pandas dependency is unavailable.

        Side Effects:
            Imports pandas on first analysis access only; retained rows are not
            mutated or published.
        """

        import pandas as pd

        records = []
        for row in self._rows.values():
            facts = row["facts"]
            record = {
                name: (pd.NA if facts[name] is _MISSING else copy.deepcopy(facts[name]))
                for name in _FACT_NAMES
            }
            record.update({
                "expected_artifacts": list(row["expected_artifacts"]),
                "artifact_inputs": dict(row["artifact_inputs"]),
                "eval_artifacts": dict(row["eval_artifacts"]),
                "evaluation_status": row["evaluation_status"],
                "failed_artifact": row["failed_artifact"],
            })
            for name in self._columns:
                value = row["scalars"].get(name, _MISSING)
                record[name] = pd.NA if value is _MISSING else copy.deepcopy(value)
            records.append(record)
        columns = [
            *_FACT_NAMES,
            "expected_artifacts",
            "artifact_inputs",
            "eval_artifacts",
            "evaluation_status",
            "failed_artifact",
            *self._columns,
        ]
        return pd.DataFrame.from_records(records, columns=columns)

    @classmethod
    def _definition(cls, experiment: ObjectRef):
        return cls.defn(Ref(experiment)).concretize()

    @classmethod
    def _load_current(cls, experiment: ObjectRef, *, repo, store):
        cdef = cls._definition(experiment)
        evidence = repo.reference_evidence(cdef)
        references = {item.object_ref for item in evidence.declarations}
        references.update(item.state_ref.object for item in evidence.states)
        if not references:
            return None, None, cdef
        if len(references) != 1:
            raise StoreAuthorityError("ExperimentData identity authority is ambiguous.")
        reference = next(iter(references))
        try:
            current = repo.resolve_state_alias(reference, _ALIAS, store=store)
        except KeyError as error:
            raise StoreAuthorityError("ExperimentData identity exists without a current history alias.") from error
        loaded = repo.load_state_ref(current, reuse_live="never", cache="none", source_store=store)
        if not isinstance(loaded, cls) or loaded.experiment != experiment:
            raise StoreAuthorityError("Current ExperimentData alias has incompatible history authority.")
        return loaded, current, cdef

    @classmethod
    def find(cls, experiment: Ref[ObjectRef], *, repo, store=None):
        """Load the Store-current history for one projected Experiment subject.

        Args:
            experiment: Default-policy projected Experiment ObjectRef.
            repo: Connected Repo holding declaration and state authority.
            store: Optional selected Store. It is required for ambiguous Store
                topology and restricts loading to that authority.

        Returns:
            A fresh uncached ExperimentData object, or ``None`` only when no
            matching declaration or snapshot identity exists.

        Raises:
            ExperimentDataError: If the projected subject is invalid.
            StoreAuthorityError: If matching history is incomplete, corrupt, or
            ambiguous. Repo load errors remain explicit.
        """

        if not isinstance(experiment, ObjectRef):
            raise TypeError("ExperimentData.find requires an Experiment ObjectRef projection.")
        if store is None:
            store = repo._selected_writable_physical_store(None, "find ExperimentData")
        loaded, _, _ = cls._load_current(experiment, repo=repo, store=store)
        return loaded

    @classmethod
    def get_or_create(cls, experiment: Ref[ObjectRef], *, repo, store=None):
        """Return the unique current history, creating or safely recovering it.

        A new history uses the U2 declaration claim and absent-alias CAS. Recovery
        can adopt only one validated empty initial snapshot; a populated or
        ambiguous alias-less history remains an authority error.

        Args:
            experiment: Default-policy projected Experiment ObjectRef.
            repo: Connected Repo holding or receiving history authority.
            store: Optional selected writable Store. Omit it only for an
                unambiguous writable physical Store topology.

        Returns:
            A fresh or newly initialized current ExperimentData object.

        Raises:
            TypeError: If ``experiment`` is not a projected ObjectRef.
            StoreAuthorityError: If initialization evidence is incomplete,
                populated, malformed, or ambiguous.
            RepoSaveError: If Store selection or initial immutable publication
                cannot complete.

        Side Effects:
            May declare/build an Object, publish its empty immutable snapshot, or
            CAS-adopt a single recovered initial snapshot in the selected Store.
        """

        if not isinstance(experiment, ObjectRef):
            raise TypeError("ExperimentData.get_or_create requires an Experiment ObjectRef projection.")
        selected = repo._selected_writable_physical_store(store, "create ExperimentData")
        cdef = cls._definition(experiment)
        try:
            loaded, _, _ = cls._load_current(experiment, repo=repo, store=selected)
        except StoreAuthorityError as error:
            if "without a current history alias" not in str(error):
                raise
            loaded = None
        if loaded is not None:
            return loaded

        reference = repo.get_or_declare_object_ref(cdef, store=selected)
        try:
            current = repo.resolve_state_alias(reference, _ALIAS, store=selected)
        except KeyError:
            current = None
        if current is not None:
            return repo.load_state_ref(current, reuse_live="never", cache="none", source_store=selected)

        evidence = repo.reference_evidence(cdef)
        states = tuple(item.state_ref for item in evidence.states if item.state_ref.object == reference)
        if states:
            if len(states) != 1:
                raise StoreAuthorityError("ExperimentData recovery found multiple alias-less snapshots.")
            recovered = repo.load_state_ref(states[0], reuse_live="never", cache="none", source_store=selected)
            if not isinstance(recovered, cls) or recovered.experiment != experiment or recovered._rows:
                raise StoreAuthorityError("Only one valid empty ExperimentData snapshot may be adopted.")
            try:
                selected.compare_and_set_state_alias(
                    StateAliasRecord(_ALIAS, reference, states[0].digest()),
                    expected_state_ref_digest=None,
                )
                selected.commit()
            except StoreAliasConflictError:
                winner = repo.resolve_state_alias(reference, _ALIAS, store=selected)
                return repo.load_state_ref(winner, reuse_live="never", cache="none", source_store=selected)
            winner = repo.resolve_state_alias(reference, _ALIAS, store=selected)
            if winner != states[0]:
                raise StoreAuthorityError("ExperimentData initialization alias changed unexpectedly.")
            return recovered

        created = repo.build_object_ref(reference, store=selected)
        try:
            repo.save_object_if_current(created, alias=_ALIAS, expected=None, store=selected)
        except StoreAliasConflictError:
            winner = repo.resolve_state_alias(reference, _ALIAS, store=selected)
            return repo.load_state_ref(winner, reuse_live="never", cache="none", source_store=selected)
        return created

    def add_row(
            self, *, expected_artifacts: Sequence[str], state_ref: Ref[StateRef],
            prev_state_ref: Ref[StateRef | None], time: int, examples_seen: int,
            parameters=None, dataset_size: Cardinality = Cardinality.UNKNOWN,
            training_loss: float | None = None, eval_artifacts: Mapping[str, StateRef] | None = None,
            scalar_values: Mapping[str, object] | None = None,
            evaluation_status: Literal["pending", "completed", "failed"] = "pending",
            row_key: str | None = None, prev_row_key: str | None = None) -> str:
        """Append one immutable checkpoint fact row or idempotently reuse its key.

        ``state_ref`` and an optional predecessor must project to this history's
        Experiment subject. Standalone calls generate a UUID key; callback callers
        supply a durable occurrence key to make retries idempotent.

        Args:
            expected_artifacts: Ordered, unique configured Artifact names.
            state_ref: Exact checkpoint StateRef described by the row.
            prev_state_ref: Optional exact predecessor StateRef.
            time: Non-negative observation timestamp in milliseconds.
            examples_seen: Non-negative retained training exposure.
            parameters: Optional object exposing non-negative ``total`` and
                ``trainable`` counts.
            dataset_size: Effective training cardinality.
            training_loss: Optional finite numeric loss observation.
            eval_artifacts: Optional completed Artifact references.
            scalar_values: Optional supported scalar result mapping.
            evaluation_status: Pending, completed, or failed evaluation state.
            row_key: Optional durable occurrence key.
            prev_row_key: Optional preceding occurrence key.

        Returns:
            The new or idempotently matched opaque row key.

        Raises:
            ExperimentDataError: If facts, references, values, status, or limits
                violate v1, or an existing key has conflicting immutable facts.

        Side Effects:
            Changes only this live Object and queues an idempotent publication
            operation; it does not write Store authority.
        """

        key = str(uuid4()) if row_key is None else _name(row_key, "row_key")
        expected = tuple(_name(name, "Artifact name") for name in expected_artifacts)
        if len(expected) != len(set(expected)) or len(expected) > _MAX_ARTIFACTS:
            raise ExperimentDataError("expected_artifacts must be ordered, unique, and within the row limit.")
        state = _require_state_ref(state_ref, "state_ref")
        previous = _require_state_ref(prev_state_ref, "prev_state_ref") if prev_state_ref is not None else None
        self._validate_association(state, "state_ref")
        if previous is not None:
            self._validate_association(previous, "prev_state_ref")
        if prev_row_key is not None:
            prev_row_key = _name(prev_row_key, "prev_row_key")
        total, trainable = self._parameters(parameters)
        dataset_kind, dataset_value = self._dataset_size(dataset_size)
        facts = {
            "row_key": key, "prev_row_key": prev_row_key if prev_row_key is not None else _MISSING,
            "state_ref": state, "prev_state_ref": previous, "time": _counter(time, "time"),
            "examples_seen": _counter(examples_seen, "examples_seen"),
            "total_parameters": total, "trainable_parameters": trainable,
            "dataset_size_kind": dataset_kind, "dataset_size": dataset_value,
            "training_loss": _MISSING if training_loss is None else _scalar(training_loss, "training_loss"),
        }
        if facts["training_loss"] is not _MISSING and type(facts["training_loss"]) not in {int, float}:
            raise ExperimentDataError("training_loss must be a finite numeric value or missing.")
        row = self._new_row(
            facts, expected, (), _mapping(eval_artifacts, expected, "eval_artifacts"),
            evaluation_status, None, self._scalars(scalar_values),
        )
        existing = self._rows.get(key)
        if existing is not None:
            if not self._same_facts(existing, row):
                raise ExperimentDataError(f"row_key {key!r} conflicts with immutable checkpoint facts.")
            return key
        self._install_row(key, row)
        self._pending_operations.append(("add", _copy(row)))
        return key

    def update_row(
            self, row_key: str, *, artifact_inputs: Mapping[str, StateRef] | None = None,
            eval_artifacts: Mapping[str, StateRef], scalar_values: Mapping[str, object],
            evaluation_status: Literal["pending", "completed", "failed"],
            failed_artifact: str | None = None) -> None:
        """Merge Artifact inputs/results into an existing checkpoint row.

        Identical reference and scalar retries are no-ops. Conflicting completed
        values and invalid status transitions fail before local history changes.

        Args:
            row_key: Existing opaque checkpoint occurrence key.
            artifact_inputs: Optional ordered initial Artifact StateRef mapping.
            eval_artifacts: Ordered completed Artifact StateRef mapping.
            scalar_values: Supported scalar result mapping.
            evaluation_status: Requested pending, failed, or completed status.
            failed_artifact: Required expected incomplete name for failed status.

        Returns:
            ``None`` after an in-memory merge.

        Raises:
            KeyError: If ``row_key`` is absent.
            ExperimentDataError: If mappings, scalar values, completed evidence,
                or status transition conflict with the retained row.

        Side Effects:
            Changes only this live Object and queues a replayable publication
            operation; initial Artifact inputs remain distinct from results.
        """

        key = _name(row_key, "row_key")
        existing = self._rows.get(key)
        if existing is None:
            raise KeyError(f"ExperimentData has no row {key!r}.")
        candidate = _copy(existing)
        expected = candidate["expected_artifacts"]
        for field, supplied in (("artifact_inputs", artifact_inputs), ("eval_artifacts", eval_artifacts)):
            incoming = _mapping(supplied, expected, field)
            values = dict(candidate[field])
            for name, reference in incoming:
                if name in values and values[name] != reference:
                    raise ExperimentDataError(f"{field} cannot replace existing Artifact evidence for {name!r}.")
                values[name] = reference
            candidate[field] = tuple((name, values[name]) for name in expected if name in values)
        incoming_scalars = self._scalars(scalar_values)
        for name, value in incoming_scalars.items():
            old = candidate["scalars"].get(name, _MISSING)
            if old is not _MISSING and old != value:
                raise ExperimentDataError(f"Scalar result {name!r} cannot replace an existing value.")
            candidate["scalars"][name] = value
        if candidate["evaluation_status"] == "completed" and evaluation_status != "completed":
            raise ExperimentDataError("Completed rows cannot return to a non-completed status.")
        candidate["evaluation_status"] = evaluation_status
        candidate["failed_artifact"] = failed_artifact
        self._validate_row(candidate)
        self._install_row(key, candidate)
        self._pending_operations.append(("update", key, _copy(candidate)))

    def publish(self, *, repo, store=None):
        """CAS-publish queued row changes and return the resulting StateRef.

        Fresh history is loaded for each attempt, queued idempotent mutations are
        reapplied, and U2's current-alias CAS publishes one immutable snapshot.
        At most eight stale-writer retries are attempted; failures retain local
        changes for caller inspection or retry.

        Args:
            repo: Connected Repo used for fresh load, immutable save, and CAS.
            store: Optional explicit writable Store authority.

        Returns:
            The exact newly current ExperimentData StateRef.

        Raises:
            StoreAliasConflictError: If eight stale-writer attempts lose CAS.
            StoreAuthorityError: If current history authority disappears or is
                malformed.
            RepoSaveError: If immutable publication fails for another reason.

        Side Effects:
            Can publish immutable orphan snapshots on losing CAS; successful calls
            replace only the selected Store's current history alias.
        """

        selected = repo._selected_writable_physical_store(store, "publish ExperimentData")
        operations = _copy(self._pending_operations)
        if not operations:
            if self.last_state_ref is None:
                raise RepoSaveError("ExperimentData has no current state to publish.")
            return self.last_state_ref
        for _ in range(8):
            current = type(self).find(self.experiment, repo=repo, store=selected)
            if current is None or current.last_state_ref is None:
                raise StoreAuthorityError("ExperimentData current authority disappeared during publication.")
            try:
                for operation in operations:
                    current._apply_operation(operation)
                result = repo.save_object_if_current(
                    current, alias=_ALIAS, expected=current.last_state_ref, store=selected,
                )
            except StoreAliasConflictError:
                continue
            except RepoSaveError as error:
                # Independent Repo handles in one interpreter can contend on the
                # same temporary live-graph reservation before Store CAS. Reload
                # and reapply rather than turning that stale local view into loss.
                if "already reserved" in str(error):
                    continue
                raise
            self._columns = current._columns
            self._rows = current._rows
            self._pending_operations.clear()
            self._last_state_ref = result
            return result
        raise StoreAliasConflictError("ExperimentData publication exceeded eight stale-writer retries.")

    def _apply_operation(self, operation):
        if operation[0] == "add":
            row = operation[1]
            key = row["facts"]["row_key"]
            existing = self._rows.get(key)
            if existing is None:
                self._install_row(key, _copy(row))
            elif not self._same_facts(existing, row):
                raise ExperimentDataError(f"row_key {key!r} conflicts with immutable checkpoint facts.")
        else:
            _, key, target = operation
            existing = self._rows.get(key)
            if existing is None or not self._same_facts(existing, target):
                raise ExperimentDataError(f"Cannot apply history update for missing or conflicting row {key!r}.")
            self._merge_target_row(key, target)

    def _merge_target_row(self, key, target):
        current = self._rows[key]
        for field in ("artifact_inputs", "eval_artifacts"):
            values = dict(current[field])
            for name, reference in target[field]:
                if name in values and values[name] != reference:
                    raise ExperimentDataError(f"{field} cannot replace existing Artifact evidence for {name!r}.")
                values[name] = reference
            current[field] = tuple((name, values[name]) for name in current["expected_artifacts"] if name in values)
        for name, value in target["scalars"].items():
            old = current["scalars"].get(name, _MISSING)
            if old is not _MISSING and old != value:
                raise ExperimentDataError(f"Scalar result {name!r} cannot replace an existing value.")
            current["scalars"][name] = value
        if current["evaluation_status"] == "completed" and target["evaluation_status"] != "completed":
            raise ExperimentDataError("Completed rows cannot return to a non-completed status.")
        current["evaluation_status"] = target["evaluation_status"]
        current["failed_artifact"] = target["failed_artifact"]
        self._validate_row(current)
        self._install_row(key, current)

    def _validate_association(self, reference: StateRef, what: str):
        if reference.object_projection() != self.experiment:
            raise ExperimentDataError(f"{what} does not project to this ExperimentData subject.")

    @staticmethod
    def _parameters(parameters):
        if parameters is None:
            return _MISSING, _MISSING
        total = getattr(parameters, "total", None)
        trainable = getattr(parameters, "trainable", None)
        total, trainable = _counter(total, "total_parameters"), _counter(trainable, "trainable_parameters")
        if trainable > total:
            raise ExperimentDataError("trainable_parameters cannot exceed total_parameters.")
        return total, trainable

    @staticmethod
    def _dataset_size(value):
        if not isinstance(value, Cardinality):
            raise ExperimentDataError("dataset_size must be a Cardinality.")
        if value.kind is CardinalityKind.FINITE:
            return "finite", _counter(value.value, "dataset_size")
        if value.kind is CardinalityKind.UNKNOWN:
            return "unknown", _MISSING
        return "infinite", _MISSING

    def _scalars(self, values):
        if values is None:
            return OrderedDict()
        if not isinstance(values, Mapping):
            raise ExperimentDataError("scalar_values must be a mapping.")
        result = OrderedDict()
        for name, value in values.items():
            name = _name(name, "scalar column")
            if name in _RESERVED_COLUMNS:
                raise ExperimentDataError(f"Scalar column {name!r} is reserved.")
            result[name] = _scalar(value, f"scalar column {name!r}")
        return result

    def _new_row(self, facts, expected, inputs, results, status, failed, scalars):
        row = {
            "facts": facts, "expected_artifacts": expected, "artifact_inputs": tuple(inputs),
            "eval_artifacts": tuple(results), "evaluation_status": status,
            "failed_artifact": failed, "scalars": OrderedDict(scalars),
        }
        self._validate_row(row)
        return row

    def _validate_row(self, row):
        facts = row["facts"]
        if set(facts) != set(_FACT_NAMES):
            raise ExperimentDataError("Row facts must contain exactly the v1 built-ins.")
        _name(facts["row_key"], "row_key")
        if facts["prev_row_key"] is not _MISSING:
            _name(facts["prev_row_key"], "prev_row_key")
        self._validate_association(_require_state_ref(facts["state_ref"], "state_ref"), "state_ref")
        if facts["prev_state_ref"] is not None:
            self._validate_association(_require_state_ref(facts["prev_state_ref"], "prev_state_ref"), "prev_state_ref")
        _counter(facts["time"], "time")
        _counter(facts["examples_seen"], "examples_seen")
        for name in ("total_parameters", "trainable_parameters", "dataset_size", "training_loss"):
            _scalar(facts[name], name)
        if facts["total_parameters"] is not _MISSING:
            _counter(facts["total_parameters"], "total_parameters")
            _counter(facts["trainable_parameters"], "trainable_parameters")
            if facts["trainable_parameters"] > facts["total_parameters"]:
                raise ExperimentDataError("trainable_parameters cannot exceed total_parameters.")
        if facts["dataset_size_kind"] not in {"finite", "unknown", "infinite"}:
            raise ExperimentDataError("dataset_size_kind is invalid.")
        if facts["dataset_size_kind"] == "finite":
            _counter(facts["dataset_size"], "dataset_size")
        elif facts["dataset_size"] is not _MISSING:
            raise ExperimentDataError("Only finite dataset_size_kind may include dataset_size.")
        if facts["training_loss"] is not _MISSING and type(facts["training_loss"]) not in {int, float}:
            raise ExperimentDataError("training_loss must be numeric or missing.")
        expected = row["expected_artifacts"]
        if len(expected) != len(set(expected)) or len(expected) > _MAX_ARTIFACTS:
            raise ExperimentDataError("expected_artifacts is invalid.")
        for name in expected:
            _name(name, "Artifact name")
        for field in ("artifact_inputs", "eval_artifacts"):
            pairs = row[field]
            names = tuple(name for name, _ in pairs)
            if names != tuple(name for name in expected if name in names):
                raise ExperimentDataError(f"{field} must use expected Artifact declaration order.")
            for name, reference in pairs:
                _require_state_ref(reference, field)
        status = row["evaluation_status"]
        completed = {name for name, _ in row["eval_artifacts"]}
        failed = row["failed_artifact"]
        if status not in {"pending", "completed", "failed"}:
            raise ExperimentDataError("evaluation_status is invalid.")
        if status == "completed" and (completed != set(expected) or failed is not None):
            raise ExperimentDataError("Completed rows require every expected Artifact and no failed_artifact.")
        if status == "failed":
            if failed not in expected or failed in completed:
                raise ExperimentDataError("Failed rows require one expected, incomplete failed_artifact.")
        elif failed is not None:
            raise ExperimentDataError("Only failed rows may retain failed_artifact.")
        for name, value in row["scalars"].items():
            if name in _RESERVED_COLUMNS:
                raise ExperimentDataError(f"Scalar column {name!r} is reserved.")
            _name(name, "scalar column")
            _scalar(value, f"scalar column {name!r}")

    def _install_row(self, key, row):
        columns = tuple(OrderedDict.fromkeys((*self._columns, *row["scalars"])))
        if len(_COLUMN_KINDS) + len(columns) > _MAX_COLUMNS:
            raise ExperimentDataError("ExperimentData exceeds the 1,024-column v1 limit.")
        self._rows[key] = row
        self._columns = columns

    @staticmethod
    def _same_facts(left, right):
        return (
            left["facts"] == right["facts"]
            and left["expected_artifacts"] == right["expected_artifacts"]
        )

    def _payload(self):
        columns = [
            {"name": name, "kind": kind}
            for name, kind in _COLUMN_KINDS.items()
        ] + [{"name": name, "kind": "scalar"} for name in self._columns]
        rows = []
        for row in self._rows.values():
            facts = row["facts"]
            encoded_facts = {
                name: _encode_ref(facts[name]) if name == "state_ref"
                else (None if name == "prev_state_ref" and facts[name] is None
                      else _encode_ref(facts[name]) if name == "prev_state_ref"
                      else _encode_cell(facts[name]))
                for name in _FACT_NAMES
            }
            rows.append({
                "facts": encoded_facts,
                "expected_artifacts": list(row["expected_artifacts"]),
                "artifact_inputs": [
                    {"name": name, "state_ref": _reference_to_json(reference.to_data())}
                    for name, reference in row["artifact_inputs"]
                ],
                "eval_artifacts": [
                    {"name": name, "state_ref": _reference_to_json(reference.to_data())}
                    for name, reference in row["eval_artifacts"]
                ],
                "evaluation_status": row["evaluation_status"],
                "failed_artifact": row["failed_artifact"],
                "scalars": [
                    {"name": name, "value": _encode_cell(value)}
                    for name, value in row["scalars"].items()
                ],
            })
        return {"format": _FORMAT, "version": _VERSION, "columns": columns, "rows": rows}

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Write the closed, schema-controlled v1 JSON history payload.

        Args:
            dest_dir: Framework-provided empty payload directory.
            codec: Accepted Object state-codec marker; v1 payload semantics are
                fixed independently of this marker.

        Raises:
            ExperimentDataError: If rows or encoded bytes exceed v1 limits.

        Side Effects:
            Creates ``experiment_data.json`` in ``dest_dir``; no pandas serializer
            or referenced checkpoint payload is used.
        """

        raw = json.dumps(self._payload(), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(raw) > _MAX_BYTES:
            raise ExperimentDataError("ExperimentData payload exceeds the 64 MiB v1 limit.")
        Path(dest_dir, _FILENAME).write_bytes(raw)

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Decode and validate a v1 history payload before replacing local rows.

        Args:
            src_dir: Framework payload directory containing ``experiment_data.json``.
            codec: Accepted Object state-codec marker.

        Raises:
            ExperimentDataError: If JSON, exact references, bounds, or row status
                invariants violate the closed v1 grammar.

        Side Effects:
            Replaces retained rows only after the entire payload validates; failed
            decoding leaves this Object's existing in-memory history unchanged.
        """

        data = _json_loads(Path(src_dir, _FILENAME).read_bytes())
        columns, rows = self._decode_payload(data)
        self._columns = columns
        self._rows = rows
        self._pending_operations = []

    def _decode_payload(self, data):
        if not isinstance(data, Mapping) or set(data) != {"format", "version", "columns", "rows"}:
            raise ExperimentDataError("ExperimentData envelope must contain exactly format, version, columns, and rows.")
        if data["format"] != _FORMAT or type(data["version"]) is not int or data["version"] != _VERSION:
            raise ExperimentDataError("Unsupported ExperimentData payload format or version.")
        if not isinstance(data["columns"], list) or not isinstance(data["rows"], list):
            raise ExperimentDataError("ExperimentData columns and rows must be arrays.")
        if len(data["columns"]) > _MAX_COLUMNS or len(data["rows"]) > _MAX_ROWS:
            raise ExperimentDataError("ExperimentData payload exceeds a v1 table bound.")
        names = []
        dynamic = []
        for entry in data["columns"]:
            if not isinstance(entry, Mapping) or set(entry) != {"name", "kind"}:
                raise ExperimentDataError("Column declarations require exactly name and kind.")
            name = _name(entry["name"], "column name")
            kind = entry["kind"]
            expected = _COLUMN_KINDS.get(name, "scalar")
            if kind != expected or (name in _COLUMN_KINDS and kind == "scalar"):
                raise ExperimentDataError("Column declaration has an invalid v1 kind.")
            names.append(name)
            if kind == "scalar":
                dynamic.append(name)
        if len(names) != len(set(names)) or tuple(name for name in names if name in _COLUMN_KINDS) != tuple(_COLUMN_KINDS):
            raise ExperimentDataError("Columns must declare every built-in once in v1 order.")
        if any(name in _RESERVED_COLUMNS for name in dynamic):
            raise ExperimentDataError("A scalar column collides with a reserved history field.")
        decoded = OrderedDict()
        for encoded in data["rows"]:
            row = self._decode_row(encoded, tuple(dynamic))
            key = row["facts"]["row_key"]
            if key in decoded:
                raise ExperimentDataError(f"ExperimentData payload repeats row_key {key!r}.")
            decoded[key] = row
        return tuple(dynamic), decoded

    def _decode_row(self, encoded, columns):
        fields = {
            "facts", "expected_artifacts", "artifact_inputs", "eval_artifacts",
            "evaluation_status", "failed_artifact", "scalars",
        }
        if not isinstance(encoded, Mapping) or set(encoded) != fields:
            raise ExperimentDataError("ExperimentData rows require exactly the v1 row fields.")
        raw_facts = encoded["facts"]
        if not isinstance(raw_facts, Mapping) or set(raw_facts) != set(_FACT_NAMES):
            raise ExperimentDataError("ExperimentData facts require exactly the v1 fact fields.")
        facts = {
            name: _decode_ref(raw_facts[name]) if name == "state_ref"
            else _decode_ref(raw_facts[name], nullable=True) if name == "prev_state_ref"
            else _decode_cell(raw_facts[name])
            for name in _FACT_NAMES
        }
        expected_raw = encoded["expected_artifacts"]
        if not isinstance(expected_raw, list):
            raise ExperimentDataError("expected_artifacts must be an ordered array.")
        expected = tuple(_name(name, "Artifact name") for name in expected_raw)
        if len(expected) != len(set(expected)) or len(expected) > _MAX_ARTIFACTS:
            raise ExperimentDataError("expected_artifacts is invalid.")
        inputs = self._decode_mapping(encoded["artifact_inputs"], expected, "artifact_inputs")
        results = self._decode_mapping(encoded["eval_artifacts"], expected, "eval_artifacts")
        scalars = OrderedDict()
        if not isinstance(encoded["scalars"], list):
            raise ExperimentDataError("scalars must be an ordered array.")
        for entry in encoded["scalars"]:
            if not isinstance(entry, Mapping) or set(entry) != {"name", "value"}:
                raise ExperimentDataError("Scalar mapping entries require name and value.")
            name = _name(entry["name"], "scalar column")
            if name not in columns or name in scalars:
                raise ExperimentDataError("Scalar mapping has an unknown or duplicate column.")
            scalars[name] = _decode_cell(entry["value"])
        row = self._new_row(
            facts, expected, inputs, results, encoded["evaluation_status"],
            encoded["failed_artifact"], scalars,
        )
        return row

    @staticmethod
    def _decode_mapping(value, expected, what):
        if not isinstance(value, list) or len(value) > _MAX_ARTIFACTS:
            raise ExperimentDataError(f"{what} must be an ordered bounded array.")
        result = []
        for entry in value:
            if not isinstance(entry, Mapping) or set(entry) != {"name", "state_ref"}:
                raise ExperimentDataError(f"{what} entries require exactly name and state_ref.")
            name = _name(entry["name"], "Artifact name")
            try:
                reference = StateRef.from_data(_reference_from_json(entry["state_ref"]))
            except (TypeError, ValueError) as error:
                raise ExperimentDataError(f"{what} contains a malformed StateRef.") from error
            result.append((name, reference))
        names = tuple(name for name, _ in result)
        if len(names) != len(set(names)) or names != tuple(name for name in expected if name in names):
            raise ExperimentDataError(f"{what} names must be unique expected names in declaration order.")
        return tuple(result)


__all__ = ["ExperimentData", "ExperimentDataError"]
