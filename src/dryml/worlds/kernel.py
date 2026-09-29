"""Scheduled world requirement collection for code analysis targets."""

from __future__ import annotations

from dataclasses import dataclass

from dryml.annotations.collect import (
    _RawAnnotationBudget,
    _annotation_targets as _direct_annotation_targets,
    _class_annotation_targets,
    _method_annotation_targets,
    _reserve_annotation_targets,
)
from dryml.annotations import own_annotations
from dryml.annotations.errors import UnsupportedAnnotationTargetError
from dryml.code import AnalysisKernel, StaticDependencies
from dryml.code.inspection import InspectionCapture, InspectionTarget
from dryml.code.kernels import KernelContext
from dryml.code.static_dependencies import StaticDependenciesKernel
from dryml.code.targets import CodeTarget
from dryml.requirements import (
    RequirementDeclaration,
    RequirementResult,
    RequirementSource,
)

from .combination import _WorldCombiner
from .declarations import WORLD_REQUIREMENT_KEY
from .errors import WorldRequirementError
from .specs import WorldRequirement

_MAX_RAW_ATTACHMENTS = 4096
_MAX_OCCURRENCES = 4096
_MAX_TEXT = 128


@dataclass(frozen=True, slots=True)
class _Occurrence:
    """One owner-decoded carrier retained by occurrence identity."""

    identifier: str
    value: WorldRequirement
    source_ordinal: int


@dataclass(frozen=True, slots=True)
class _DeclarationView:
    """Immutable closed world declaration projection for one snapshot."""

    snapshot_identity: str
    target_occurrences: tuple[tuple[str, tuple[str, ...]], ...]
    occurrences: tuple[_Occurrence, ...]
    sources: tuple[RequirementSource, ...]


def _source_data(source: RequirementSource) -> dict[str, str | None]:
    """Encode one bounded shared source through primitive transport values."""

    return {
        "label": source.label,
        "module": source.module,
        "qualname": source.qualname,
    }


def _source_from_data(data: object) -> RequirementSource:
    """Decode one closed primitive source mapping for a declaration view."""

    if type(data) is not dict or set(data) != {"label", "module", "qualname"}:
        raise WorldRequirementError("world declaration source is invalid")
    try:
        return RequirementSource(**data)
    except Exception:
        raise WorldRequirementError(
            "world declaration source is invalid"
        ) from None


def _annotation_targets(target: CodeTarget) -> tuple[object, ...]:
    """Return static annotation carriers for one normalized live target."""

    if (
        target.info.kind == "callable_instance"
        and target.original is not None
    ):
        try:
            own_annotations(target.original)
        except UnsupportedAnnotationTargetError:
            pass
        else:
            return (target.original,)
    if target.owner is not None and target.info.kind in (
        "bound_method",
        "callable_instance",
        "descriptor",
    ):
        if target.info.name is None:
            raise WorldRequirementError("world method target is invalid")
        return _method_annotation_targets(target.owner, target.info.name)
    value = (
        target.original if target.original is not None else target.descriptor
    )
    if value is None:
        return ()
    if isinstance(value, type):
        return _class_annotation_targets(value)
    return _direct_annotation_targets(value)


def _declarations_for_targets(
    targets: tuple[CodeTarget, ...],
) -> tuple[
    tuple[tuple[int, RequirementDeclaration[WorldRequirement]], ...], ...
]:
    """Collect declarations after one capture-wide raw attachment preflight."""

    sources = tuple(_annotation_targets(target) for target in targets)
    budget = _RawAnnotationBudget(_MAX_RAW_ATTACHMENTS)
    for target_sources in sources:
        _reserve_annotation_targets(target_sources, budget)
    result: list[
        tuple[tuple[int, RequirementDeclaration[WorldRequirement]], ...]
    ] = []
    for target_sources in sources:
        entries: list[tuple[int, RequirementDeclaration[WorldRequirement]]] = (
            []
        )
        seen: set[int] = set()
        for source in target_sources:
            for annotation in own_annotations(source):
                if annotation.key != WORLD_REQUIREMENT_KEY:
                    continue
                if id(annotation) in seen:
                    continue
                seen.add(id(annotation))
                declaration = annotation.value
                if (
                    type(declaration) is not RequirementDeclaration
                    or type(declaration.value) is not WorldRequirement
                ):
                    raise WorldRequirementError(
                        "world requirement annotation is invalid"
                    )
                entries.append((id(annotation), declaration))
        result.append(tuple(entries))
    return tuple(result)


def _view_from_capture(capture: InspectionCapture) -> _DeclarationView:
    """Capture a bounded immutable world view from local associations."""

    if type(capture) is not InspectionCapture:
        raise WorldRequirementError("world inspection capture is invalid")
    snapshot = capture.target.snapshot
    targets: list[CodeTarget] = []
    for record in snapshot.records:
        target = capture.local_target(record.target_id)
        if type(target) is not CodeTarget:
            raise WorldRequirementError(
                "world inspection association is unavailable"
            )
        targets.append(target)
    collected = _declarations_for_targets(tuple(targets))
    occurrences: list[_Occurrence] = []
    sources: list[RequirementSource] = []
    identifiers: dict[int, str] = {}
    mapping: list[tuple[str, tuple[str, ...]]] = []
    for record, declarations in zip(snapshot.records, collected, strict=True):
        occurrence_ids: list[str] = []
        for carrier_id, declaration in declarations:
            identifier = identifiers.get(carrier_id)
            if identifier is None:
                if len(occurrences) >= _MAX_OCCURRENCES:
                    raise WorldRequirementError(
                        "world declaration occurrence limit exceeded"
                    )
                identifier = f"o{len(occurrences):04d}"
                identifiers[carrier_id] = identifier
                sources.append(declaration.source)
                occurrences.append(
                    _Occurrence(
                        identifier, declaration.value, len(sources) - 1
                    )
                )
            occurrence_ids.append(identifier)
        mapping.append((record.target_id, tuple(occurrence_ids)))
    return _DeclarationView(
        snapshot.identity, tuple(mapping), tuple(occurrences), tuple(sources)
    )


def _view_to_data(view: _DeclarationView) -> dict[str, object]:
    """Encode a validated private declaration view using owner value codecs."""

    return {
        "snapshot": view.snapshot_identity,
        "targets": [
            {"id": target_id, "occurrences": list(occurrence_ids)}
            for target_id, occurrence_ids in view.target_occurrences
        ],
        "occurrences": [
            {
                "id": occurrence.identifier,
                "value": occurrence.value.to_data(),
                "source": occurrence.source_ordinal,
            }
            for occurrence in view.occurrences
        ],
        "sources": [_source_data(source) for source in view.sources],
    }


def _view_from_data(snapshot: object, data: object) -> _DeclarationView:
    """Decode a closed transport view and bind it to one detached snapshot."""

    if type(snapshot) is not InspectionTarget:
        raise WorldRequirementError("world inspection target is invalid")
    target = snapshot
    if (
        type(data) is not dict
        or set(data) != {"snapshot", "targets", "occurrences", "sources"}
        or type(data["snapshot"]) is not str
        or data["snapshot"] != target.snapshot.identity
        or type(data["targets"]) is not list
        or type(data["occurrences"]) is not list
        or type(data["sources"]) is not list
    ):
        raise WorldRequirementError("world declaration view is invalid")
    raw_targets = data["targets"]
    raw_occurrences = data["occurrences"]
    raw_sources = data["sources"]
    if (
        len(raw_occurrences) > _MAX_OCCURRENCES
        or len(raw_sources) > _MAX_OCCURRENCES
    ):
        raise WorldRequirementError("world declaration view exceeds limit")
    sources = tuple(_source_from_data(item) for item in raw_sources)
    occurrences: list[_Occurrence] = []
    identifiers: set[str] = set()
    for item in raw_occurrences:
        if (
            type(item) is not dict
            or set(item) != {"id", "value", "source"}
            or type(item["id"]) is not str
            or not item["id"]
            or len(item["id"]) > _MAX_TEXT
            or type(item["source"]) is not int
            or isinstance(item["source"], bool)
            or not 0 <= item["source"] < len(sources)
            or item["id"] in identifiers
        ):
            raise WorldRequirementError(
                "world declaration occurrence is invalid"
            )
        try:
            value = WorldRequirement.from_data(item["value"])
        except Exception:
            raise WorldRequirementError(
                "world declaration occurrence is invalid"
            ) from None
        identifiers.add(item["id"])
        occurrences.append(_Occurrence(item["id"], value, item["source"]))
    expected_ids = tuple(
        record.target_id for record in target.snapshot.records
    )
    if len(raw_targets) != len(expected_ids):
        raise WorldRequirementError("world declaration targets are not closed")
    mappings: list[tuple[str, tuple[str, ...]]] = []
    mapped: set[str] = set()
    for item, expected_id in zip(raw_targets, expected_ids, strict=True):
        if (
            type(item) is not dict
            or set(item) != {"id", "occurrences"}
            or item["id"] != expected_id
            or type(item["occurrences"]) is not list
            or len(item["occurrences"]) > _MAX_OCCURRENCES
            or any(
                type(identifier) is not str or identifier not in identifiers
                for identifier in item["occurrences"]
            )
        ):
            raise WorldRequirementError(
                "world declaration targets are invalid"
            )
        mapping = tuple(item["occurrences"])
        mappings.append((expected_id, mapping))
        mapped.update(mapping)
    if mapped != identifiers:
        raise WorldRequirementError(
            "world declaration occurrences are not closed"
        )
    return _DeclarationView(
        target.snapshot.identity, tuple(mappings), tuple(occurrences), sources
    )


class WorldRequirementsKernel(
    AnalysisKernel[None, RequirementResult[WorldRequirement]]
):
    """Collect and combine world declarations for resolved static targets.

    The public no-argument form collects live targets. Private factories bind a
    validated immutable inspection declaration view for the same ordinary
    scheduler path; they never invoke this kernel directly.
    """

    input_type = type(None)
    output_type = RequirementResult
    requires = (StaticDependenciesKernel,)

    def __init__(self) -> None:
        """Create the public no-argument live collection kernel."""

        self._view: _DeclarationView | None = None

    @classmethod
    def _from_capture(
        cls, capture: InspectionCapture
    ) -> "WorldRequirementsKernel":
        """Bind declarations from coordinator-local live associations."""

        kernel = cls()
        kernel._view = _view_from_capture(capture)
        return kernel

    @classmethod
    def _from_data(
        cls, target: InspectionTarget, data: object
    ) -> "WorldRequirementsKernel":
        """Bind one decoded owner-codec view to its exact snapshot target."""

        kernel = cls()
        kernel._view = _view_from_data(target, data)
        return kernel

    def _snapshot_declarations(
        self,
        target: InspectionTarget,
        dependencies: StaticDependencies,
    ) -> tuple[RequirementDeclaration[WorldRequirement], ...]:
        """Flatten snapshot occurrences after exact view validation."""

        if (
            self._view is None
            or self._view.snapshot_identity != target.snapshot.identity
        ):
            raise WorldRequirementError(
                "world declaration view has wrong snapshot"
            )
        expected = tuple(
            record.target_id for record in target.snapshot.records
        )
        if (
            tuple(item[0] for item in self._view.target_occurrences)
            != expected
        ):
            raise WorldRequirementError("world declaration view is not closed")
        table = {
            occurrence.identifier: occurrence
            for occurrence in self._view.occurrences
        }
        mapping = dict(self._view.target_occurrences)
        declarations: list[RequirementDeclaration[WorldRequirement]] = []
        seen: set[str] = set()
        for dependency in dependencies.targets:
            if (
                type(dependency) is not InspectionTarget
                or dependency.snapshot.identity != target.snapshot.identity
                or dependency.target_id not in mapping
            ):
                raise WorldRequirementError(
                    "world dependency snapshot is invalid"
                )
            for identifier in mapping[dependency.target_id]:
                if identifier not in seen:
                    seen.add(identifier)
                    occurrence = table[identifier]
                    declarations.append(
                        RequirementDeclaration(
                            occurrence.value,
                            source=self._view.sources[
                                occurrence.source_ordinal
                            ],
                        )
                    )
        return tuple(declarations)

    def run(
        self, graph: object, value: None, context: KernelContext
    ) -> RequirementResult[WorldRequirement]:
        """Collect resolved declarations and invoke the existing combiner once.

        Args:
            graph: Scheduler-owned graph for the canonical target.
            value: Required ``None`` caller input.
            context: Scheduler context supplying the canonical target and
                static dependency result.

        Returns:
            The existing empty, valued, or conflict ``RequirementResult``.

        Raises:
            WorldRequirementError: If a live declaration, snapshot view, or
                dependency association is malformed or mismatched.

        Side Effects:
            Live collection reads passive attachments only. It does not bind or
            invoke descriptors, execute targets, or alter domain state.
        """

        dependencies = context.require(StaticDependenciesKernel)
        if type(dependencies) is not StaticDependencies:
            raise WorldRequirementError(
                "world static dependencies are invalid"
            )
        if type(context.target) is InspectionTarget:
            declarations = self._snapshot_declarations(
                context.target, dependencies
            )
        else:
            targets = tuple(dependencies.targets)
            if any(type(target) is not CodeTarget for target in targets):
                raise WorldRequirementError("world live dependency is invalid")
            groups = _declarations_for_targets(targets)
            declarations_list: list[
                RequirementDeclaration[WorldRequirement]
            ] = []
            seen: set[int] = set()
            for group in groups:
                for identifier, declaration in group:
                    if identifier not in seen:
                        seen.add(identifier)
                        declarations_list.append(declaration)
            declarations = tuple(declarations_list)
        from dryml.requirements import combine_requirements

        return combine_requirements(declarations, combiner=_WorldCombiner())


__all__ = ["WorldRequirementsKernel"]
