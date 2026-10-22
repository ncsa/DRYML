"""Detached Store authority facts for query capture and format-4 projection.

Knowledge inventory, stored membership, and metadata-holder eligibility are
kept separate because existing Store authority assigns them different meanings.
The module owns no Repo configuration. Format-4 capture returns detached rows and
byte/cut witnesses for later short revalidation, without payload IO or mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

from ..definition import ConcreteDefinition
from ..reference_values import ObjectRef, StateRef
from ..store.records import (
    DeclarationRecord, DefinitionRecord, ObjectAliasRecord, StateAliasRecord,
    StateRefRecord, StoredRootRecord,
)
from ..store.store import StoreAuthorityError
from .identity import IdentityKey, IdentitySet, SourceEvidence, identity_key
from .model import QueryError
from .relationships import iter_direct_relationships


IdentityValue = ConcreteDefinition | ObjectRef | StateRef


@dataclass(frozen=True, slots=True)
class CapturedStoreFacts:
    """One complete detached Store authority cut and its derived identity facts.

    Record tuples supply complete knowledge, independent membership, metadata
    eligibility, and validated authoritative ObjectId projections. A fresh
    ``capture_token`` identifies every replacement, including metadata-only
    enrichment; it is unrelated to a derived SQLite generation. Knowledge is
    derived once per fact table without Store reads or authority writes.
    """

    definitions: tuple[DefinitionRecord, ...]
    stored_roots: tuple[StoredRootRecord, ...]
    declarations: tuple[DeclarationRecord, ...]
    state_refs: tuple[StateRefRecord, ...]
    object_aliases: tuple[ObjectAliasRecord, ...]
    state_aliases: tuple[StateAliasRecord, ...]
    main_definition: ConcreteDefinition | None
    source: SourceEvidence
    metadata: tuple[tuple[IdentityKey, str, object], ...] = ()
    metadata_scopes: frozenset[str] = frozenset()
    capture_token: object = field(init=False, default_factory=object, compare=False, repr=False)
    _reference_projections: dict = field(default_factory=dict, compare=False, repr=False)
    _knowledge: IdentitySet | None = field(init=False, default=None, compare=False, repr=False)
    _metadata_index: dict[tuple[IdentityKey, str], object] = field(
        init=False, compare=False, repr=False,
    )
    _stored_cdefs: frozenset[IdentityKey] = field(init=False, compare=False, repr=False)
    _stored_objects: frozenset[IdentityKey] = field(init=False, compare=False, repr=False)
    _stored_states: frozenset[IdentityKey] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_metadata_index",
            {(key, scope): value for key, scope, value in self.metadata},
        )
        object.__setattr__(
            self, "_stored_cdefs",
            frozenset(identity_key(value) for value in self._authoritative_root_definitions()),
        )
        object.__setattr__(
            self, "_stored_objects",
            frozenset(identity_key(record.object_ref) for record in self.declarations) | frozenset(
                identity_key(record.state_ref.object) for record in self.state_refs
            ),
        )
        object.__setattr__(
            self, "_stored_states",
            frozenset(identity_key(record.state_ref) for record in self.state_refs),
        )

    @classmethod
    def build(
        cls,
        *,
        definitions: Iterable[DefinitionRecord],
        stored_roots: Iterable[StoredRootRecord],
        declarations: Iterable[DeclarationRecord],
        state_refs: Iterable[StateRefRecord],
        object_aliases: Iterable[ObjectAliasRecord],
        state_aliases: Iterable[StateAliasRecord],
        main_definition: ConcreteDefinition | None,
        source: SourceEvidence,
        metadata: Iterable[tuple[IdentityKey, str, object]] = (),
        metadata_scopes: frozenset[str] = frozenset(),
    ) -> "CapturedStoreFacts":
        """Validate cross-record authority and return a detached fact table.

        Record iterables contain validated definitions, markers, declarations,
        snapshots, and aliases. ``main_definition`` is the fenced, resolved main
        target; ``source`` is detached evidence. Optional ``metadata`` triples and
        ``metadata_scopes`` describe fields read in the same cut. Duplicate
        digests, missing or incompatible targets, and authoritative ObjectId
        conflicts raise StoreAuthorityError. No Store access or writes occur.
        """

        definitions = tuple(definitions)
        by_digest = {record.digest: record for record in definitions}
        if len(by_digest) != len(definitions):
            raise StoreAuthorityError("Definition authority has duplicate record digests.")
        stored_roots = tuple(stored_roots)
        for root in stored_roots:
            if root.definition_digest not in by_digest:
                raise StoreAuthorityError("Stored-root authority targets a missing definition record.")
        state_refs = tuple(state_refs)
        state_by_digest = {record.digest: record for record in state_refs}
        if len(state_by_digest) != len(state_refs):
            raise StoreAuthorityError("StateRef authority has duplicate record digests.")
        state_aliases = tuple(state_aliases)
        for alias in state_aliases:
            record = state_by_digest.get(alias.state_ref_digest)
            if record is None or record.state_ref.object != alias.object_ref:
                raise StoreAuthorityError("State alias targets missing or incompatible StateRef authority.")
        declarations = tuple(declarations)
        object_aliases = tuple(object_aliases)
        projections = _validate_object_reference_roots((
            *(record.object_ref for record in declarations),
            *(record.state_ref.object for record in state_refs),
            *(record.object_ref for record in object_aliases),
            *(record.object_ref for record in state_aliases),
        ))
        return cls(
            definitions, stored_roots, declarations, state_refs,
            object_aliases, state_aliases, main_definition, source,
            tuple(metadata), metadata_scopes, projections,
        )

    def with_metadata(
        self, metadata: Iterable[tuple[IdentityKey, str, object]], *, scopes: frozenset[str],
    ) -> "CapturedStoreFacts":
        """Enrich this cut with detached metadata and a fresh capture token.

        ``metadata`` contains captured identity/scope/value triples; ``scopes``
        is the complete requested scope set. Returns a new table sharing already
        validated identity structure. The caller owns fenced reads and failures;
        this method performs no Store access and never revalidates reference roots.
        """

        enriched = replace(self, metadata=tuple(metadata), metadata_scopes=scopes)
        object.__setattr__(enriched, "_knowledge", self._knowledge)
        return enriched

    def knowledge(self) -> IdentitySet:
        """Return cached full represented/derivable knowledge as an IdentitySet.

        The first call expands detached direct relationships; later calls reuse
        that structure and source evidence. It does not read Store records,
        promote embedded identities to stored authority, or change authority.
        """

        if self._knowledge is not None:
            return self._knowledge
        roots: list[IdentityValue] = [record.definition for record in self.definitions]
        roots.extend(record.object_ref for record in self.declarations)
        roots.extend(record.state_ref for record in self.state_refs)
        roots.extend(record.object_ref for record in self.object_aliases)
        roots.extend(record.object_ref for record in self.state_aliases)
        if self.main_definition is not None:
            roots.append(self.main_definition)
        known = IdentitySet((value, self.source) for value in _derive_identities(roots))
        object.__setattr__(self, "_knowledge", known)
        return known

    def is_stored(self, value: IdentityValue) -> bool:
        """Return kind-specific independent stored authority from captured records."""

        if isinstance(value, ConcreteDefinition):
            return identity_key(value) in self._stored_cdefs
        if isinstance(value, ObjectRef):
            return identity_key(value) in self._stored_objects
        if isinstance(value, StateRef):
            return identity_key(value) in self._stored_states
        raise TypeError("Stored membership requires a Query V3 identity value.")

    def holds_metadata(self, value: ObjectRef | StateRef) -> bool:
        """Return existing metadata-holder eligibility without widening storage proof."""

        if isinstance(value, StateRef):
            return self.is_stored(value)
        if not isinstance(value, ObjectRef):
            raise TypeError("Metadata holder checks require an ObjectRef or StateRef.")
        if self.is_stored(value):
            return True
        for record in self.state_refs:
            for path in record.state_ref.object.objects:
                if record.state_ref.object.at(path) == value:
                    return True
        return False

    def captured_metadata(self, value: ObjectRef | StateRef, scope: str):
        """Return one fact captured under this Store cut.

        A missing entry means that the target was not a holder or authority was
        unavailable while fenced; it is deliberately not interpreted as a
        missing metadata field.
        """

        try:
            return self._metadata_index[identity_key(value), scope]
        except KeyError:
            raise KeyError("captured metadata authority is unavailable") from None

    def _authoritative_root_definitions(self) -> Iterable[ConcreteDefinition]:
        by_digest = {record.digest: record.definition for record in self.definitions}
        yield from (by_digest[root.definition_digest] for root in self.stored_roots)
        yield from (record.object_ref.definition for record in self.declarations)
        yield from (record.state_ref.definition for record in self.state_refs)
        yield from (record.object_ref.definition for record in self.object_aliases)
        if self.main_definition is not None:
            yield self.main_definition


def _derive_identities(roots: Iterable[IdentityValue]) -> tuple[IdentityValue, ...]:
    """Expand retained direct relationships without Store reads or side effects."""

    values: dict[object, IdentityValue] = {}
    pending = list(roots)
    while pending:
        value = pending.pop()
        key = identity_key(value)
        if key in values:
            continue
        values[key] = value
        pending.extend(edge.target for edge in iter_direct_relationships(value))
    return tuple(values.values())


def validate_captured_reference_authority(
        facts: Iterable[CapturedStoreFacts]) -> None:
    """Validate complete captured ObjectId mappings without revisiting roots.

    ``facts`` supplies validated Store cuts. Their authoritative projection maps
    are merged, returning None on success and raising QueryError for incompatible
    ObjectId mappings. Embedded references never enter these maps independently.
    This performs no Store access, payload reads, or authority writes.
    """

    by_object_id = {}
    for item in facts:
        for object_id, projected in item._reference_projections.items():
            previous = by_object_id.setdefault(object_id, projected)
            from ..node import Node
            equal = previous.object_projection_equal(projected) if isinstance(previous, Node) else previous == projected
            if not equal:
                raise QueryError("Query V3 object-reference authority conflicts.")


def _validate_object_reference_roots(roots: Iterable[ObjectRef]) -> dict:
    """Return validated projections without promoting embedded references."""

    by_object_id = {}
    for root in roots:
        for path, object_id in root.objects.items():
            projected = root.at(path)
            previous = by_object_id.setdefault(object_id, projected)
            if previous != projected:
                raise StoreAuthorityError(
                    "Object-reference authority contains incompatible ObjectId mappings."
                )
    return by_object_id


# Format-4 projection preparation intentionally consumes framed metadata bytes,
# not the legacy Query V3 inventory above. It is reusable by incremental replay
# and explicit rebuild; neither path reads data/ or constructs an Object.
@dataclass(frozen=True)
class StoreProjection:
    """Detached projection work at an exact Store cut.

    ``cut`` is AuthorityRecord; ``sources`` lists replaced record paths; ``rows``
    contains (family, source path, correlation key, canonical JSON bytes) tuples.
    ``receipts`` contains exact classified MutationIntentRecords. ``witness`` is
    the detached byte/directory capture, revalidated before SQL publication.
    No live graph, hook, codec or callback is carried into the writer fence.
    """

    cut: object
    sources: tuple
    rows: tuple
    receipts: tuple
    witness: object = field(repr=False, compare=False)


def projection_record_type(path):
    """Return the closed format-4 metadata reader for a Store-relative path.

    Unknown paths raise StoreAuthorityError before prepared publication effects.
    State data directories are handled separately, never as framed records.
    """
    import re
    from ..store import records as r
    from ..store import retention as t
    uuid = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    if re.fullmatch(rf"retention/intents/{uuid}\.record", path):
        return t.RetentionIntentRecord
    if path == "retention/pending.record":
        return t.RetentionPendingRecord
    digest = r"[0-9a-f]{64}"
    shard = r"[0-9a-f]{2}"
    patterns = (
        (rf"definitions/blocks/{shard}/{digest}\.record", r.NodeBlockRecord),
        (rf"(?:definitions/roots|graphs)/{shard}/{digest}\.record", r.NodeRootRecord),
        (rf"membership/(?:recipe|saved)/{shard}/{digest}\.record", r.NodeMembershipRecord),
        (rf"declarations/anchors/{shard}/{digest}\.record", r.DeclarationAnchorRecord),
        (rf"declarations/required/{shard}/{digest}\.record", r.DeclarationRequirementRecord),
        (r"names/[^/]+\.record", r.NodeNameRecord),
        (rf"states/{shard}/{digest}/def\.pkl", r.NodeRootRecord),
        (rf"states/{shard}/{digest}/[^/]+/manifest\.record", r.StateVariantManifest),
        (rf"states/{shard}/{digest}/[^/]+/availability\.record", r.StateAvailabilityRecord),
        (rf"states/{shard}/{digest}/[^/]+/grants/{digest}\.record", t.RetentionGrantRecord),
        (rf"states/{shard}/{digest}/[^/]+/removals/{uuid}\.record", t.StateRemovalRecord),
        (rf"metadata/(?:current_object|current_state|lineage|capture)/{shard}/{digest}\.record", r.NodeMetadataRecord),
    )
    for pattern, reader in patterns:
        if re.fullmatch(pattern, path):
            parts = path.split("/")
            if reader is not r.NodeNameRecord:
                shard, digest = ((parts[1], parts[2]) if parts[0] == "states"
                                 else (parts[-2], parts[-1][:-7]))
                if shard != digest[:2]:
                    raise StoreAuthorityError("Projection record has a mismatched digest shard.")
            return reader
    raise StoreAuthorityError("Unsupported query-visible projection record: " + path)


def mutation_projection_paths(records):
    """Return bounded changed metadata paths from classified journal records.

    Child variant/data installs expand only to their manifest and availability
    records. Unknown transition families fail closed rather than imply coverage.
    No Store access or payload IO occurs.
    """
    paths = set()
    for record in records:
        data = record.intent
        if data["status"] == "pending":
            raise StoreAuthorityError("Unclassified mutation cannot certify projection coverage.")
        for row in data["projection"]:
            path = row["path"]
            if row["kind"] in {"variant", "data"}:
                if row["kind"] == "data":
                    if not path.endswith("/data"):
                        raise StoreAuthorityError("Invalid state data transition.")
                    path = path[:-5]
                paths.update((path + "/manifest.record", path + "/availability.record"))
            else:
                paths.add(path)
    for path in paths:
        projection_record_type(path)
    return tuple(sorted(paths))


def validate_node_root_views(selections, prepared, *, validated_views=None, validated_slots=None):
    """Validate named root purposes and reachable state associations once per owner.

    selections contains (NodeRootRecord, decoded Node) pairs authenticated by the
    same immutable block preparation. Optional sets are private pure operation
    memoization, never Store/cut/availability evidence. Raises StoreAuthorityError
    for noncanonical views or invalid state admission without modifying input.
    """
    from ..node import require_saved
    from ..node_views import creation_cdef, object_projection
    validated_views = set() if validated_views is None else validated_views
    validated_slots = set() if validated_slots is None else validated_slots
    for record, root in selections:
        purpose = record.root_kind, record.view_bytes
        if purpose not in validated_views:
            try:
                if record.root_kind in {"construction", "projection"}:
                    canonical = creation_cdef(root) if record.root_kind == "construction" else object_projection(root)
                    if canonical is not root and not root.graph_equal(canonical):
                        raise ValueError("Noncanonical named Node view.")
                elif record.root_kind == "saved":
                    require_saved(root)
            except (TypeError, ValueError) as error:
                raise StoreAuthorityError("Malformed or incompatible named Node root authority.") from error
            validated_views.add(purpose)
        pending = [prepared._slot(root)]
        while pending:
            slot = pending.pop()
            if slot in validated_slots:
                continue
            node = prepared.nodes[slot]
            state = getattr(node, "state", None)
            if state is not None:
                try:
                    state.validate_for(node, codec=state.codec)
                except (TypeError, ValueError) as error:
                    raise StoreAuthorityError("Invalid selected state creation/compatibility evidence.") from error
            validated_slots.add(slot)
            pending.extend(prepared._children[slot])
            if slot in prepared._construction:
                pending.append(prepared._construction[slot])


def capture_store_projection(store, *, rebuild=False):
    """Prepare changed-record replay or an explicit metadata-only Store inventory.

    ``store`` is an open format-4 DirStore; ``rebuild`` permits full metadata
    inventory only when explicitly requested by the maintenance owner. Returns
    StoreProjection. Graph-block decoding/validation is shared once per block and
    runs outside short Store fences. Missing/malformed authority raises
    StoreAuthorityError; changed cuts raise StoreCutChanged. No authority writes,
    payload reads, application capture or restore occur.
    """
    import json
    import os
    from ..store import authority as a, records as r
    from ..store import retention as t
    from ..node_codec import _decode_node_graph_views
    from ..node_prepare import NodePreparation
    from ..node_views import COMPATIBILITY_VIEW

    with store._metadata_read() as captured:
        receipts = store.pending_mutations()
        paths = set(mutation_projection_paths(receipts))
        if rebuild:
            def inventory(relative):
                for name in captured.directory(relative):
                    path = relative + "/" + name
                    full = os.path.join(store.base_dir, path)
                    if name.startswith(".dryml-"):
                        continue
                    if os.path.isdir(full):
                        if name != "data":
                            inventory(path)
                    else:
                        projection_record_type(path)
                        paths.add(path)
            for area in ("definitions", "graphs", "membership", "declarations", "names", "states", "retention", "metadata"):
                inventory(area)

        rows, decoded, blocks, views, creations, anchors = [], {}, {}, {}, {}, {}
        discovery = {}
        root_records = {}

        def emit(family, source, key, value):
            rows.append((family, source, str(key), r._json_bytes(value)))

        def record(path, required=False):
            if path in decoded:
                value = decoded[path]
                if required and value is None:
                    raise StoreAuthorityError("Missing required projection authority: " + path)
                return value
            reader = projection_record_type(path)
            raw = a.raw(store, path)
            if raw is None:
                decoded[path] = None
                if required:
                    raise StoreAuthorityError("Missing required projection authority: " + path)
                return None
            if reader is r.DeclarationAnchorRecord:
                try:
                    value = reader.from_bytes(raw)
                except r.StoreRecordError:
                    value = r.RetiredDeclarationAnchorRecord.from_bytes(raw)
            else:
                value = reader.from_bytes(raw)
            decoded[path] = value
            return value

        def root_view(root):
            path = store._relative(store._node_path("definitions/blocks", root.block_digest))
            block = record(path, True)
            if block.digest != root.block_digest:
                raise StoreAuthorityError("Mismatched projection block address.")
            blocks[path] = block
            views.setdefault(path, {})[root.view_bytes] = root.view
            root_records.setdefault(path, {})[root.root_kind, root.view_bytes] = root

        for path in sorted(paths):
            value = record(path)
            if value is None:
                continue
            emit("record", path, "", value.to_data())
            if isinstance(value, r.NodeMetadataRecord):
                if path != store._relative(store._node_metadata_path(value.scope, value.target_digest)):
                    raise StoreAuthorityError("Mismatched Node metadata projection address.")
                emit("requested_metadata", path, "", value.to_data())
            elif isinstance(value, t.RetentionGrantRecord):
                from ..state_values import StateKey
                parts = path.split("/")
                key = StateKey(parts[2], parts[3])
                if key not in value.keys or parts[-1] != value.address + ".record":
                    raise StoreAuthorityError("Mismatched per-key retention grant address.")
                emit("retention", path, "", {**value.to_data(), "key": r.StateKeyRecord(key).to_data()})
            elif isinstance(value, t.RetentionIntentRecord):
                if (path != "retention/intents/" + value.token + ".record"
                        or value.owner_domain != store.retention_domain):
                    raise StoreAuthorityError("Mismatched retention intent owner/token.")
            elif isinstance(value, t.RetentionPendingRecord):
                # A head can precede its intent during partial publication. Its
                # exact bytes are projected, not certified as adopted roots.
                pass
            elif isinstance(value, t.StateRemovalRecord):
                if (path.split("/")[2:4] != [value.key.cdef_digest, value.key.codec_hash]
                        or path.split("/")[-1] != value.incarnation + ".record"):
                    raise StoreAuthorityError("Mismatched removal incarnation address.")
            elif isinstance(value, r.NodeBlockRecord):
                if path != store._relative(store._node_path("definitions/blocks", value.digest)):
                    raise StoreAuthorityError("Mismatched Node block address.")
                blocks[path] = value
            elif isinstance(value, r.NodeRootRecord):
                root_view(value)
                if path.startswith("states/"):
                    if value.root_kind != "construction" or path.split("/")[2] != value.digest:
                        raise StoreAuthorityError("Mismatched creation projection address.")
                    block_path = store._relative(store._node_path("definitions/blocks", value.block_digest))
                    creations.setdefault(block_path, []).append((path, value))
                elif value.digest != path.split("/")[-1][:-7]:
                    raise StoreAuthorityError("Mismatched root projection address.")
                elif value.root_kind != ("saved" if path.startswith("graphs/") else "recipe"):
                    raise StoreAuthorityError("Mismatched root projection kind.")
            elif isinstance(value, r.NodeMembershipRecord):
                kind, digest = value.root_kind, value.root_digest
                if path != store._relative(store._node_path("membership/" + kind, digest)):
                    raise StoreAuthorityError("Mismatched membership projection address.")
                root = record(store._relative(store._node_path(
                    "graphs" if kind == "saved" else "definitions/roots", digest)), True)
                if root.root_kind != kind or root.digest != digest:
                    raise StoreAuthorityError("Mismatched membership root authority.")
                root_view(root)
                for identity in value.identities:
                    key = store._anchor_key(identity)
                    anchor = record(store._relative(store._node_path("declarations/anchors", key)), True)
                    required = record(store._relative(store._node_path("declarations/required", key)), True)
                    if (anchor.object_id != identity or required.object_id != identity
                            or required.projection_digest != anchor.projection.digest
                            or (kind, digest) not in anchor.sources):
                        raise StoreAuthorityError("Incomplete membership declaration contribution.")
                emit(kind + "_root", path, digest, root.to_data())
                discovery.setdefault(root.block_digest, []).append((path, root))
            elif isinstance(value, r.DeclarationAnchorRecord):
                key = store._anchor_key(value.object_id)
                if path != store._relative(store._node_path("declarations/anchors", key)):
                    raise StoreAuthorityError("Mismatched declaration projection address.")
                required = record(store._relative(store._node_path("declarations/required", key)), True)
                if required.object_id != value.object_id or required.projection_digest != value.projection.digest:
                    raise StoreAuthorityError("Incomplete declaration projection witness.")
                root_view(value.projection)
                anchors.setdefault(value.projection.block_digest, []).append((value.projection.view_bytes, value.object_id))
                for kind, digest in value.sources:
                    membership = record(store._relative(store._node_path("membership/" + kind, digest)), True)
                    if value.object_id not in membership.identities:
                        raise StoreAuthorityError("Incomplete declaration source contribution.")
                emit("declaration", path, key, value.to_data())
            elif isinstance(value, r.DeclarationRequirementRecord):
                key = store._anchor_key(value.object_id)
                if path != store._relative(store._node_path("declarations/required", key)):
                    raise StoreAuthorityError("Mismatched declaration requirement address.")
                anchor = record(store._relative(store._node_path("declarations/anchors", key)), True)
                if anchor.object_id != value.object_id or anchor.projection.digest != value.projection_digest:
                    raise StoreAuthorityError("Missing or incompatible established declaration anchor.")
                root_view(anchor.projection)
                anchors.setdefault(anchor.projection.block_digest, []).append((anchor.projection.view_bytes, value.object_id))
            elif isinstance(value, (r.StateVariantManifest, r.StateAvailabilityRecord)):
                variant = path.rsplit("/", 1)[0]
                if path.split("/")[2:4] != [value.key.cdef_digest, value.key.codec_hash]:
                    raise StoreAuthorityError("Mismatched full-StateKey projection address.")
                header = record("/".join(path.split("/")[:3]) + "/def.pkl", True)
                if header.root_kind != "construction" or header.digest != value.key.cdef_digest:
                    raise StoreAuthorityError("Missing compatible state creation authority.")
                root_view(header)
                if isinstance(value, r.StateVariantManifest):
                    availability = record(variant + "/availability.record", True)
                    if availability.key != value.key:
                        raise StoreAuthorityError("Mismatched state manifest/availability.")
                    emit("manifest", path, "", value.to_data())
                else:
                    manifest = record(variant + "/manifest.record", True)
                    if manifest.key != value.key:
                        raise StoreAuthorityError("Mismatched state manifest/availability.")
                    # Directory presence is evidence, not a payload authentication.
                    present = os.path.isdir(os.path.join(store.base_dir, variant, "data"))
                    emit("availability", path, "", {"record": value.to_data(),
                         "available": present and value.status == "published"})
            elif isinstance(value, r.NodeNameRecord):
                if path != "names/" + value.name + ".record":
                    raise StoreAuthorityError("Mismatched name projection address.")
                membership = record(store._relative(store._node_path("membership/" + value.root_kind, value.root_digest)), True)
                if (membership.root_kind, membership.root_digest) != (value.root_kind, value.root_digest):
                    raise StoreAuthorityError("Mismatched named membership authority.")
                root = record(store._relative(store._node_path(
                    "graphs" if value.root_kind == "saved" else "definitions/roots", value.root_digest)), True)
                if (root.root_kind, root.digest) != (value.root_kind, value.root_digest):
                    raise StoreAuthorityError("Missing or mismatched named root authority.")
                root_view(root)
                emit("name", path, value.name, value.to_data())

        for path, block in blocks.items():
            packet = block.blocks
            selected = list(views.get(path, {}).values())
            if selected:
                packet["views"] = selected
                try:
                    roots, decoded_owner = _decode_node_graph_views(packet, with_owner=True)
                except (TypeError, ValueError) as error:
                    raise StoreAuthorityError("Malformed shared projection graph authority.") from error
                by_view = {r._json_bytes(view): root for view, root in zip(selected, roots)}
                validate_node_root_views(((header, by_view[header.view_bytes])
                                          for header in root_records.get(path, {}).values()), decoded_owner)
                from .relationships import _discovery_rows
                for source, header in discovery.get(block.digest, ()):
                    for ordinal, row in enumerate(_discovery_rows(by_view[header.view_bytes], header)):
                        emit("discovery", source, str(ordinal), row)
                for view_bytes, identity in anchors.get(block.digest, ()):
                    if by_view[view_bytes].object_id != identity:
                        raise StoreAuthorityError("Declaration projection names another ObjectId.")
                headers = creations.get(path, ())
                if headers:
                    prepared = NodePreparation((by_view[header.view_bytes] for _, header in headers),
                                               view=COMPATIBILITY_VIEW)
                    for (source, header), root in zip(headers, prepared.roots):
                        emit("creation", source, header.digest, {**header.to_data(),
                             "compatibility_digest": prepared.digest(root, domain=COMPATIBILITY_VIEW.name)})
            # Postings correlate by authenticated block/table/overlay and typed
            # value path. Root membership joins to its own selected slot only.
            if path not in paths:
                continue
            values = packet["values"]
            for table_key, table in packet["tables"].items():
                for slot, overlay in enumerate(table["overlays"]):
                    prefix = table_key + ":" + str(slot)
                    emit("node", path, prefix, {"table": table_key, "slot": slot, **overlay})
                    def walk(item, location=(), role="materialize", active=frozenset()):
                        if isinstance(item, dict):
                            tag = item.get("tag")
                            if tag == "block":
                                key = item["key"]
                                if key in active:
                                    raise StoreAuthorityError("Cyclic projection value blocks.")
                                walk(values[key], location, role, active | {key})
                                return
                            if tag == "link":
                                role = item["role"]
                            elif tag in {"quote", "selector", "object-ref", "state-ref",
                                         "state-selector", "object-selector", "local-state"}:
                                role = tag
                            key = prefix + ":" + json.dumps(location, separators=(",", ":"))
                            if tag == "node":
                                emit("edge", path, key, {"table": table_key, "owner": slot,
                                     "path": location, "role": role, "target": item["slot"]})
                                return
                            if tag is not None:
                                emit("facet", path, key, {"table": table_key, "owner": slot,
                                     "path": location, "role": role, "value": item})
                            for name, child in item.items():
                                walk(child, location + (("field", name),), role, active)
                        elif isinstance(item, list):
                            for index, child in enumerate(item):
                                walk(child, location + (("index", index),), role, active)
                    walk(values[overlay["value"]])
                    if overlay["construction"] is not None:
                        emit("edge", path, prefix + ":construction", {"table": table_key,
                             "owner": slot, "path": (), "role": "construction",
                             "target": overlay["construction"]["slot"]})
        return StoreProjection(captured.cut, tuple(sorted(paths)), tuple(rows), receipts, captured)
