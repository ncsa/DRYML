from pathlib import Path

import hashlib

import dill
import pytest

from dryml.core import Definition, Object
from dryml.core.store.records import (
    DefinitionRecord, LocalStateManifest, StoreFormatRecord, StoreRecordError,
)


class RecordObject(Object):
    pass


class MappingRecordObject(Object):
    def __init__(self, config):
        self.config = config


def test_definition_record_round_trips_and_recomputes_all_digest_fields():
    record = DefinitionRecord(RecordObject().definition)

    decoded = DefinitionRecord.from_bytes(record.to_bytes())

    assert decoded.digest == record.digest
    assert decoded.definition.graph_equal(record.definition)
    data = record.to_data()
    data["graph_hash"] = "0" * 64
    with pytest.raises(StoreRecordError, match="hash fields"):
        DefinitionRecord.from_data(data)


def test_definition_record_round_trip_preserves_mapping_identity():
    definition = Definition(
        MappingRecordObject,
        config={"batch_size": 32, "seed": 7, "dtype": "float32"},
    ).concretize()

    decoded = DefinitionRecord.from_bytes(DefinitionRecord(definition).to_bytes())

    assert decoded.definition.graph_equal(definition)
    assert decoded.definition.stable_hash() == definition.stable_hash()


def test_record_codecs_reject_unknown_version_and_trailing_bytes():
    payload = StoreFormatRecord().to_bytes()
    with pytest.raises(StoreRecordError, match="trailing"):
        StoreFormatRecord.from_bytes(payload + b"trailing")
    data = StoreFormatRecord().to_data()
    data["version"] = 0
    with pytest.raises(StoreRecordError, match="Unsupported"):
        StoreFormatRecord.from_data(data)


def test_v2_local_state_manifest_fixture_preserves_identity_and_is_eager(tmp_path):
    payload = b"v2 payload"
    digest = hashlib.sha256(payload).hexdigest()
    record = DefinitionRecord(RecordObject().definition)
    data = {
        "schema": "local-state-manifest",
        "version": 2,
        "codec": "Codec1",
        "graph_hash": record.graph_hash,
        "definition_digest": record.digest,
        "definition_file_digest": "0" * 64,
        "local_digest": hashlib.sha256(
            b'dryml-local-state-manifest-v2\0{"codec":"Codec1","files":[["payload",10,"'
            + digest.encode("ascii") + b'"]]}'
        ).hexdigest(),
        "files": [{"path": "payload", "size": len(payload), "digest": digest}],
    }
    fixture = b"DRYML-STORE-RECORD/local-state-manifest/1\n" + dill.dumps(data, protocol=5)

    manifest = LocalStateManifest.from_bytes(fixture)
    payload_path = tmp_path / "payload"
    payload_path.write_bytes(payload)

    assert manifest.version == 2
    assert manifest.deferred_paths == ()
    assert manifest.state_hash == f"Codec1-{data['local_digest']}"
    manifest.validate_payload(tmp_path, defer_payload=True)
    payload_path.write_bytes(b"v2 corrupt")
    with pytest.raises(StoreRecordError, match="exactly match"):
        manifest.validate_payload(tmp_path, defer_payload=True)


def test_json_fact_representation_is_detached_and_conservatively_charged():
    """Owned-size estimation covers all prospective tagged containers and leaves."""
    import sys
    from dryml.formats import canonical_json_load_bytes
    from dryml.core._content_facts import _primitive_size, _freeze_json_fact, _thaw_json_fact

    parsed = canonical_json_load_bytes(b'{"a":[{"nested":[1,true,null,2.5]},"text"],"b":[]}', max_depth=64)
    prepared = _freeze_json_fact(parsed)
    def owned_size(value):
        size = sys.getsizeof(value)
        return size + sum(owned_size(item) for item in value) if type(value) is tuple else size
    assert _primitive_size(parsed) >= owned_size(prepared)
    first, second = _thaw_json_fact(prepared), _thaw_json_fact(prepared)
    assert first == second == parsed
    assert first is not second and first["a"] is not second["a"]
    assert first["a"][0] is not second["a"][0]
    with pytest.raises(TypeError):
        first["a"][0]["nested"] = ()


def test_valid_large_json_uses_uncached_parser_working_memory(tmp_path):
    """Retention bounds are not input rejection or a decoding-memory ceiling."""
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    path = tmp_path / "metadata.json"
    path.write_bytes(b"[" + b",".join([b'"' + b"x" * 100 + b'"'] * 25000) + b"]")
    repo = Repo()
    with repo._read_cut() as cut:
        value = DirStore._read_json(str(path), cut=cut, kind="metadata.json")
        assert len(value) == 25000
        assert repo._validation_evidence.accounted_bytes == 0
    assert not repo._validation_evidence._resident


def _prepared_read(repo, source, record_type=DefinitionRecord):
    from dryml.core.store.records import _RecordPreparation
    with repo._read_cut() as cut:
        return record_type._from_bytes_prepared(source, _RecordPreparation(cut))


def test_prepared_definition_sharing_arrays_roles_and_owned_budget():
    """Warm realization owns mutable storage and regenerates all private tokens."""
    import numpy as np
    import sys
    from dryml.core import Repo
    from dryml.core.cdef_identity import cdef_node_key
    from tests.core.test_cdef_graph_codec import CodecLeaf, CodecParent
    from tests.core.test_exact_load_metadata_evidence import exact_work_counts

    shared = Definition(CodecLeaf, np.arange(6, dtype=np.float64).reshape(2, 3))
    source = DefinitionRecord(Definition(CodecParent, [shared, shared]).concretize()).to_bytes()
    repo = Repo()
    first = _prepared_read(repo, source)
    with exact_work_counts() as counts:
        second = _prepared_read(repo, source)
    assert counts["decode_cdef_graph"] == counts["_graph_and_labels"] == counts["_role_paths"] == 0
    assert counts["_realize_prepared_graph"] == counts["_build_graph_nodes"] == 1
    assert counts["dill_load"] == 2
    assert counts["_graph_content_matches"] > 0
    left, right = second.definition.parameters["children"]
    assert left is right
    assert cdef_node_key(left) is not cdef_node_key(first.definition.parameters["children"][0])
    assert second.definition.cls is not first.definition.cls
    array = left.parameters["value"]
    array.base.setflags(write=True)
    array.setflags(write=True)
    array[:] = 99
    roles = second.roles
    roles[0]["path"]["segments"].append({"kind": "parameter", "name": "wrong"})
    roles[0]["stateful"] = True
    third = _prepared_read(repo, source)
    assert np.array_equal(third.definition.parameters["children"][0].parameters["value"], np.arange(6).reshape(2, 3))
    assert third.roles == first.roles
    def owned(value):
        return sys.getsizeof(value) + (sum(owned(item) for item in value) if type(value) is tuple else 0)
    fact = next(iter(repo._validation_evidence._resident.values()))
    prepared = fact.prepared
    assert prepared.graph is dict(prepared.content[1])[("str", "definition")]
    measured = sum(owned(getattr(prepared, name)) for name in prepared.__slots__)
    assert fact.charge >= measured + sys.getsizeof(source)
    assert repo._validation_evidence.high_water_bytes <= 32 * 1024 * 1024


@pytest.mark.parametrize("damage", ["label", "order", "unreachable", "cycle", "duplicate", "roles", "role-bit", "trailing", "frame"])
def test_prepared_record_warm_malformed_authority_is_not_repaired(damage):
    """Lossless matching never repairs ordering, labels or enclosing role claims."""
    from copy import deepcopy
    from dryml.core import Repo
    from tests.core.test_cdef_graph_codec import CodecLeaf, CodecParent

    source = DefinitionRecord(Definition(CodecParent, [Definition(CodecLeaf, 1)]).concretize()).to_bytes()
    repo = Repo()
    _prepared_read(repo, source)
    data = DefinitionRecord._decode_framed(source)
    graph = data["definition"]
    if damage == "label":
        graph["nodes"][0]["label"] = "n00"
        graph["root"] = "n00"
    elif damage == "order":
        graph["nodes"].reverse()
    elif damage == "unreachable":
        node = deepcopy(graph["nodes"][1])
        node["label"] = "n999"
        graph["nodes"].append(node)
    elif damage == "cycle":
        graph["nodes"][0]["parameters"]["items"][0][1] = {"kind": "cdef", "label": graph["root"]}
    elif damage == "duplicate":
        graph["nodes"].append(deepcopy(graph["nodes"][0]))
    elif damage == "roles":
        data["roles"].reverse()
    elif damage == "role-bit":
        graph["nodes"][0]["stateful_role"] = True
    changed = b"DRYML-STORE-RECORD/definition/1\n" + dill.dumps(data, protocol=5)
    if damage == "trailing":
        changed = source + b"trailing"
    elif damage == "frame":
        changed = source.replace(b"definition/1", b"definition/0", 1)
    residents = tuple(repo._validation_evidence._resident.values())
    with pytest.raises(StoreRecordError):
        _prepared_read(repo, changed)
    assert tuple(repo._validation_evidence._resident.values()) == residents
    assert not repo._validation_evidence._pending


def test_prepared_collision_ties_fresh_dill_and_policy_context(monkeypatch):
    """Same bucket/bytes require bit/type-exact fresh content and current code."""
    from dryml.core import Repo
    import dryml.core.repo as repo_module
    import dryml.core.cdef_codec as codec
    from tests.core.test_exact_load_metadata_evidence import exact_work_counts

    monkeypatch.setattr(repo_module, "_content_hash", lambda _: b"collision")
    repo = Repo()
    sources = [DefinitionRecord(Definition(MappingRecordObject, {"value": i}).concretize()).to_bytes() for i in (1, 2)]
    for source in sources:
        _prepared_read(repo, source)
    with exact_work_counts() as counts:
        results = [_prepared_read(repo, source) for source in sources]
    assert counts["decode_cdef_graph"] == 0 and counts["dill_load"] == 4
    assert results[0].definition != results[1].definition
    original_load = dill.load
    def changed_load(stream):
        payload = original_load(stream)
        atom = payload["definition"]["nodes"][0]["parameters"]["items"][0][1]["items"][0][1]
        atom["value"] = True  # Public structural equality can equate 1 and True.
        return payload
    with monkeypatch.context() as patch:
        patch.setattr(dill, "load", changed_load)
        with exact_work_counts() as counts:
            try:
                _prepared_read(repo, sources[0])
            except StoreRecordError:
                pass
        assert counts["decode_cdef_graph"] == 1
        assert counts["_realize_prepared_graph"] == 0
    with monkeypatch.context() as patch:
        patch.setattr(codec, "_GRAPH_PREPARATION_POLICY", 2)
        with exact_work_counts() as counts:
            _prepared_read(repo, sources[1])
        assert counts["decode_cdef_graph"] == 1
    with monkeypatch.context() as patch:
        function = codec._eligible_graph_size
        patch.setattr(function, "__code__", function.__code__.replace(co_name="revised_eligibility"))
        with exact_work_counts() as counts:
            _prepared_read(repo, sources[1])
        assert counts["decode_cdef_graph"] == 1
    original = codec._graph_content_matches
    def matches(*args):
        return original(*args)
    monkeypatch.setattr(codec, "_graph_content_matches", matches)
    with exact_work_counts() as counts:
        _prepared_read(repo, sources[0])
    assert counts["decode_cdef_graph"] == 1


def test_prepared_state_ref_keeps_every_path_and_object_id_check():
    """Reference realization validates fresh identity and state association data."""
    from dryml.core import Repo, ObjectId, ObjectRef, StateRef
    from dryml.core.store.records import StateRefRecord
    from dryml.core.utils.graph.path import GraphPath
    from tests.core.test_cdef_graph_codec import CodecStatefulLeaf
    from tests.core.test_exact_load_metadata_evidence import exact_work_counts

    ref = StateRef(ObjectRef(Definition(CodecStatefulLeaf, 1).concretize(), {GraphPath(): ObjectId()}),
                   {GraphPath(): "Dill-" + "1" * 64})
    source = StateRefRecord(ref).to_bytes()
    repo = Repo()
    first = _prepared_read(repo, source, StateRefRecord)
    with exact_work_counts() as counts:
        second = _prepared_read(repo, source, StateRefRecord)
    assert counts["decode_cdef_graph"] == counts["_graph_and_labels"] == 0
    assert second.state_ref == first.state_ref == ref
    assert second.state_ref.object.object_id is not first.state_ref.object.object_id
    for field in ("objects", "states"):
        data = StateRefRecord._decode_framed(source)
        entries = data["state_ref"]["object"][field] if field == "objects" else data["state_ref"][field]
        entries.append(dict(entries[0]))
        changed = b"DRYML-STORE-RECORD/state-ref/1\n" + dill.dumps(data, protocol=5)
        with pytest.raises(StoreRecordError):
            _prepared_read(repo, changed, StateRefRecord)


def test_plain_repo_definition_reads_explicitly_reach_preparation(tmp_path):
    """Plain structural authority inspection benefits while public Store reads validate."""
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.core.store.store import StoreAuthorityError
    from tests.core.test_exact_load_metadata_evidence import exact_work_counts

    store = DirStore(tmp_path / "store", query_index="none")
    record = DefinitionRecord(Definition(MappingRecordObject, {"value": 1}).concretize())
    store.write_definition_record(record)
    repo = Repo(store)
    assert repo.has_cdef_light(record.definition)
    with exact_work_counts() as counts:
        assert repo.has_cdef_light(record.definition)
    assert counts["decode_cdef_graph"] == 0 and counts["_realize_prepared_graph"] == 1
    assert store._can_reuse_exact_validation()
    with exact_work_counts() as counts:
        assert store.read_definition_record(record.digest) is not None
    assert counts["decode_cdef_graph"] == 1
    replacement = DefinitionRecord(Definition(MappingRecordObject, {"value": 2}).concretize())
    Path(store._definition_path(record.digest)).write_bytes(replacement.to_bytes())
    with pytest.raises(StoreAuthorityError, match="direct path"):
        repo.has_cdef_light(record.definition)
    assert not repo._validation_evidence._pending


def _isolate_records_decoder_code(monkeypatch):
    """Keep other decoder bindings intact while replacing only records' code."""
    from types import FunctionType
    import dryml.core.cdef_codec as codec
    import dryml.core.store.records as records

    original = records.decode_cdef_graph
    delegate = FunctionType(original.__code__, original.__globals__, original.__name__,
                            original.__defaults__, original.__closure__)
    monkeypatch.setattr(codec, "decode_cdef_graph", delegate)
    return original


def _reject_records_decoder(monkeypatch, override, original):
    """Reject graph admission through the records binding or its code object."""
    import dryml.core.store.records as records

    def reject(data):
        raise ValueError("records decoder policy rejects graph")

    if override == "alias":
        monkeypatch.setattr(records, "decode_cdef_graph", reject)
    else:
        monkeypatch.setattr(original, "__code__", reject.__code__)


@pytest.mark.parametrize("override", ["alias", "code"])
@pytest.mark.parametrize("phase", ["cold", "warm", "cleared"])
def test_records_decoder_override_remains_authoritative(tmp_path, monkeypatch, override, phase):
    """Public Store and light Repo reads obey the actual decoder after warming."""
    from dryml.core import Repo
    import dryml.core.store.records as records
    from dryml.core.store.dir import DirStore
    from dryml.core.store.store import StoreAuthorityError

    store = DirStore(tmp_path / "store", query_index="none")
    record = DefinitionRecord(Definition(MappingRecordObject, {"value": 1}).concretize())
    store.write_definition_record(record)
    original = (_isolate_records_decoder_code(monkeypatch) if override == "code"
                else records.decode_cdef_graph)
    repo = Repo(store)
    if phase != "cold":
        assert repo.has_cdef_light(record.definition)
        assert repo._validation_evidence._resident
    if phase == "cleared":
        repo.clear_cache(strong=False, weak=False)
        assert repo._validation_evidence.accounted_bytes == 0
    retained = tuple(repo._validation_evidence._resident.values())
    _reject_records_decoder(monkeypatch, override, original)
    assert store._can_reuse_exact_validation()

    with pytest.raises(StoreRecordError, match="graph authority") as error:
        DefinitionRecord.from_bytes(record.to_bytes())
    assert str(error.value.__cause__) == "records decoder policy rejects graph"
    with pytest.raises(StoreAuthorityError, match="graph authority"):
        store.read_definition_record(record.digest)
    for _ in range(2):
        with pytest.raises(StoreAuthorityError, match="graph authority"):
            repo.has_cdef_light(record.definition)
        assert tuple(repo._validation_evidence._resident.values()) == retained
        assert not repo._validation_evidence._pending


@pytest.mark.parametrize("phase", ["cold", "warm", "cleared"])
def test_records_decoder_delegate_never_retains_custom_policy(tmp_path, monkeypatch, phase):
    """Successful wrappers stay uncached when policy changes without rebinding."""
    from dryml.core import Repo
    import dryml.core.store.records as records
    from dryml.core.store.dir import DirStore
    from dryml.core.store.store import StoreAuthorityError
    from tests.core.test_exact_load_metadata_evidence import exact_work_counts

    store = DirStore(tmp_path / "store", query_index="none")
    record = DefinitionRecord(Definition(MappingRecordObject, {"value": 1}).concretize())
    source = record.to_bytes()
    store.write_definition_record(record)
    repo = Repo(store)
    if phase != "cold":
        assert repo.has_cdef_light(record.definition)
    if phase == "cleared":
        repo.clear_cache(strong=False, weak=False)
    retained = tuple(repo._validation_evidence._resident.values())
    original = records.decode_cdef_graph
    calls = []
    rejecting = False

    def decoder(data):
        calls.append(True)
        if rejecting:
            raise ValueError("custom decoder now rejects graph")
        return original(data)

    monkeypatch.setattr(records, "decode_cdef_graph", decoder)
    for _ in range(2):
        calls.clear()
        with exact_work_counts() as counts:
            assert _prepared_read(repo, source)._prepared is None
            assert repo.has_cdef_light(record.definition)
        assert len(calls) == 2
        assert counts["_realize_prepared_graph"] == 0
        assert tuple(repo._validation_evidence._resident.values()) == retained
        assert not repo._validation_evidence._pending
    rejecting = True
    with pytest.raises(StoreAuthorityError, match="graph authority"):
        repo.has_cdef_light(record.definition)


@pytest.mark.parametrize("kind", ["source", "atom", "quote"])
def test_valid_excluded_records_keep_full_decoder_fallback(kind):
    """Valid source/atom/quoted records are decoded twice and never retained."""
    from dryml.core import Repo
    from dryml.core.symbol import ImportRef, SourceSpec
    from dryml.core.tensor_spec import TensorSpec
    from tests.core.test_exact_load_metadata_evidence import exact_work_counts

    if kind == "source":
        cls = SourceSpec.from_source("class SourceTarget(Object):\n    def __init__(self, config):\n        self.config = config",
                                     kind="class", name="SourceTarget", imports={"Object": ImportRef("dryml.core.object", "Object")})
        definition = Definition(cls, config=1).concretize()
    elif kind == "atom":
        definition = Definition(MappingRecordObject, config=TensorSpec("float32", shape=(2,), backend="numpy")).concretize()
    else:
        from dryml.core import ConcreteDefinition
        from dryml.core.bound_args import BoundArguments
        from dryml.core.cdef_graph import EdgeKind
        from dryml.core.links import DefLink
        from dryml.core.quoted import QuotedDef
        definition = ConcreteDefinition._from_bound_record(
            ImportRef("tests.core.test_store_records", "MappingRecordObject"),
            BoundArguments((("config", DefLink.finalized(EdgeKind.REF, QuotedDef(Definition(RecordObject)))),)),
            stateful_role=False,
        )
    source = DefinitionRecord(definition).to_bytes()
    repo = Repo()
    for _ in range(2):
        with exact_work_counts() as counts:
            record = _prepared_read(repo, source)
        assert counts["decode_cdef_graph"] == 1 and counts["_realize_prepared_graph"] == 0
        assert record._prepared is None
        assert not repo._validation_evidence._resident


def test_prepared_graph_reserves_before_optional_owned_copies():
    """Admission failure skips all retained normalization, not full validation."""
    from dryml.core import Repo
    from tests.core.test_exact_load_metadata_evidence import exact_work_counts
    repo = Repo()
    repo._validation_evidence.max_entry_bytes = 1
    source = DefinitionRecord(Definition(MappingRecordObject, {"value": 1}).concretize()).to_bytes()
    with exact_work_counts() as counts:
        record = _prepared_read(repo, source)
    assert counts["decode_cdef_graph"] == 1
    assert counts["_freeze_graph_content"] == 0
    assert record._prepared is None
    assert repo._validation_evidence.accounted_bytes == 0


def test_caught_record_error_discards_previously_staged_graph_facts():
    """Catching a later malformed record inside a cut cannot admit earlier facts."""
    from dryml.core import Repo
    from dryml.core.store.records import _RecordPreparation
    repo = Repo()
    source = DefinitionRecord(Definition(MappingRecordObject, {"value": 1}).concretize()).to_bytes()
    with repo._read_cut() as cut:
        preparation = _RecordPreparation(cut)
        DefinitionRecord._from_bytes_prepared(source, preparation)
        assert cut._pending
        with pytest.raises(StoreRecordError):
            DefinitionRecord._from_bytes_prepared(source + b"trailing", preparation)
        assert not cut._pending
    assert repo._validation_evidence.accounted_bytes == 0


@pytest.mark.parametrize("order,dtype", [("C", "float64"), ("F", "bool")])
def test_prepared_arrays_preserve_fresh_decoded_layout_and_flags(order, dtype):
    """Numeric/bool C/F arrays retain their public fresh-decoder representation."""
    import numpy as np
    from dryml.core import Repo
    array = np.array([[0, 1, 0], [1, 0, 1]], dtype=dtype, order=order)
    source = DefinitionRecord(Definition(MappingRecordObject, array).concretize()).to_bytes()
    repo = Repo()
    first = _prepared_read(repo, source).definition.parameters["config"]
    record = _prepared_read(repo, source)
    assert record._prepared is not None
    second = record.definition.parameters["config"]
    assert first.dtype == second.dtype and first.shape == second.shape
    assert first.strides == second.strides
    assert first.flags.writeable == second.flags.writeable
    assert np.array_equal(first, second)
