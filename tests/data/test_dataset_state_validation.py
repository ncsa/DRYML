"""Exact target binding and closed-schema failure proofs."""

import numpy as np
import pytest

import dryml.data as data
from dryml.core.tensor_spec import TensorSpec
from dryml.data.continuation import encode_value
from dryml.methods import Method, traits
from dryml.core import Mat
from dryml.numpy import NumpyArray


class PayloadScale(Method):
    def __init__(self, payload: Mat[NumpyArray]):
        self.payload = payload

    @traits(backend="numpy")
    def numpy(self, value):
        return value * self.payload.array[0]

    def infer_output_spec(self, spec):
        return spec


def _capture(dataset):
    with dataset.open() as traversal:
        traversal.enable_replay_recovery_if_needed()
        next(traversal.next_epoch())
        return traversal.capture_state()


def test_equivalent_materialized_target_allowed_but_configuration_sharing_and_source_state_reject():
    shared = data.ArrayDataset(np.arange(8))
    state = _capture(data.Zip(shared, shared))
    other = data.ArrayDataset(np.arange(8))
    state.validate(data.Zip(other, other))
    with pytest.raises(data.ContinuationMismatchError):
        state.validate(data.Zip(data.ArrayDataset(np.arange(8)), data.ArrayDataset(np.arange(8))))
    with pytest.raises(data.ContinuationMismatchError):
        state.validate(data.Zip(data.ArrayDataset(np.arange(8) + 1), other))
    taken = _capture(data.Take(shared, 5))
    with pytest.raises(data.ContinuationMismatchError):
        taken.validate(data.Take(other, 6))
    with pytest.raises(data.ContinuationMismatchError):
        state.validate(data.Zip(data.ArrayDataset(np.arange(8), spec=TensorSpec("int64", shape=(), backend="torch")), other))


def test_frozen_target_binding_survives_control_mutation_and_open_revalidates():
    dataset = data.Take(data.ArrayDataset(np.arange(8)), 5)
    prepared = dataset.prepare()
    dataset.n = 2
    with prepared.open() as traversal:
        view = traversal.next_epoch()
        next(view)
        state = traversal.capture_state()
        assert state.cdef.parameters["n"] == 5
        assert list(view) == [1, 2, 3, 4]
    state.validate(prepared)
    with prepared.open(state=state) as resumed:
        assert list(resumed.next_epoch()) == [1, 2, 3, 4]
    with pytest.raises(data.ContinuationMismatchError):
        dataset.open(state=state)
    with pytest.raises(data.ContinuationMismatchError):
        prepared.open(state=state, start=3)


def test_each_open_decodes_once_and_revalidates_before_position_checks(
        monkeypatch):
    import dryml.data.state as state_module

    dataset = data.ArrayDataset(np.arange(8))
    prepared = dataset.prepare()
    state = _capture(dataset)
    assert state.validate(prepared) is None
    original = state_module.decode_value
    decodes = []

    def counted(metadata, payloads):
        if metadata == state._metadata:
            decodes.append(metadata)
        return original(metadata, payloads)

    monkeypatch.setattr(state_module, "decode_value", counted)
    for _ in range(2):
        with prepared.open(state=state) as resumed:
            assert list(resumed.next_epoch()) == list(range(1, 8))
    assert len(decodes) == 2
    dataset.arrays[0] = -1
    with pytest.raises(data.ContinuationMismatchError, match="Frozen Dataset"):
        prepared.open(state=state, start="invalid")
    assert len(decodes) == 3


@pytest.mark.parametrize("loaded", [False, True])
def test_metadata_properties_decode_only_independent_inert_packets(
        tmp_path, monkeypatch, loaded):
    import dryml.data.state as state_module

    dataset = data.ArrayDataset(np.arange(64))
    state = _capture(dataset)
    if loaded:
        state = data.DatasetState.load(state.save(tmp_path / "cut"))
    original = state_module.decode_value
    decoded = []

    def counted(metadata, payloads):
        value = original(metadata, payloads)
        decoded.append(value)
        return value

    monkeypatch.setattr(state_module, "decode_value", counted)
    first, second = state.cdef, state.cdef
    assert state.state_ref is None
    assert len(decoded) == 2
    assert all("snapshots" not in value for value in decoded)
    assert first is not second
    assert not np.shares_memory(first.parameters["arrays"],
                                second.parameters["arrays"])
    with pytest.raises(ValueError, match="read-only"):
        first.parameters["arrays"][:] = -1
    np.testing.assert_array_equal(second.parameters["arrays"], np.arange(64))
    np.testing.assert_array_equal(state.cdef.parameters["arrays"],
                                  np.arange(64))


@pytest.mark.parametrize("mutation,message", [
    ("duplicate", "Duplicate or missing target occurrence"),
    ("missing_parent", "Duplicate or missing target occurrence"),
    ("order", "Invalid target child ordering"),
    ("arity", "Invalid target arity"),
])
def test_target_child_inventory_preserves_topology_diagnostics(
        mutation, message):
    from dryml.core._value_json import _value_to_json

    source = data.ArrayDataset(np.arange(3))
    state = _capture(data.Chain(source, source))
    value = state._data()
    rows = value["target"]
    if mutation == "duplicate":
        rows.append(rows[1])
    elif mutation == "missing_parent":
        rows[1]["path"] = [2, 0]
    elif mutation == "order":
        rows[1], rows[2] = rows[2], rows[1]
    else:
        rows[0]["kind"] = "skip"
        rows[0]["controls"] = _value_to_json((0,), grammar=3)
    with pytest.raises(data.ContinuationFormatError, match=message):
        data.DatasetState._from_bytes(*encode_value(value))


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_exact_target_saved_state_round_trips_without_descendant_live_receipt_identity(tmp_path, monkeypatch):
    import dryml.data.state as state_module
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    payload = NumpyArray((1,), "int64", repo=repo)
    method = PayloadScale(payload, repo=repo)
    dataset = data.Map(data.ArrayDataset(np.arange(5), repo=repo), method, repo=repo)
    dataset.method.payload.array = np.asarray([2], dtype=np.int64)
    target_ref = repo.save_object(dataset)
    state = _capture(dataset)
    assert state.state_ref.exact_key() == target_ref.exact_key()
    decoded = []
    original = state_module.decode_value

    def counted(metadata, payloads):
        value = original(metadata, payloads)
        decoded.append(value)
        return value

    with monkeypatch.context() as patch:
        patch.setattr(state_module, "decode_value", counted)
        first, second = state.state_ref, state.state_ref
        assert first is not second
        assert first.exact_key() == second.exact_key()
        assert first.exact_key() == target_ref.exact_key()
        assert len(decoded) == 2
        assert all("snapshots" not in value for value in decoded)
    independent = repo.load_state_ref(target_ref, reuse_live="never")
    with independent.open(state=state) as traversal:
        assert list(traversal.next_epoch()) == [2, 4, 6, 8]


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_stateless_saved_target_receipt_does_not_defeat_independent_equivalence(tmp_path):
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    dataset = data.ArrayDataset(np.arange(5), repo=repo)
    repo.save_object(dataset)
    state = _capture(dataset)
    assert state.state_ref is None
    state.validate(data.ArrayDataset(np.arange(5)))


def test_stateful_method_without_selected_state_remains_live_but_cannot_claim_exact_capture():
    dataset = data.Map(data.ArrayDataset(np.arange(5)), PayloadScale(NumpyArray((1,), "int64")))
    dataset.method.payload.array = np.asarray([3], dtype=np.int64)
    with dataset.open() as traversal:
        view = traversal.next_epoch()
        assert next(view) == 0
        with pytest.raises(data.UnsupportedContinuationError, match="saved source-state"):
            traversal.capture_state()
        assert list(view) == [3, 6, 9, 12]


@pytest.mark.parametrize("mutation", ["extra", "missing", "version", "protocol", "duplicate", "tag", "counter", "negative", "bool", "source_extra", "phase", "split", "snapshot"])
def test_closed_state_rejects_corrupted_envelope_occurrences_and_counters(mutation):
    state = _capture(data.Shuffle(data.ArrayDataset(np.arange(12)), buffer_size=4, seed=7))
    value = state._data()
    if mutation == "extra":
        value["unexpected"] = 1
    elif mutation == "missing":
        del value["policy"]
    elif mutation == "version":
        value["version"] = True
    elif mutation == "protocol":
        value["protocol"]["replay"] = 9
    elif mutation == "duplicate":
        value["execution"]["children"].append(value["execution"]["children"][0])
    elif mutation == "tag":
        value["execution"]["kind"] = "unknown"
    elif mutation == "counter":
        value["execution"]["children"][0]["source"]["fetched"] += 1
    elif mutation == "negative":
        value["position"]["sequence"] = -1
    elif mutation == "bool":
        value["position"]["epoch"] = True
    elif mutation == "source_extra":
        value["execution"]["children"][0]["source"]["unknown"] = 1
    elif mutation == "phase":
        value["execution"]["phase"] = "unknown"
    elif mutation == "split":
        value["execution"]["split_index"] = 99
    else:
        value["snapshots"] = {}
    with pytest.raises(data.ContinuationFormatError):
        data.DatasetState._from_bytes(*encode_value(value))


@pytest.mark.parametrize("field", ["spec", "controls", "factory", "imports", "node_value"])
def test_malformed_metadata_cannot_import_source_targets_during_inert_validation(monkeypatch, field):
    import importlib
    from tests.data.test_generator_recovery import ordinary, SPEC
    state = _capture(data.GeneratorDataset(ordinary, spec=SPEC))
    value = state._data()
    row = value["target"][0]
    enum = {"type": "enum", "module": "u3_forbidden", "qualname": "Secret", "member": "ONE"}
    if field in {"spec", "controls"}:
        row[field] = enum
    elif field == "factory":
        row["source"]["factory"] = enum
    elif field == "imports":
        row["source"]["factory"] = {"type": "source_spec", "kind": "function", "name": "factory",
                                     "source": "def factory():\n    return iter(())", "imports": {"bad": enum}}
    else:
        row["recipe"]["values"]["invalid"] = {"tag": "leaf", "value": enum}
    original = importlib.import_module
    def guarded(name, *args, **kwargs):
        if name == "u3_forbidden":
            pytest.fail("metadata reader imported a source target")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", guarded)
    with pytest.raises(data.ContinuationFormatError):
        data.DatasetState._from_bytes(*encode_value(value))


@pytest.mark.parametrize("mutation", ["initial", "seed", "policy", "nonempty", "child"])
def test_inconsistent_source_initialization_and_frontier_evidence_rejects(mutation):
    from tests.data.test_generator_recovery import ordinary, SPEC
    state = _capture(data.GeneratorDataset(ordinary, {"size": 12}, seed=7, seed_aware=True, spec=SPEC))
    value = state._data()
    if mutation == "initial":
        value["execution"]["source"]["initial"][1][0][1][0][1][0][1][1] = 99
    elif mutation == "seed":
        value["execution"]["source"]["seed"] = 1
    elif mutation == "policy":
        value["policy"] = "disabled"
        value["active_replay"] = False
    elif mutation == "nonempty":
        value["pass_nonempty"] = False
    else:
        value["execution"]["source"] = None
    with pytest.raises(data.ContinuationFormatError):
        data.DatasetState._from_bytes(*encode_value(value))
