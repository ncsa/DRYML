"""Focused v1 portable Repo-definition coverage."""

from copy import deepcopy
import os
import subprocess
import sys
import textwrap
import threading

import pytest

from dryml.core import (
    AnyValue, Choice, Definition, Exact, IntRange, Mat, ObjectId, ObjectRef,
    Par, Present, Ref, Repo, RepoDefinition, RepoDefinitionError, Satisfies,
    Selector, StateRef, UniformFromSet,
)
from dryml.core.object import Object, Serializable
from dryml.core.params import ExactMatcher, UniformIntRangeGenerator
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore
from dryml.core.repo_plan import SaveRouting
from dryml.core.symbol import ImportRef
from dryml.core.utils.graph.path import GraphPath, Parameter
from dryml.core.utils.graph.value import iter_set_members


class DefinitionTarget:
    def __init__(self, value=None):
        self.value = value


class ReferenceLeaf(Serializable):
    def __init__(self, value=1):
        self.value = value


class ReferenceWrapper(Object):
    def __init__(self, child, child_alias=None):
        self.child = child
        self.child_alias = child_alias


def _state_hash(char="a"):
    return "pkl-" + char * 64


def _route_data(tmp_path):
    """Return one valid detached envelope with a non-empty selector graph."""

    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(
        store,
        save_routing=SaveRouting(((Selector(Definition(DefinitionTarget, "base")), store),)),
    )
    return repo.to_definition().to_data()


def _archive_store(path):
    store = ZipStore(path)
    store._archive_dirty = True
    store.commit()
    return store


def test_definition_round_trip_detaches_mixed_configuration(tmp_path):
    """Export retains only ordered portable descriptors and detached routing."""
    directory = DirStore(tmp_path / "dir", query_index="memory")
    archive = _archive_store(tmp_path / "store.zip")
    shared = Definition(DefinitionTarget, "shared")
    selector = Selector(Definition(DefinitionTarget, shared, value=Exact("x")))
    repo = Repo(
        [directory, archive], config={"nested": {"value": [1]}},
        save_routing=SaveRouting(((selector, archive), (selector, directory)), "all", "closure"),
    )
    repo.save_objs_on_deletion = True

    definition = repo.to_definition()
    data = definition.to_data()

    assert data["stores"] == [
        {"kind": "dir", "path": str(tmp_path / "dir"), "query_index": "memory"},
        {"kind": "zip", "path": str(tmp_path / "store.zip")},
    ]
    assert data["routing"]["match_mode"] == "all"
    assert data["routing"]["graph_mode"] == "closure"
    assert data["routing"]["routes"][0]["store"] == 1
    assert definition.to_json() == RepoDefinition.from_json(definition.to_json()).to_json()
    data["settings"]["config"]["nested"]["value"].append(2)
    assert definition.to_data()["settings"]["config"]["nested"]["value"] == [1]
    repo.save_objs_on_deletion = False


def test_definition_supports_omitted_args_shared_and_parameter_variants(tmp_path):
    """The closed selector grammar retains partial spelling and supported Pars."""
    store = DirStore(tmp_path / "store", query_index="none")
    shared = Definition(DefinitionTarget, "shared")
    root = Definition(
        DefinitionTarget,
        [shared, shared],
        present=Present("p"), any_value=AnyValue("a"), choices=Choice([1, 2]),
        ranged=IntRange(1, 3), generated=UniformFromSet(["x", "y"]),
    )
    repo = Repo(store, save_routing=SaveRouting(((Selector(root, strict=True), store),)))

    route = repo.to_definition().to_data()["routing"]["routes"][0]["selector"]

    assert route["strict"] is True
    assert len(route["nodes"]) == 2
    assert {node["node_kind"] for node in route["nodes"]} == {"definition"}


def test_definition_distinguishes_disabled_empty_and_shorthand_routing(tmp_path):
    """Disabled routing remains distinct from a normalized configured empty policy."""
    store = DirStore(tmp_path / "store", query_index="none")
    disabled = Repo(store).to_definition().to_data()
    configured = Repo(store, save_routing="closure").to_definition().to_data()

    assert disabled["routing"] is None
    assert configured["routing"] == {"graph_mode": "closure", "match_mode": "first", "routes": []}


def test_definition_rejects_dirty_and_nonportable_runtime_configuration(tmp_path):
    """Export refuses dirty archives, custom backends, and runtime factories."""
    archive = _archive_store(tmp_path / "store.zip")
    archive._archive_dirty = True
    with pytest.raises(RepoDefinitionError, match="clean committed archive"):
        Repo(archive).to_definition()
    assert archive._archive_dirty is True

    store = DirStore(tmp_path / "store", query_index="none")
    with pytest.raises(RepoDefinitionError, match="owner-token"):
        Repo(store, owner_token_factory=lambda: "owner").to_definition()
    from dryml.core.query.sqlite import SQLiteQueryIndexConfig
    custom_index = DirStore(tmp_path / "custom", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    with pytest.raises(RepoDefinitionError, match="query policy"):
        Repo(custom_index).to_definition()
    with pytest.raises(RepoDefinitionError) as error:
        Repo(store, config={"secret": object()}).to_definition()
    assert "object at" not in str(error.value)


@pytest.mark.parametrize("mutate", [
    lambda data: data.__setitem__("version", True),
    lambda data: data.__setitem__("unknown", 1),
    lambda data: data["stores"].append(deepcopy(data["stores"][0])),
    lambda data: data["routing"]["routes"][0].__setitem__("store", 3),
])
def test_definition_rejects_malformed_data(tmp_path, mutate):
    """Malformed envelope fields and indices cannot produce partial definitions."""
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store, save_routing=SaveRouting(((Selector(Definition(DefinitionTarget)), store),)))
    data = repo.to_definition().to_data()
    mutate(data)
    with pytest.raises(RepoDefinitionError):
        RepoDefinition.from_data(data)


def test_definition_decoding_is_inert(monkeypatch, tmp_path):
    """Mapping and JSON decode never instantiate Stores or resolve symbols."""
    store = DirStore(tmp_path / "store", query_index="none")
    data = Repo(store, save_routing=SaveRouting(((Selector(Definition(ImportRef("no.such.module", "Type"))), store),))).to_definition().to_data()

    monkeypatch.setattr(DirStore, "__init__", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("opened store")))
    monkeypatch.setattr(ImportRef, "resolve", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resolved symbol")))

    assert RepoDefinition.from_data(data).to_data() == data
    assert RepoDefinition.from_json(RepoDefinition.from_data(data).to_json()).to_data() == data


@pytest.mark.parametrize("field, value", [
    ("string", "x" * (1024 * 1024 + 1)),
    ("integer", 1 << 4096),
    ("entries", list(range(4097))),
    ("nonfinite", float("inf")),
])
def test_definition_enforces_closed_json_value_bounds(tmp_path, field, value):
    """Settings use the same bounded JSON validation as the complete envelope."""
    store = DirStore(tmp_path / "store", query_index="none")
    data = Repo(store).to_definition().to_data()
    data["settings"]["config"] = {field: value}

    with pytest.raises(RepoDefinitionError):
        RepoDefinition.from_data(data)


def test_definition_rejects_duplicate_json_keys_and_cycles(tmp_path):
    """Parser duplicate handling and recursive configuration fail without values."""
    store = DirStore(tmp_path / "store", query_index="none")
    data = Repo(store).to_definition().to_data()
    duplicated = '{"schema":"dryml-repo-definition","schema":"dryml-repo-definition"}'
    with pytest.raises(RepoDefinitionError):
        RepoDefinition.from_json(duplicated)

    data["settings"]["config"] = {}
    data["settings"]["config"]["cycle"] = data["settings"]["config"]
    with pytest.raises(RepoDefinitionError) as error:
        RepoDefinition.from_data(data)
    assert "object at" not in str(error.value)


def test_definition_encodes_complete_nested_reference_topology_inertly(tmp_path, monkeypatch):
    """Exact reference leaves retain all paths, identities, states, and edge roles."""

    leaf = Definition(ReferenceLeaf).concretize()
    child_path = GraphPath((Parameter("child"),))
    imported = ObjectRef(leaf, {GraphPath(): ObjectId(("imported",))})
    imported_state = StateRef(imported, {GraphPath(): _state_hash("b")})
    outer = Definition(
        ReferenceWrapper,
        Mat(imported_state),
        child_alias=Ref(imported_state),
    ).concretize()
    outer_ref = ObjectRef(outer, {child_path: imported.object_id})
    state = StateRef(outer_ref, {child_path: _state_hash("c")})
    selector = Selector(
        Definition(DefinitionTarget, Mat(state), ref=Ref(state), open_leaf={"x": state})
    )
    store = DirStore(tmp_path / "store", query_index="none")
    data = Repo(store, save_routing=SaveRouting(((selector, store),))).to_definition().to_data()

    monkeypatch.setattr(
        ObjectRef,
        "from_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("materialized reference")),
    )
    monkeypatch.setattr(
        StateRef,
        "from_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("materialized state")),
    )

    assert RepoDefinition.from_data(data).to_data() == data

    def first_descriptor(value, kind):
        if isinstance(value, dict):
            if value.get("kind") == kind:
                return value
            for item in value.values():
                found = first_descriptor(item, kind)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = first_descriptor(item, kind)
                if found is not None:
                    return found
        return None

    state_descriptor = first_descriptor(data["routing"]["routes"][0]["selector"], "state-ref")
    assert state_descriptor is not None
    state_descriptor["states"] = []
    with pytest.raises(RepoDefinitionError, match="state paths"):
        RepoDefinition.from_data(data)

    state_descriptor["object"]["objects"] = []
    with pytest.raises(RepoDefinitionError, match="CDef topology"):
        RepoDefinition.from_data(data)


def test_definition_enforces_total_encoded_bound_and_sanitizes_unicode(tmp_path):
    """Mapping, constructor, export, and JSON paths share encoded-byte limits."""

    data = _route_data(tmp_path)
    data["settings"]["config"] = {
        f"key-{index}": "x" * 4096 for index in range(4096)
    }
    for constructor in (RepoDefinition, RepoDefinition.from_data):
        with pytest.raises(RepoDefinitionError, match="encoded"):
            constructor(data)

    store = DirStore(tmp_path / "unicode", query_index="none")
    with pytest.raises(RepoDefinitionError) as error:
        Repo(store, config={"value": "\ud800"}).to_definition()
    assert "\\ud800" not in str(error.value)


def test_definition_validates_materializing_reference_paths_inside_sets(tmp_path):
    """Set-member fingerprints retain and validate exact materializing paths."""

    leaf = Definition(ReferenceLeaf).concretize()
    segment = iter_set_members({leaf})[0][0]
    root = Definition(ReferenceWrapper, {leaf}).concretize()
    path = GraphPath((Parameter("child"), segment))
    reference = ObjectRef(root, {path: ObjectId(("setchild",))})
    state = StateRef(reference, {path: _state_hash("d")})
    store = DirStore(tmp_path / "store", query_index="none")
    selector = Selector(Definition(DefinitionTarget, state))
    data = Repo(store, save_routing=SaveRouting(((selector, store),))).to_definition().to_data()

    assert RepoDefinition.from_data(data).to_data() == data

    def set_descriptor(value):
        if isinstance(value, dict):
            if value.get("kind") == "set":
                return value
            for item in value.values():
                found = set_descriptor(item)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = set_descriptor(item)
                if found is not None:
                    return found
        return None

    encoded_set = set_descriptor(data["routing"]["routes"][0]["selector"])
    assert encoded_set is not None
    encoded_set["items"][0]["fingerprint"] = "0" * 64
    with pytest.raises(RepoDefinitionError, match="CDef topology"):
        RepoDefinition.from_data(data)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda selector: selector.__setitem__("cls_policy", "unsupported"),
        lambda selector: selector["nodes"][0]["args"].__setitem__(0, {"kind": "atom", "value": []}),
        lambda selector: selector["nodes"][0]["args"].__setitem__(0, {"kind": "atom", "value": {}}),
        lambda selector: selector.__setitem__("root", {"kind": "atom", "value": None}),
        lambda selector: selector["nodes"][0].__setitem__("node_kind", "cdef"),
        lambda selector: selector["nodes"][0]["cls"]["symbol"].__setitem__("qualname", ""),
    ],
)
def test_definition_rejects_closed_selector_grammar_violations(tmp_path, mutate):
    """Every selector tag, operand kind, policy, and source field is closed."""

    data = _route_data(tmp_path)
    selector = data["routing"]["routes"][0]["selector"]
    mutate(selector)
    with pytest.raises(RepoDefinitionError):
        RepoDefinition.from_data(data)


def test_definition_rejects_bad_reference_descriptor_kinds_and_duplicate_sets(tmp_path):
    """Reference descriptors require exact topology kinds and canonical set members."""

    data = _route_data(tmp_path)
    selector = data["routing"]["routes"][0]["selector"]
    root = selector["root"]
    selector["nodes"][0]["args"] = [{"kind": "object-ref", "definition": root, "objects": []}]
    with pytest.raises(RepoDefinitionError):
        RepoDefinition.from_data(data)

    data = _route_data(tmp_path)
    selector = data["routing"]["routes"][0]["selector"]
    selector["nodes"][0]["args"] = [{"kind": "state-ref", "object": selector["root"], "states": []}]
    with pytest.raises(RepoDefinitionError):
        RepoDefinition.from_data(data)

    data = _route_data(tmp_path)
    selector = data["routing"]["routes"][0]["selector"]
    selector["nodes"][0]["args"] = [
        {"kind": "set", "items": [{"kind": "atom", "value": 1}, {"kind": "atom", "value": 1}]}
    ]
    with pytest.raises(RepoDefinitionError):
        RepoDefinition.from_data(data)


class _CustomExact(ExactMatcher):
    def __repr__(self):
        return "CUSTOM-REPR-MUST-NOT-LEAK"

    def stable_key(self):
        raise AssertionError("unsupported matcher must not call stable_key")


class _CustomGenerator(UniformIntRangeGenerator):
    def __repr__(self):
        return "CUSTOM-REPR-MUST-NOT-LEAK"

    def stable_key(self):
        raise AssertionError("unsupported generator must not call stable_key")


@pytest.mark.parametrize(
    "parameter",
    [
        Par("custom", _CustomExact(1)),
        Par("custom", ExactMatcher(1), _CustomGenerator(1, 2)),
    ],
)
def test_definition_rejects_custom_matcher_and_generator_subclasses_without_repr(tmp_path, parameter):
    """Only exact supported matcher/generator classes have portable semantics."""

    store = DirStore(tmp_path / "store", query_index="none")
    selector = Selector(Definition(DefinitionTarget, parameter))
    with pytest.raises(RepoDefinitionError) as error:
        Repo(store, save_routing=SaveRouting(((selector, store),))).to_definition()
    assert "CUSTOM-REPR-MUST-NOT-LEAK" not in str(error.value)
    assert error.value.__cause__ is None


def test_definition_rejects_satisfies_without_invoking_predicate(tmp_path):
    """Predicate closure matchers are excluded from the portable selector grammar."""

    store = DirStore(tmp_path / "store", query_index="none")
    selector = Selector(Definition(DefinitionTarget, Satisfies(lambda _: True, name="always")))
    with pytest.raises(RepoDefinitionError):
        Repo(store, save_routing=SaveRouting(((selector, store),))).to_definition()


def test_definition_set_topology_is_deterministic_across_processes(tmp_path):
    """Set traversal assigns stable labels without merging shared CDef identities."""

    script = textwrap.dedent(
        """
        from dryml.core import Definition, Repo, Selector
        from dryml.core.repo_plan import SaveRouting
        from dryml.core.store.dir import DirStore
        from dryml.core.symbol import ImportRef
        import sys

        store = DirStore(sys.argv[1], query_index="none")
        shared = Definition(ImportRef("builtins", "dict"), value="shared")
        left = Definition(ImportRef("builtins", "dict"), value=shared)
        right = Definition(ImportRef("builtins", "dict"), value=shared)
        root = Definition(ImportRef("builtins", "dict"), values={left, right})
        repo = Repo(store, save_routing=SaveRouting(((Selector(root), store),)))
        print(repo.to_definition().to_json())
        """
    )
    outputs = []
    for seed in ("1", "2"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "store")],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]

    store = DirStore(tmp_path / "local", query_index="none")
    shared = Definition(ReferenceLeaf, 1).concretize()
    independent = Definition(ReferenceLeaf, 2).concretize()
    selector = Selector(Definition(DefinitionTarget, values=[shared, shared, independent]))
    nodes = Repo(store, save_routing=SaveRouting(((selector, store),))).to_definition().to_data()["routing"]["routes"][0]["selector"]["nodes"]
    assert len(nodes) == 3


def test_definition_export_releases_config_lock_before_archive_fence(tmp_path, monkeypatch):
    """Configuration updates cannot deadlock behind an archive export fence."""

    archive = _archive_store(tmp_path / "store.zip")
    repo = Repo(archive)
    entered = threading.Event()
    release = threading.Event()
    original_fence = archive.transaction_fence

    def fenced():
        context = original_fence()

        class Fence:
            def __enter__(self):
                entered.set()
                assert release.wait(2)
                return context.__enter__()

            def __exit__(self, *args):
                return context.__exit__(*args)

        return Fence()

    monkeypatch.setattr(archive, "transaction_fence", fenced)
    export = threading.Thread(target=repo.to_definition)
    export.start()
    assert entered.wait(2)
    update = threading.Thread(target=repo.set_config, args=("concurrent", True))
    update.start()
    update.join(1)
    release.set()
    export.join(2)
    update.join(2)

    assert not export.is_alive()
    assert not update.is_alive()
