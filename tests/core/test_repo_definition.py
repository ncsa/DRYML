"""Focused v1 portable Repo-definition coverage."""

from copy import deepcopy
from io import BytesIO
import os
import shutil
import subprocess
import sys
import textwrap
import threading
from typing import Annotated

import pytest

from dryml.core import (
    AnyValue, Choice, Definition, Exact, IntRange, Mat, Missing, ObjectId,
    ObjectRef, Par, Present, Ref, Repo, RepoDefinition, RepoDefinitionError,
    Satisfies, Selector, SKIP_ARGS, StateRef, SubclassOf, UniformFromSet,
    UniformIntRange,
)
from dryml.core.object import Object, Serializable
from dryml.core.params import ExactMatcher, UniformIntRangeGenerator
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore
from dryml.core.repo_plan import SaveRouting
from dryml.core.symbol import ImportRef
from dryml.core.arg_roles import SelectorArg
from dryml.core.arg_roles import RefCDefArg
import dryml.core.session as session
from dryml.core.utils.graph.path import GraphPath, Parameter
from dryml.core.utils.graph.value import iter_set_members


class DefinitionTarget:
    def __init__(self, value=None):
        self.value = value


class SelectorParityBase:
    pass


class SelectorParityChild(SelectorParityBase):
    pass


class ReconstructionTarget(Object):
    def __init__(self, value=""):
        super().__init__()
        self.value = value


class SelectorArgumentTarget(Object):
    def __init__(self, selector: Annotated[Definition, SelectorArg()]):
        super().__init__()
        self.selector = selector


class RefArgumentTarget(Object):
    def __init__(self, child: Annotated[Definition, RefCDefArg()]):
        super().__init__()
        self.child = child


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


@pytest.mark.parametrize("case", ["missing", "zero", "file-like"], ids=str)
def test_definition_rejects_unreconstructable_archive_without_mutating_it(tmp_path, case):
    """Export never commits, repairs, or substitutes unavailable archive authority."""

    if case == "file-like":
        buffer = BytesIO()
        archive = ZipStore(buffer)
        before = buffer.getvalue()
    else:
        path = tmp_path / f"{case}.zip"
        archive = _archive_store(path)
        if case == "missing":
            path.unlink()
        else:
            path.write_bytes(b"")
        before = path.read_bytes() if path.exists() else None

    with pytest.raises(RepoDefinitionError, match="clean committed archive"):
        Repo(archive).to_definition()

    if case == "file-like":
        assert buffer.getvalue() == before
    else:
        assert (path.read_bytes() if path.exists() else None) == before
    archive.close()


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


def test_from_definition_reopens_existing_stores_and_preserves_route_behavior(tmp_path):
    """Explicit reconstruction creates fresh handles with equivalent selection."""

    directory = DirStore(tmp_path / "dir", query_index="none")
    archive = _archive_store(tmp_path / "store.zip")
    selector = Selector(Definition(ReconstructionTarget, value=Exact("selected")), strict=True)
    source = Repo(
        [directory, archive],
        config={"mode": "portable"},
        save_routing=SaveRouting(((selector, archive),), "first", "closure"),
    )

    rebuilt = Repo.from_definition(source.to_definition())
    try:
        assert rebuilt is not source
        assert [type(store) for store in rebuilt.stores] == [DirStore, ZipStore]
        assert all(left is not right for left, right in zip(source.stores, rebuilt.stores))
        assert rebuilt.config == {"mode": "portable"}
        assert rebuilt.save_routing.graph_mode == "closure"
        target = Definition(ReconstructionTarget, value="selected").concretize()
        unmatched = Definition(ReconstructionTarget, value="other").concretize()
        with source._retain_save_context() as source_context:
            with rebuilt._retain_save_context() as rebuilt_context:
                assert source.save_routing.routes[0][0].matches(target)
                assert rebuilt.save_routing.routes[0][0].matches(target)
                assert source._select_save_destinations(source_context, target)[0] is archive
                assert type(rebuilt._select_save_destinations(rebuilt_context, target)[0]) is ZipStore
                assert source._select_save_destinations(source_context, unmatched)[0] is directory
                assert type(rebuilt._select_save_destinations(rebuilt_context, unmatched)[0]) is DirStore
    finally:
        rebuilt.close(flush=False)
        archive.close()


@pytest.mark.parametrize("case", ["missing", "empty", "wrong-type", "malformed"], ids=str)
def test_from_definition_never_initializes_invalid_directory_authority(tmp_path, case):
    """Existing-only reconstruction rejects every invalid directory without repair."""

    store = DirStore(tmp_path / "source", query_index="none")
    data = Repo(store).to_definition().to_data()
    target = tmp_path / case
    data["stores"][0]["path"] = str(target)
    if case == "empty":
        target.mkdir()
    elif case == "wrong-type":
        target.write_text("not a directory")
    elif case == "malformed":
        target.mkdir()
        (target / "store-format.record").write_bytes(b"not a store format")

    with pytest.raises(RepoDefinitionError):
        Repo.from_definition(RepoDefinition.from_data(data))

    assert not target.exists() if case == "missing" else True
    if case in {"empty", "malformed"}:
        assert not (target / "definitions").exists()


@pytest.mark.parametrize("kind", ["dir-missing", "dir-incompatible", "zip-missing", "zip-incompatible"])
def test_from_definition_invalid_resources_never_initialize_or_activate_session(
        tmp_path, monkeypatch, kind):
    """Invalid DirStore and ZipStore descriptors fail before session or storage creation."""

    source = DirStore(tmp_path / "source", query_index="none")
    data = Repo(source).to_definition().to_data()
    target = tmp_path / kind
    if kind.startswith("dir"):
        data["stores"][0] = {"kind": "dir", "path": str(target), "query_index": "none"}
        if kind.endswith("incompatible"):
            target.mkdir()
            (target / "store-format.record").write_bytes(b"incompatible")
    else:
        data["stores"][0] = {"kind": "zip", "path": str(target)}
        if kind.endswith("incompatible"):
            target.write_bytes(b"not an archive")
    before = target.read_bytes() if target.is_file() else None
    monkeypatch.setattr(
        session, "configure",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("activated session")),
    )

    with pytest.raises(RepoDefinitionError):
        Repo.from_definition(RepoDefinition.from_data(data))

    assert not target.exists() if kind.endswith("missing") else (
        target.read_bytes() == before if target.is_file() else not (target / "definitions").exists()
    )


def test_from_definition_removed_after_preflight_never_creates_replacement(tmp_path, monkeypatch):
    """A normal authority removal after descriptor preflight still fails closed."""

    source = DirStore(tmp_path / "source", query_index="none")
    definition = Repo(source).to_definition()
    original_open = DirStore.open_existing

    def remove_then_open(cls, path, *, query_index):
        shutil.rmtree(path)
        return original_open(path, query_index=query_index)

    monkeypatch.setattr(DirStore, "open_existing", classmethod(remove_then_open))
    with pytest.raises(RepoDefinitionError):
        Repo.from_definition(definition)

    assert not (tmp_path / "source").exists()


def test_reconstruction_failure_closes_only_freshly_opened_resources(tmp_path, monkeypatch):
    """A later failed Store open unwinds fresh handles without touching the source."""

    source_store = DirStore(tmp_path / "source", query_index="none")
    source = Repo(source_store)
    data = source.to_definition().to_data()
    archive = _archive_store(tmp_path / "archive.zip")
    data["stores"].append({"kind": "zip", "path": str(tmp_path / "archive.zip")})
    data["default_store"] = 0
    opened = []
    original_close = DirStore.close

    def observe_close(self):
        opened.append(self)
        return original_close(self)

    monkeypatch.setattr(DirStore, "close", observe_close)
    monkeypatch.setattr(
        ZipStore,
        "open_existing",
        classmethod(lambda cls, path: (_ for _ in ()).throw(OSError("later open failed"))),
    )
    with pytest.raises(RepoDefinitionError):
        Repo.from_definition(RepoDefinition.from_data(data))

    assert source_store not in opened
    assert opened
    assert source_store.read_main_ref() is None
    archive.close()


def test_reconstruction_retains_exact_selector_data_and_subclass_semantics(tmp_path):
    """Nested exact CDefs retain SelectorArg data, links, and live subclasses."""

    nested = Selector(Definition(ReconstructionTarget, SKIP_ARGS, value=Exact("nested")))
    selector_cdef = Definition(SelectorArgumentTarget, nested).concretize()
    ref_cdef = Definition(RefArgumentTarget, Definition(ReconstructionTarget)).concretize()
    selector = Selector(
        Definition(
            ReconstructionTarget,
            value=Choice([selector_cdef, ref_cdef]),
            type_value=SubclassOf(ReconstructionTarget),
        )
    )
    store = DirStore(tmp_path / "store", query_index="none")
    source = Repo(store, save_routing=SaveRouting(((selector, store),)))

    rebuilt = Repo.from_definition(source.to_definition())
    try:
        rebuilt_selector = rebuilt.save_routing.routes[0][0]
        original_value = selector.root.kwargs["value"].matcher.values[0]
        rebuilt_value = rebuilt_selector.root.kwargs["value"].matcher.values[0]
        assert original_value._stateful_role is rebuilt_value._stateful_role
        assert type(rebuilt_value.parameters["selector"]).__name__ == "SelectorSpec"
        assert rebuilt_value.parameters["selector"].selector.root.args is None
        rebuilt_ref = rebuilt_selector.root.kwargs["value"].matcher.values[1]
        assert rebuilt_ref.parameters["child"].kind.value == "ref"
    finally:
        rebuilt.close(flush=False)


def test_definition_reconstructs_in_subprocess_from_a_different_directory(tmp_path):
    """JSON transport recreates fresh routing without inheriting source handles."""

    directory = DirStore(tmp_path / "dir", query_index="none")
    archive = _archive_store(tmp_path / "store.zip")
    source = Repo(
        [directory, archive],
        save_routing=SaveRouting(((Selector(Definition(Object)), archive),)),
    )
    later = tmp_path / "later"
    later.mkdir()
    script = textwrap.dedent(
        """
        import sys
        from dryml.core import Definition, Object, Repo, RepoDefinition

        repo = Repo.from_definition(RepoDefinition.from_json(sys.argv[1]))
        try:
            target = Definition(Object).concretize()
            with repo._retain_save_context() as context:
                selected = repo._select_save_destinations(context, target)[0]
                print(type(selected).__name__, selected.archive_path)
        finally:
            repo.close(flush=False)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, source.to_definition().to_json()],
        cwd=later,
        check=True,
        capture_output=True,
        text=True,
    )
    archive.close()

    assert result.stdout.strip() == f"ZipStore {tmp_path / 'store.zip'}"


@pytest.mark.parametrize(
    ("selector", "matching", "unmatched"),
    (
        (
            Selector(Definition(ReconstructionTarget, "selected")),
            Definition(ReconstructionTarget, "selected").concretize(),
            Definition(ReconstructionTarget, "other").concretize(),
        ),
        (
            Selector(Definition(ReconstructionTarget, SKIP_ARGS, value=Exact("selected"))),
            Definition(ReconstructionTarget, "selected").concretize(),
            Definition(ReconstructionTarget, "other").concretize(),
        ),
        (
            Selector(Definition(ReconstructionTarget, value=Present())),
            Definition(ReconstructionTarget, "selected").concretize(),
            Definition(ReconstructionTarget).concretize(),
        ),
        (
            Selector(Definition(ReconstructionTarget, value=Missing())),
            Definition(ReconstructionTarget).concretize(),
            Definition(ReconstructionTarget, "selected").concretize(),
        ),
        (
            Selector(Definition(ReconstructionTarget, value=AnyValue())),
            Definition(ReconstructionTarget, "selected").concretize(),
            Definition(ReconstructionTarget).concretize(),
        ),
        (
            Selector(Definition(ReconstructionTarget, value=Choice(["one", "two"]))),
            Definition(ReconstructionTarget, "one").concretize(),
            Definition(ReconstructionTarget, "other").concretize(),
        ),
        (
            Selector(Definition(ReconstructionTarget, value=IntRange(1, 2))),
            Definition(ReconstructionTarget, 1).concretize(),
            Definition(ReconstructionTarget, 3).concretize(),
        ),
        (
            Selector(Definition(ReconstructionTarget, value=UniformIntRange(1, 2))),
            Definition(ReconstructionTarget, 2).concretize(),
            Definition(ReconstructionTarget, 3).concretize(),
        ),
        (
            Selector(Definition(ReconstructionTarget, value=UniformFromSet(["one", "two"]))),
            Definition(ReconstructionTarget, "two").concretize(),
            Definition(ReconstructionTarget, "other").concretize(),
        ),
        (
            Selector(Definition(ReconstructionTarget, value=SubclassOf(SelectorParityBase))),
            Definition(ReconstructionTarget, SelectorParityChild).concretize(),
            Definition(ReconstructionTarget, ReconstructionTarget).concretize(),
        ),
        (
            Selector(Definition(Object, SKIP_ARGS)),
            Definition(Object).concretize(),
            Definition(ReconstructionTarget).concretize(),
        ),
        (
            Selector(Definition(Object, SKIP_ARGS), cls_policy="exact"),
            Definition(Object).concretize(),
            Definition(ReconstructionTarget).concretize(),
        ),
        (
            Selector(Definition(Object, SKIP_ARGS), strict=True),
            Definition(Object).concretize(),
            Definition(ReconstructionTarget).concretize(),
        ),
        (
            Selector(Definition(ImportRef("dryml.core.object", "Object"), SKIP_ARGS)),
            Definition(Object).concretize(),
            Definition(ReconstructionTarget).concretize(),
        ),
    ),
    ids=(
        "positional", "skip-args", "present", "missing", "any", "choice",
        "int-range", "uniform-int-range", "uniform-from-set", "subclass",
        "selector-class", "exact-class", "strict", "symbolic-class",
    ),
)
def test_definition_reconstruction_preserves_selector_matches_and_routes(
        tmp_path, selector, matching, unmatched):
    """Live reconstruction preserves each portable selector's route decisions."""

    first = DirStore(tmp_path / "first", query_index="none")
    second = DirStore(tmp_path / "second", query_index="none")
    source = Repo(
        [first, second], save_routing=SaveRouting(((selector, second),)),
    )
    rebuilt = Repo.from_definition(source.to_definition())
    try:
        rebuilt_selector = rebuilt.save_routing.routes[0][0]
        for target in (matching, unmatched):
            assert rebuilt_selector.matches(target) is selector.matches(target)
            with source._retain_save_context() as source_context:
                source_destination = source._select_save_destinations(source_context, target)[0]
            with rebuilt._retain_save_context() as rebuilt_context:
                rebuilt_destination = rebuilt._select_save_destinations(rebuilt_context, target)[0]
            assert source.stores.index(source_destination) == rebuilt.stores.index(rebuilt_destination)
    finally:
        rebuilt.close(flush=False)


def test_definition_reconstruction_preserves_roles_and_exact_reference_identity(tmp_path):
    """Quoted roles and exact shared/independent references keep match semantics."""

    shared = Definition(ReferenceLeaf, 1).concretize()
    independent = Definition(ReferenceLeaf, 1).concretize()
    object_ref = ObjectRef(shared, {GraphPath(): ObjectId(("selector",))})
    state_ref = StateRef(object_ref, {GraphPath(): _state_hash("e")})
    nested = Selector(Definition(ReconstructionTarget, SKIP_ARGS, value=Exact("nested")))
    role_value = Definition(SelectorArgumentTarget, nested).concretize()
    ref_value = Definition(RefArgumentTarget, Definition(ReconstructionTarget)).concretize()
    selector = Selector(Definition(
        ReconstructionTarget,
        value=Exact([shared, shared, independent, object_ref, state_ref, role_value, ref_value]),
    ))
    matching = Definition(
        ReconstructionTarget,
        [shared, shared, independent, object_ref, state_ref, role_value, ref_value],
    ).concretize()
    unmatched = Definition(
        ReconstructionTarget,
        [shared, independent, independent, object_ref, state_ref, role_value, ref_value],
    ).concretize()
    store = DirStore(tmp_path / "store", query_index="none")
    source = Repo(store, save_routing=SaveRouting(((selector, store),)))
    rebuilt = Repo.from_definition(source.to_definition())
    try:
        rebuilt_selector = rebuilt.save_routing.routes[0][0]
        assert selector.matches(matching)
        assert rebuilt_selector.matches(matching) is selector.matches(matching)
        assert rebuilt_selector.matches(unmatched) is selector.matches(unmatched)
    finally:
        rebuilt.close(flush=False)


def test_definition_rejects_classless_positional_selector_descriptor(tmp_path):
    """Classless partial Definitions cannot silently discard encoded positional args."""

    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store, save_routing=SaveRouting(((Selector(Definition()), store),)))
    data = repo.to_definition().to_data()
    data["routing"]["routes"][0]["selector"]["nodes"][0]["args"] = [
        {"kind": "atom", "value": "discarded"},
    ]

    with pytest.raises(RepoDefinitionError, match="classless"):
        RepoDefinition.from_data(data)
