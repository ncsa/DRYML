"""Focused coverage for the internal semantic categorical projection engine."""

import pickle

import pytest

from dryml.core import Definition, Object, SKIP_ARGS
from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_identity import V2_IDENTITY_VERSION
from dryml.core.cdef_graph import EdgeKind
from dryml.core.definition import ConcreteDefinition
from dryml.core.factory import FactorySpec
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.links import DefLink
from dryml.core.quoted import QuotedDef
from dryml.core.symbol import ImportRef, SourceSpec


class ProjectionFixture(Object):
    def __init__(self, first, /, width=2, *items, seed=3, **options):
        self.first = first
        self.width = width
        self.items = items
        self.seed = seed
        self.options = options


class ProjectionContainer(Object):
    def __init__(self, child, values=None):
        self.child = child
        self.values = values


def _project(value, **controls):
    from dryml.core.categorical import project_categorical_definition

    return project_categorical_definition(value, **controls)


def test_classless_skipped_definition_and_prepared_projection_preserve_named_fields():
    classless = Definition(SKIP_ARGS, width=4)

    assert classless.cls is None
    assert classless.args is None
    assert classless.parameters == FrozenDict({"width": 4})
    assert _project(classless).parameters == FrozenDict({"width": 4})

    prepared = Definition(ImportRef("missing.projection", "Fixture"), SKIP_ARGS, width=4)
    repeated = _project(_project(prepared))

    assert repeated.cls == prepared.cls
    assert repeated.args is None
    assert repeated.parameters == FrozenDict({"width": 4})
    assert pickle.loads(pickle.dumps(repeated)) == repeated


def test_prepared_selector_round_trips_through_the_supported_repo_codec(tmp_path):
    from dryml.core import Repo, RepoDefinition, Selector
    from dryml.core.repo_plan import SaveRouting
    from dryml.core.store.dir import DirStore

    selector = Selector(_project(Definition(ProjectionFixture, 1, seed=9)))
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store, save_routing=SaveRouting(((selector, store),)))

    restored = Repo.from_definition(RepoDefinition.from_data(repo.to_definition().to_data()))
    try:
        root = restored.save_routing.routes[0][0].root
        assert root.args is None
        assert root.parameters == FrozenDict({"first": 1, "seed": 9})
    finally:
        restored.close(flush=False)


def test_authored_calls_become_named_without_defaults_or_positional_shifts():
    positional = Definition(ProjectionFixture, 1, 4, "tail", seed=9, flag=True)
    keyword = Definition(ProjectionFixture, 1, width=4, seed=9, flag=True)

    positional_projection = _project(positional, drop=("seed",))
    keyword_projection = _project(keyword, drop=("seed",))

    assert positional_projection.args is None
    assert positional_projection.parameters == FrozenDict({
        "first": 1,
        "width": 4,
        "items": FrozenTuple(("tail",)),
        "options": FrozenDict({"flag": True}),
    })
    assert keyword_projection.parameters == FrozenDict({"first": 1, "width": 4, "options": FrozenDict({"flag": True})})
    assert "seed" not in positional_projection.parameters
    assert "seed" not in keyword_projection.parameters
    assert "first" not in _project(positional, drop=("first",)).parameters
    assert _project(keyword, drop=("seed", "seed")) == keyword_projection
    assert _project(Definition(ProjectionFixture, 1, 4, seed=9), drop=("seed",)) == _project(
        Definition(ProjectionFixture, 1, width=4, seed=9), drop=("seed",)
    )


def test_cdef_projection_reads_named_records_without_class_resolution(monkeypatch):
    source = ConcreteDefinition._from_persisted_record(
        ImportRef("missing.projection", "Fixture"),
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((
            ("first", 1),
            ("width", 4),
            ("items", FrozenTuple()),
            ("seed", 9),
            ("options", FrozenDict({"flag": True})),
        )),
    )
    monkeypatch.setattr(ImportRef, "resolve", lambda self: pytest.fail("CDef projection must not resolve classes"))

    result = _project(source, drop=("seed",))

    assert result.cls == source.cls
    assert result.args is None
    assert result.parameters == FrozenDict({
        "first": 1,
        "width": 4,
        "items": FrozenTuple(),
        "options": FrozenDict({"flag": True}),
    })


def test_cdef_source_spec_projection_preserves_authority_without_resolution(monkeypatch):
    source = SourceSpec.from_source(
        "class SourceProjection:\n    def __init__(self, value):\n        self.value = value",
        kind="class",
        name="SourceProjection",
    )
    cdef = ConcreteDefinition._from_persisted_record(
        source,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("value", 4), ("seed", 9))),
    )
    monkeypatch.setattr(SourceSpec, "resolve", lambda self: pytest.fail("CDef projection must not resolve source"))

    result = _project(cdef, drop=("seed",))

    assert result.cls is source
    assert result.parameters == FrozenDict({"value": 4})


def test_source_spec_signature_keeps_source_authority_and_never_constructs():
    source = SourceSpec.from_source(
        "class SourceProjection:\n    def __init__(self, value, /, *, seed=3):\n        raise AssertionError('constructor must not run')",
        kind="class",
        name="SourceProjection",
    )

    result = _project(Definition(source, 4, seed=9), drop=("seed",))

    assert result.cls is source
    assert result.parameters == FrozenDict({"value": 4})
    assert _project(result).cls is source


def test_partial_omissions_stay_absent_while_cdef_defaults_remain_bound():
    partial = _project(Definition(ProjectionFixture, 1))
    exact = _project(Definition(ProjectionFixture, 1).concretize())

    assert partial.parameters == FrozenDict({"first": 1})
    assert exact.parameters == FrozenDict({
        "first": 1,
        "width": 2,
        "items": FrozenTuple(),
        "seed": 3,
        "options": FrozenDict({}),
    })


def test_controls_validate_atomically_and_skip_signature_when_dropping_all_args(monkeypatch):
    source = Definition(ProjectionFixture, 1, width=4, seed=9)
    original_hash = source.stable_hash()
    original_args = source.args

    with pytest.raises(TypeError, match="sequence"):
        _project(source, drop="seed")
    with pytest.raises(TypeError, match="strings"):
        _project(source, drop=("seed", 1))
    with pytest.raises(TypeError, match="bool"):
        _project(source, drop_args=1)
    with pytest.raises(TypeError, match="bool"):
        _project(source, drop_class=1)
    with pytest.raises(TypeError, match="bool"):
        _project(source, recursive=1)
    with pytest.raises(ValueError, match="missing"):
        _project(source, drop=("seed", "missing"), drop_args=True)

    monkeypatch.setattr("dryml.core.categorical.resolve_symbol", lambda value: pytest.fail("no signature is needed"))
    broad = _project(Definition(ImportRef("missing.projection", "Fixture"), 1), drop_args=True, drop_class=True)
    monkeypatch.undo()
    classless = _project(source, drop_class=True)
    argumentless = _project(source, drop_args=True)

    assert broad.cls is None
    assert broad.args is None
    assert broad.parameters == FrozenDict({})
    assert classless.cls is None
    assert classless.parameters == FrozenDict({"first": 1, "width": 4, "seed": 9})
    assert argumentless.cls is not None
    assert argumentless.parameters == FrozenDict({})
    assert source.stable_hash() == original_hash
    assert source.args == original_args


def test_named_drops_validate_original_names_and_repeated_projection():
    source = Definition(ProjectionFixture, 1, seed=9)
    assert _project(source, drop=("seed",), drop_args=True).parameters == {}
    projected = _project(source, drop=("seed",))
    with pytest.raises(ValueError, match="seed"):
        _project(projected, drop=("seed",))
    with pytest.raises(ValueError, match="seed"):
        _project(Definition(ProjectionFixture, 1), drop=("seed",))
    assert _project(_project(source, drop_class=True), drop_class=True).cls is None


def test_recursive_projection_preserves_aliases_links_and_opaque_values():
    child = Definition(ProjectionFixture, 1, seed=9)
    linked = DefLink.finalized(EdgeKind.REF, child)
    materialized = DefLink.finalized(EdgeKind.MATERIALIZE, child)
    factory = FactorySpec("builtins:tuple", "opaque")
    quoted = QuotedDef(Definition(ProjectionFixture, 3, seed=7))
    root = Definition(ProjectionContainer, child, values={"again": child, "link": linked, "materialized": materialized, "factory": factory, "quoted": quoted})

    root_only = _project(root, recursive=False)
    recursive = _project(root, recursive=True, drop=("seed",))

    assert root_only.parameters["child"] is child
    transformed_child = recursive.parameters["child"]
    assert transformed_child is recursive.parameters["values"]["again"]
    assert "seed" not in transformed_child.parameters
    assert recursive.parameters["values"]["link"].kind is EdgeKind.REF
    assert recursive.parameters["values"]["link"].target is transformed_child
    assert recursive.parameters["values"]["materialized"].kind is EdgeKind.MATERIALIZE
    assert recursive.parameters["values"]["materialized"].target is transformed_child
    assert recursive.parameters["values"]["factory"] is factory
    assert recursive.parameters["values"]["quoted"] is quoted


def test_projection_rejects_set_collapse_and_cdef_cycles_without_mutating_source():
    left = Definition(ProjectionFixture, 1, seed=1)
    right = Definition(ProjectionFixture, 1, seed=2)
    source = Definition(ProjectionContainer, left, values={left, right})
    source_hash = source.stable_hash()

    with pytest.raises(ValueError, match="set cardinality"):
        _project(source, recursive=True, drop=("seed",))
    assert source.stable_hash() == source_hash

    cyclic = ConcreteDefinition._from_persisted_record(
        ProjectionFixture,
        identity_version=V2_IDENTITY_VERSION,
        parameters=BoundArguments((("first", 1),)),
    )
    object.__setattr__(cyclic, "_bound_args", BoundArguments((("first", cyclic),)))

    with pytest.raises(ValueError, match="Cycle"):
        _project(cyclic, recursive=True)
