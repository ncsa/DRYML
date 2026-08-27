import pytest

import dryml.core as core
from dryml.core import ConcreteDefinition, Definition, Object
from dryml.core.bound_args import (
    BoundArguments,
    bind_complete_arguments,
    bind_partial_arguments,
    decode_bound_arguments,
    project_bound_arguments,
)
from dryml.core.cdef_identity import V2_IDENTITY_VERSION
from dryml.core.definition import thaw_concrete
from dryml.core.freeze import FrozenDict, FrozenList, FrozenTuple


class BindingFixture(Object):
    def __init__(self, positional_only, /, value=2, *items, keyword=3, **options):
        self.positional_only = positional_only
        self.value = value
        self.items = items
        self.keyword = keyword
        self.options = options


class PreparedFixture(Object):
    prepare_count = 0

    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        cls.prepare_count += 1
        return args, {**kwargs, "uid": "prepared", "metadata": {"source": "prepare"}}

    def __init__(self, name, *, uid=None, metadata=None):
        self.name = name
        self.uid = uid
        self.metadata = metadata


class DefaultFixture(Object):
    def __init__(self, value=3):
        self.value = value


class MutableDefaultFixture(Object):
    def __init__(self, values=[]):
        self.values = values


class InvalidPreparedFixture(Object):
    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        return args, {"unknown": 1}

    def __init__(self, value):
        self.value = value


class UnsupportedDefaultFixture(Object):
    def __init__(self, value=object()):
        self.value = value


class RequiredFixture(Object):
    def __init__(self, required):
        self.required = required


class KeywordOnlyProjectionFixture(Object):
    def __init__(self, *, value):
        self.value = value


class OriginalVarargProjectionFixture(Object):
    def __init__(self, value=2, *items):
        self.value = value
        self.items = items


class AddedLeadingVarargProjectionFixture(Object):
    def __init__(self, new=1, value=2, *items):
        self.new = new
        self.value = value
        self.items = items


class ClassParameterFixture(Object):
    def __init__(self, cls, dryml_cls=None):
        self.cls = cls
        self.dryml_cls = dryml_cls


def test_bound_arguments_are_immutable_semantic_name_value_records():
    record = BoundArguments((("value", 1),))

    assert record["value"] == 1
    assert tuple(record.items()) == (("value", 1),)
    with pytest.raises(ValueError, match="duplicate"):
        BoundArguments((("value", 1), ("value", 2)))
    with pytest.raises(TypeError, match="strings"):
        BoundArguments(((1, "value"),))


def test_bound_arguments_are_private_to_the_bound_record_module():
    assert not hasattr(core, "BoundArguments")
    assert "BoundArguments" not in core.__all__


def test_complete_binding_captures_defaults_and_normalizes_call_spelling():
    positional = bind_complete_arguments(BindingFixture, (1, 4), {"flag": True})
    keyword = bind_complete_arguments(
        BindingFixture,
        (1,),
        {"value": 4, "keyword": 3, "flag": True},
    )

    assert tuple(positional.items()) == (
        ("positional_only", 1),
        ("value", 4),
        ("items", ()),
        ("keyword", 3),
        ("options", {"flag": True}),
    )
    assert positional == keyword


def test_binding_partial_omits_defaults_and_projection_uses_current_signature():
    partial = bind_partial_arguments(BindingFixture, (1,), {"keyword": 4})
    complete = bind_complete_arguments(BindingFixture, (1,), {"value": 5, "keyword": 4, "flag": True})

    assert tuple(partial.items()) == (("positional_only", 1), ("keyword", 4))
    args, kwargs = project_bound_arguments(BindingFixture, complete)
    assert args == (1,)
    assert kwargs == {"value": 5, "keyword": 4, "flag": True}
    assert BindingFixture(*args, **kwargs).options == {"flag": True}


def test_projection_avoids_duplicate_binding_before_nonempty_varargs():
    bound = bind_complete_arguments(BindingFixture, (1, 2, "tail"), {"keyword": 4})

    args, kwargs = project_bound_arguments(BindingFixture, bound)

    assert args == (1, 2, "tail")
    assert kwargs == {"keyword": 4}


def test_projection_preserves_values_when_current_signature_adds_a_leading_default():
    bound = bind_complete_arguments(OriginalVarargProjectionFixture, (2, "tail"), {})

    args, kwargs = project_bound_arguments(AddedLeadingVarargProjectionFixture, bound)

    assert args == (1, 2, "tail")
    assert kwargs == {}


def test_projection_uses_the_current_signature_without_changing_bound_identity():
    bound = bind_complete_arguments(DefaultFixture, (), {})

    args, kwargs = project_bound_arguments(KeywordOnlyProjectionFixture, bound)
    assert args == ()
    assert kwargs == {"value": 3}
    with pytest.raises(TypeError, match="required"):
        project_bound_arguments(RequiredFixture, BoundArguments((("renamed", 3),)))


def test_materialization_preserves_constructor_parameter_named_cls():
    direct = ClassParameterFixture(Object, dryml_cls=Definition)
    cdef = Definition(
        ClassParameterFixture,
        Object,
        dryml_cls=Definition,
    ).concretize()

    rebuilt = cdef.build(instance="new", cache="none")

    assert direct.cls is Object
    assert direct.dryml_cls is Definition
    assert rebuilt.cls is Object
    assert rebuilt.dryml_cls is Definition


def test_private_v2_pipeline_prepares_once_and_captures_injected_values(monkeypatch):
    PreparedFixture.prepare_count = 0

    cdef = Definition(PreparedFixture, "example").concretize()

    assert cdef.identity_version == V2_IDENTITY_VERSION
    assert PreparedFixture.prepare_count == 1
    assert cdef["parameters"] == FrozenDict({
        "name": "example",
        "uid": "prepared",
        "metadata": FrozenDict({"source": "prepare"}),
    })
    assert "args" not in cdef
    state = cdef.__getstate__()
    assert set(state) == {"identity_version", "cls", "parameters", "stable_hash_cache"}
    assert state["parameters"] == cdef["parameters"]
    monkeypatch.setattr(
        "dryml.core.canonical.resolve_symbol",
        lambda *args, **kwargs: pytest.fail("persisted V2 decoding must not resolve classes"),
    )
    restored = object.__new__(ConcreteDefinition)
    restored.__setstate__(state)
    assert restored == cdef


def test_public_exact_constructor_uses_v2_binding_and_compatibility_projection():
    cdef = ConcreteDefinition(BindingFixture, (1, 4, "tail"), {"keyword": 5, "flag": True})

    assert cdef.identity_version == V2_IDENTITY_VERSION
    assert cdef.parameters == FrozenDict({
        "positional_only": 1,
        "value": 4,
        "items": FrozenTuple(("tail",)),
        "keyword": 5,
        "options": FrozenDict({"flag": True}),
    })
    assert cdef.args == FrozenTuple((1, 4, "tail"))
    assert cdef.kwargs == FrozenDict({"keyword": 5, "flag": True})


def test_v2_thaw_uses_current_signature_projection_for_all_parameter_kinds():
    cdef = ConcreteDefinition(BindingFixture, (1, 4, "tail"), {"keyword": 5, "flag": True})

    thawed = thaw_concrete(cdef)

    assert thawed.concretize() == cdef


def test_private_v2_defaults_are_snapshotted_and_change_identity_later():
    first = Definition(DefaultFixture).concretize()
    old_defaults = DefaultFixture.__init__.__defaults__
    try:
        DefaultFixture.__init__.__defaults__ = (4,)
        second = Definition(DefaultFixture).concretize()
    finally:
        DefaultFixture.__init__.__defaults__ = old_defaults

    assert first["parameters"]["value"] == 3
    assert second["parameters"]["value"] == 4
    assert first != second
    assert first.stable_hash() != second.stable_hash()

    mutable = Definition(MutableDefaultFixture).concretize()
    try:
        MutableDefaultFixture.__init__.__defaults__[0].append("later")
        assert mutable["parameters"]["values"] == FrozenList(())
    finally:
        MutableDefaultFixture.__init__.__defaults__[0].clear()


def test_private_v2_binding_and_canonicalization_fail_at_semantic_parameter_paths():
    with pytest.raises(TypeError, match="required"):
        Definition(RequiredFixture).concretize()
    with pytest.raises(TypeError, match="value"):
        Definition(UnsupportedDefaultFixture).concretize()
    with pytest.raises(TypeError, match="unknown"):
        Definition(InvalidPreparedFixture, 1).concretize()


def test_persisted_bound_records_reject_malformed_names_and_values_without_binding(monkeypatch):
    monkeypatch.setattr(
        "dryml.core.canonical.resolve_symbol",
        lambda *args, **kwargs: pytest.fail("persisted V2 decoding must not resolve classes"),
    )
    with pytest.raises(ValueError, match="duplicate"):
        decode_bound_arguments((("value", 1), ("value", 2)))
    with pytest.raises(TypeError, match="strings"):
        decode_bound_arguments(((1, "value"),))
    with pytest.raises(TypeError, match="Non-canonical.*value"):
        decode_bound_arguments((("value", []),))
    assert decode_bound_arguments((("value", Object),))["value"] is Object
    with pytest.raises(TypeError, match="Non-canonical.*value"):
        decode_bound_arguments((("value", ClassParameterFixture),))
    with pytest.raises(TypeError, match="Non-canonical.*0"):
        decode_bound_arguments((("value", FrozenTuple((ClassParameterFixture,))),))

    restored = object.__new__(ConcreteDefinition)
    with pytest.raises(ValueError, match="duplicate"):
        restored.__setstate__({
            "identity_version": V2_IDENTITY_VERSION,
            "cls": BindingFixture,
            "parameters": (("value", 1), ("value", 2)),
        })
    with pytest.raises(ValueError, match="legacy fields"):
        restored.__setstate__({
            "identity_version": V2_IDENTITY_VERSION,
            "cls": BindingFixture,
            "parameters": FrozenDict({"value": 1}),
            "args": (),
            "kwargs": {},
        })
    with pytest.raises(ValueError, match="parameters"):
        restored.__setstate__({
            "identity_version": V2_IDENTITY_VERSION,
            "cls": BindingFixture,
            "args": (),
            "kwargs": {},
        })
