"""Focused activation, parsing, and staging coverage for core signatures."""

from __future__ import annotations

import asyncio
import contextvars
import pickle
from typing import Annotated

import pytest

from dryml.core import ConcreteDefinition, Definition, Object
from dryml.core.cdef_codec import encode_cdef_graph
from dryml.core.links import DefLink
from dryml.core.signatures import (
    AutoRef,
    Mat,
    Ref,
    SignatureError,
    compile_signature,
    function,
    normalize_args,
    normalize_return,
    signature_context,
)


class SignatureObject(Object):
    """Minimal Object category fixture for annotation validation."""


class SignatureObjectChild(SignatureObject):
    """Unsupported concrete Object subclass annotation fixture."""


class SignatureContainer(Object):
    """Minimal Object fixture that admits one link argument."""

    def __init__(self, child):
        self.child = child


def _cdef() -> ConcreteDefinition:
    """Return a finalized CDef fixture with no construction side effects."""

    return Definition(SignatureObject).concretize()


def _direct_helper_target(value):
    """Provide a module-visible target for direct-helper discovery coverage."""

    args, kwargs = normalize_args(value)
    return normalize_return(args[0], **kwargs)


def test_callable_subscriptions_and_auto_ref_are_distinct() -> None:
    """Ref/Mat expose one value-assertion and subscribed annotation vocabulary."""

    assert isinstance(Ref(_cdef()), DefLink)
    assert Ref[ConcreteDefinition] != Mat[ConcreteDefinition]
    assert Ref[AutoRef] != Ref[ConcreteDefinition | None]


def test_constructor_mode_uses_the_constructor_annotation_snapshot() -> None:
    """Constructor mode excludes the receiver while compiling ``__init__`` roles."""

    class ConstructorTarget:
        def __init__(self, value: Ref[ConcreteDefinition]):
            self.value = value

    assert compile_signature(ConstructorTarget, constructor=True).slots["value"].role == "ref"


@pytest.mark.parametrize(
    "annotation",
    (
        Ref[list[ConcreteDefinition]],
        Mat[list[ConcreteDefinition]],
        Ref[ConcreteDefinition] | Mat[ConcreteDefinition],
        Ref[Definition | object],
        Ref[SignatureObjectChild],
    ),
)
def test_flat_role_grammar_rejects_unsupported_annotations(annotation: object) -> None:
    """Nested, mixed, incomparable, and subclass role declarations fail closed."""

    def target(value: annotation) -> None:  # type: ignore[valid-type]
        return None

    with pytest.raises(SignatureError):
        compile_signature(target)


def test_ordinary_object_subclass_hints_remain_inert() -> None:
    """Plain typing hints do not activate signature-role validation."""

    def target(value: SignatureObjectChild) -> SignatureObjectChild:
        return value

    plan = compile_signature(target)

    assert plan.slots["value"].mode == "default"


def test_binding_and_target_keywords_are_preserved() -> None:
    """The wrapper preserves Python binding and never steals target control names."""

    @function
    def target(first, /, value=1, *items, named, repo=None, cache=None, reuse_live=None, **extra):
        return first, value, items, named, repo, cache, reuse_live, extra

    assert target(0, 2, 3, named=4, repo="target", cache="target", reuse_live="target", x=5) == (
        0, 2, (3,), 4, "target", "target", "target", {"x": 5},
    )


def test_future_annotations_activate_only_at_compilation() -> None:
    """Trusted future annotations resolve at compilation and unresolved names fail."""

    namespace: dict[str, object] = {"Ref": Ref, "ConcreteDefinition": ConcreteDefinition}
    exec("def target(value: 'Ref[ConcreteDefinition]'): return value", namespace)
    compile_signature(namespace["target"])  # type: ignore[arg-type]
    exec("def missing(value: 'Ref[NoSuchType]'): return value", namespace)
    with pytest.raises(SignatureError):
        compile_signature(namespace["missing"])  # type: ignore[arg-type]


def test_supported_receivers_and_discovery_fail_closed() -> None:
    """Functions, bound methods, and callable instances retain one receiver."""

    class Receiver:
        @function
        def method(self, value):
            return self, value

        def __call__(self, value):
            return self, value

    receiver = Receiver()
    assert receiver.method(2) == (receiver, 2)
    assert function(receiver)(3) == (receiver, 3)
    def unavailable():
        return normalize_args(1)

    with pytest.raises(SignatureError):
        unavailable()


def test_async_and_generator_targets_fail_without_driving_sync_results() -> None:
    """Unsupported declarations fail before invocation while sync results stay inert."""

    async def asynchronous():
        return 1

    def generator():
        yield 1

    with pytest.raises(SignatureError):
        compile_signature(asynchronous)
    with pytest.raises(SignatureError):
        compile_signature(generator)


def test_context_resets_and_rejects_copied_or_foreign_ambient_use() -> None:
    """Borrowed context authority is scoped to its originating task and lifetime."""

    plan = compile_signature(lambda value: value)
    with signature_context(repo=object()):
        copied = contextvars.copy_context()
        assert plan.prepare_args((1,), {}).deliver_args() == ((1,), {})
    with pytest.raises(SignatureError):
        copied.run(lambda: plan.prepare_args((1,), {}))

    async def child() -> None:
        with pytest.raises(SignatureError):
            plan.prepare_args((1,), {})

    with signature_context(repo=object()):
        asyncio.run(child())


def test_finalized_links_round_trip_but_assertions_fail_persistence() -> None:
    """Only finalized links may enter identities, codecs, hashes, or pickle streams."""

    cdef = _cdef()
    finalized = DefLink.finalized(Ref.kind, cdef)
    assert pickle.loads(pickle.dumps(finalized)) == finalized
    assert encode_cdef_graph(Definition(SignatureContainer, finalized).concretize())

    assertion = Ref(cdef)
    with pytest.raises(TypeError):
        Definition(SignatureContainer, assertion).stable_hash()
    with pytest.raises(TypeError):
        pickle.dumps(assertion)


def test_nullable_roles_do_not_bypass_wrapper_conflicts() -> None:
    """Nullable Ref/Mat slots accept plain None but still validate assertions first."""

    @function
    def target(value: Ref[ConcreteDefinition | None]) -> Ref[ConcreteDefinition | None]:
        return value

    assert target(None) is None
    with pytest.raises(SignatureError):
        target(Mat(_cdef()))


def test_direct_helpers_normalize_unique_immediate_target() -> None:
    """Direct helpers derive one immediate function target and do not guess aliases."""

    assert _direct_helper_target("value") == "value"


def test_compiled_mode_projection_and_one_shot_errors_are_bounded() -> None:
    """Compiled plans reject wrong delivery projections and repeated delivery safely."""

    plan = compile_signature(lambda value: value)
    boundary = plan.prepare_args((1,), {})
    assert boundary.deliver_args() == ((1,), {})
    with pytest.raises(SignatureError):
        boundary.deliver_args()
    with pytest.raises(SignatureError) as raised:
        plan.prepare_return(1).deliver_args()
    assert "object at" not in str(raised.value)
