"""Focused activation, parsing, and staging coverage for core signatures."""

from __future__ import annotations

import asyncio
import contextvars
import pickle
import threading
from types import SimpleNamespace
from typing import Annotated, get_args, get_type_hints

import pytest

from dryml.core import ConcreteDefinition, Definition, Object, Repo
from dryml.core.bound_args import BoundArguments
from dryml.core.cdef_codec import encode_cdef_graph
from dryml.core.cdef_graph import EdgeKind
from dryml.core.freeze import FrozenDict
from dryml.core.links import DefLink
from dryml.core.quoted import QuotedDef
from dryml.core.reference_values import ObjectRef
from dryml.core.signatures import (
    AutoRef,
    Mat,
    Ref,
    SignatureError,
    ReferenceSelection,
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


def _plan(annotation: object, *, returns: object | None = None):
    """Compile one local annotation without future-local name resolution."""

    def target(value):
        return value

    target.__annotations__["value"] = annotation
    if returns is not None:
        target.__annotations__["return"] = returns
    return compile_signature(target)


def test_callable_subscriptions_and_auto_ref_are_distinct() -> None:
    """Ref/Mat expose one value-assertion and subscribed annotation vocabulary."""

    assert isinstance(Ref(_cdef()), DefLink)
    assert Ref[ConcreteDefinition] != Mat[ConcreteDefinition]
    assert Ref[AutoRef] != Ref[ConcreteDefinition | None]


def test_auto_ref_is_a_valid_annotated_type_argument() -> None:
    """Python 3.10 requires the Annotated base to be a type, not a sentinel."""

    def target(value: Ref[AutoRef]) -> Ref[AutoRef]:
        return value

    assert isinstance(AutoRef, type)
    hints = get_type_hints(target, include_extras=True)
    assert get_args(hints["value"])[0] is AutoRef
    assert get_args(hints["return"])[0] is AutoRef
    cdef = _cdef()
    assert function(target)(cdef) is cdef


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
    wrapped = function(lambda value: value)
    with signature_context(repo=object()):
        copied = contextvars.copy_context()
        assert plan.prepare_args((1,), {}).deliver_args() == ((1,), {})
    with pytest.raises(SignatureError):
        copied.run(lambda: wrapped(1))

    async def child() -> None:
        with pytest.raises(SignatureError):
            wrapped(1)
        assert plan.prepare_args((1,), {}, repo=object()).deliver_args() == ((1,), {})

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
    with pytest.raises(SignatureError, match="only plain None"):
        target(Ref(None))

    @function
    def materializing(value: Mat[ConcreteDefinition] | None):
        return value

    assert materializing(None) is None
    with pytest.raises(SignatureError, match="only plain None"):
        materializing(Mat(None))


@pytest.mark.parametrize("role", (Ref, Mat))
def test_outer_nullable_and_distributed_same_role_unions_are_flat(role) -> None:
    """Outer None and same-role branches compile to one ordered nullable slot."""

    inner = _plan(role[ConcreteDefinition | None]).slots["value"]
    outer = _plan(role[ConcreteDefinition] | None).slots["value"]
    distributed = _plan(
        role[ConcreteDefinition] | role[ObjectRef] | None
    ).slots["value"]
    return_plan = _plan(
        object, returns=role[ConcreteDefinition] | None
    )

    assert inner == outer
    assert distributed.role == ("ref" if role is Ref else "mat")
    assert distributed.targets == (ConcreteDefinition, ObjectRef)
    assert distributed.nullable
    assert _plan(role[ConcreteDefinition] | None).prepare_args(
        (None,), {}
    ).deliver_args() == ((None,), {})
    assert return_plan.prepare_return(None).deliver_return() is None


@pytest.mark.parametrize(
    "annotation",
    (
        Ref[ConcreteDefinition] | ConcreteDefinition,
        Ref[ConcreteDefinition] | Mat[ConcreteDefinition] | None,
        Ref[Definition] | Ref[QuotedDef],
        list[Ref[ConcreteDefinition]],
    ),
)
def test_outer_role_unions_reject_plain_mixed_incomparable_and_nested_branches(
    annotation: object,
) -> None:
    """Only None may be plain and all role branches must be comparable."""

    with pytest.raises(SignatureError):
        _plan(annotation)


def test_annotated_variadics_normalize_each_occurrence_and_preserve_projection() -> None:
    """Ref roles apply to each expanded positional and keyword occurrence."""

    first, second, third = _cdef(), _cdef(), _cdef()

    def target(*values, **named):
        return values, named

    target.__annotations__ = {
        "values": Ref[ConcreteDefinition],
        "named": Ref[ConcreteDefinition],
    }
    boundary = compile_signature(target).prepare_args(
        (first, second), {"third": third}
    )

    assert boundary.authority["values"] == (first, second)
    assert boundary.authority["named"] == {"third": third}
    assert boundary.canonical["values"] == (first, second)
    assert boundary.canonical["named"] == {"third": third}
    assert boundary.deliver_args() == ((first, second), {"third": third})


def test_bound_variadics_reject_malformed_packed_values() -> None:
    """Already-bound records fail cleanly when variadic packing is invalid."""

    def positional(*values):
        return values

    def keyword(**values):
        return values

    with pytest.raises(SignatureError, match="must be a tuple"):
        compile_signature(positional).prepare_bound(BoundArguments((("values", 1),)))
    with pytest.raises(SignatureError, match="must be a mapping"):
        compile_signature(keyword).prepare_bound(BoundArguments((("values", {1: 2}),)))


def test_variadic_selections_use_exact_occurrence_keys_and_validate_before_reads() -> None:
    """Selections use ``(name, index-or-key)`` and invalid paths fail preflight."""

    cdef = _cdef()
    reference = ObjectRef(cdef, {})
    store = object()
    reads = 0

    def evidence(_cdef):
        nonlocal reads
        reads += 1
        candidate = SimpleNamespace(object_ref=reference, stores=(store,))
        return SimpleNamespace(declarations=(candidate,), states=())

    repo = SimpleNamespace(reference_evidence=evidence)

    def target(*values):
        return values

    target.__annotations__["values"] = Ref[ObjectRef]
    plan = compile_signature(target)
    boundary = plan.prepare_args(
        (cdef,), {}, repo=repo,
        selections={("values", 0): ReferenceSelection(reference, store)},
    )
    assert boundary.authority["values"] == (reference,)
    assert boundary.selections[("values", 0)].object_ref == reference
    assert boundary.selections[("values", 0)].store is store
    with pytest.raises(TypeError):
        boundary.selections[("values", 0)] = reference

    reads = 0
    with pytest.raises(SignatureError, match="selection controls are invalid"):
        plan.prepare_args(
            (cdef,), {}, repo=repo, selections={("values", 1): reference}
        )
    assert reads == 0
    with pytest.raises(SignatureError, match="selection controls are invalid"):
        plan.prepare_args((cdef,), {}, repo=repo, selections={"values": reference})
    assert reads == 0

    def keyword_target(**values):
        return values

    keyword_target.__annotations__["values"] = Ref[ObjectRef]
    keyword_boundary = compile_signature(keyword_target).prepare_args(
        (), {"item": cdef}, repo=repo,
        selections={("values", "item"): ReferenceSelection(reference, store)},
    )
    assert keyword_boundary.authority["values"] == {"item": reference}
    assert keyword_boundary.deliver_args() == ((), {"item": reference})


def test_direct_live_objects_need_no_repo_in_supported_containers() -> None:
    """Already-live Mat values stay local while frozen structural values need Repo."""

    live = SignatureObject()
    direct = _plan(Mat[Object]).prepare_args((live,), {}).deliver_args()[0][0]
    nested = FrozenDict({"plain": [live], "frozen": (live,)})
    delivered = _plan(object).prepare_args((nested,), {}).deliver_args()[0][0]

    assert direct is live
    assert delivered is nested
    with pytest.raises(SignatureError, match="requires a repo"):
        _plan(object).prepare_args(
            (FrozenDict({"value": _cdef()}),), {}
        ).deliver_args()


def test_active_finalized_links_deliver_compatible_targets_and_reject_conflicts() -> None:
    """Persisted links cannot leak through an active declared call boundary."""

    cdef = _cdef()
    ref = DefLink.finalized(EdgeKind.REF, cdef)
    mat = DefLink.finalized(EdgeKind.MATERIALIZE, cdef)

    assert _plan(Ref[ConcreteDefinition]).prepare_args(
        (ref,), {}
    ).deliver_args()[0][0] is cdef
    realized = _plan(Mat[ConcreteDefinition]).prepare_args(
        (mat,), {}, repo=Repo()
    ).deliver_args()[0][0]
    assert isinstance(realized, SignatureObject)
    with pytest.raises(SignatureError, match="conflicts"):
        _plan(Mat[ConcreteDefinition]).prepare_args((ref,), {})
    with pytest.raises(SignatureError, match="conflicts"):
        _plan(Ref[ConcreteDefinition]).prepare_args((mat,), {})
    with pytest.raises(SignatureError, match="conflicts"):
        _plan(Mat[ConcreteDefinition]).prepare_bound(
            BoundArguments((("value", ref),))
        )


@pytest.mark.parametrize("value", (1, [1], Definition(SignatureObject)))
def test_auto_ref_rejects_values_outside_its_documented_family(value: object) -> None:
    """AutoRef failures are public SignatureErrors, never private control flow."""

    with pytest.raises(SignatureError, match="requested authority is unavailable"):
        _plan(Ref[AutoRef]).prepare_args((value,), {})


def test_borrowed_boundaries_expire_and_snapshots_are_immutable() -> None:
    """Delayed delivery retains lease lifetime while explicit plans ignore ambient."""

    import dryml.core.signatures as signatures

    plan = _plan(Mat[Definition])
    selections: dict[object, object] = {}
    with signature_context(repo=Repo(), selections=selections) as context:
        controls = signatures._ambient_controls()
        boundary = plan._prepare_args((Definition(SignatureObject),), {}, controls)
        selections["value"] = object()
        assert not context.selections
        with pytest.raises(TypeError):
            plan.slots["value"] = plan.slots["value"]
        with pytest.raises(TypeError):
            context.selections["value"] = object()
        explicit = plan.prepare_args((Definition(SignatureObject),), {})
        with pytest.raises(SignatureError, match="requires a repo"):
            explicit.deliver_args()

    with pytest.raises(SignatureError, match="borrowed context is unavailable"):
        boundary.deliver_args()


def test_boundary_delivery_rejects_foreign_threads() -> None:
    """Invocation-owned boundaries cannot move to another thread."""

    boundary = _plan(object).prepare_args((1,), {})
    errors = []

    def deliver() -> None:
        try:
            boundary.deliver_args()
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=deliver)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], SignatureError)
    assert "another thread or task" in str(errors[0])


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
