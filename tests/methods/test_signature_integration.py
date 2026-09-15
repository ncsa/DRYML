"""Integration coverage for Method calls through core signature boundaries."""

import numpy as np
import pytest

from dryml.core import ConcreteDefinition, Definition, Object
from dryml.core.links import DefLink
from dryml.core.reference_values import ObjectRef
from dryml.core.signatures import Mat, Ref, SignatureError
from dryml.methods import Method, PreparedCallMismatchError, traits


class SignatureValue(Object):
    """Minimal Object category used to create inert definition values."""


def _definition() -> ConcreteDefinition:
    """Return a concrete definition that does not require a Repo to normalize."""

    return Definition(SignatureValue).concretize()


def test_selected_targets_compile_once_and_normalize_every_fresh_return(monkeypatch):
    """Carrier, eager, learning, and cached calls share selected signature activation."""

    value = _definition()

    class Direct(Method):
        def __call__(self, value: Ref[ConcreteDefinition]) -> Ref[ConcreteDefinition]:
            return value

    class Alternative(Method):
        @traits()
        def generic(self, value: Ref[ConcreteDefinition]) -> Ref[ConcreteDefinition]:
            return value

    class Learning(Method):
        @traits()
        def generic(self, tensor) -> Ref[ConcreteDefinition]:
            return value

    from dryml.core import signatures

    compiled, returns = [], []
    original_compile = signatures.compile_signature
    original_return = signatures.SignaturePlan.prepare_return

    def compile_once(target, **kwargs):
        compiled.append(target)
        return original_compile(target, **kwargs)

    def observe_return(self, value, **kwargs):
        returns.append(value)
        return original_return(self, value, **kwargs)

    monkeypatch.setattr(signatures, "compile_signature", compile_once)
    monkeypatch.setattr(signatures.SignaturePlan, "prepare_return", observe_return)

    assert object.__new__(Direct)(value) == value
    alternative = object.__new__(Alternative)
    assert alternative.find_implementation()(value) == value
    assert alternative(value) == value
    learning = object.__new__(Learning)
    learning.learn()
    assert learning(np.ones((2,), dtype=np.float32)) == value
    assert learning(np.ones((2,), dtype=np.float32)) == value

    assert len(compiled) == 4
    assert returns == [value] * 5


def test_tensor_selection_precedes_selected_annotation_normalization_without_materialization():
    """Traits select from raw tensor facts while Ref values remain metadata-only."""

    class Variants(Method):
        @traits(backend="numpy")
        def numpy(self, tensor, value: Ref[ConcreteDefinition]) -> Ref[ConcreteDefinition]:
            return value

        @traits(backend="torch")
        def torch(self, tensor, value: Mat[ConcreteDefinition]) -> Mat[ConcreteDefinition]:
            raise AssertionError("the NumPy tensor must select before this annotation is relevant")

    class ReferenceOnly(Method):
        @traits()
        def generic(self, value: Ref[ConcreteDefinition]) -> Ref[ConcreteDefinition]:
            return value

    definition = _definition()
    reference = ObjectRef(definition, {})

    assert object.__new__(Variants)(np.ones((2,), dtype=np.float32), definition) == definition
    assert object.__new__(ReferenceOnly)(reference) == definition


def test_learning_publishes_only_after_input_normalization_but_before_target_or_return_failure():
    """Failed input conversion retries learning, while later failures retain its cache."""

    definition = _definition()

    class Input(Method):
        @traits()
        def generic(self, tensor) -> Ref[ConcreteDefinition]:
            return definition

    class Target(Method):
        @traits()
        def generic(self, tensor) -> Ref[ConcreteDefinition]:
            raise ValueError("target failure")

    class Return(Method):
        @traits()
        def generic(self, tensor) -> Ref[ConcreteDefinition]:
            return Mat(definition)

    input_method = object.__new__(Input)
    input_method.learn()
    with pytest.raises(SignatureError, match="argument binding failed"):
        input_method()
    assert input_method.call_mode == "learning"
    assert input_method.cached_signature is None

    for method, error in ((object.__new__(Target), ValueError), (object.__new__(Return), SignatureError)):
        method.learn()
        with pytest.raises(error):
            method(np.ones((2,), dtype=np.float32))
        assert method.call_mode == "cached"
        assert method.cached_signature is not None


def test_cached_calls_validate_raw_tensor_signature_before_retained_plan_normalization(monkeypatch):
    """Cached raw validation precedes retained-plan delivery and never rediscovers catalogs."""

    definition = _definition()

    class Cached(Method):
        @traits(backend="numpy")
        def numpy(self, tensor) -> Ref[ConcreteDefinition]:
            return definition

    from dryml.core import signatures

    method = object.__new__(Cached)
    method.learn()
    assert method(np.ones((2,), dtype=np.float32)) == definition
    method.implementations = lambda: (_ for _ in ()).throw(AssertionError("catalog discovery"))
    monkeypatch.setattr(
        signatures,
        "compile_signature",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("signature compilation")),
    )

    with pytest.raises(PreparedCallMismatchError):
        method(np.ones((3,), dtype=np.float32))
    assert method(np.ones((2,), dtype=np.float32)) == definition


def test_cooperative_super_is_one_boundary_while_independent_method_calls_are_not(monkeypatch):
    """Only cooperative super bypasses a second conversion for the same Method boundary."""

    class Base(Method):
        def __call__(self, value: Ref[ConcreteDefinition]) -> Ref[ConcreteDefinition]:
            return value

    class Leaf(Base):
        def __call__(self, value: Ref[ConcreteDefinition]) -> Ref[ConcreteDefinition]:
            return super().__call__(value)

    class Inner(Method):
        def __call__(self, value: Ref[ConcreteDefinition]) -> Ref[ConcreteDefinition]:
            return value

    inner = object.__new__(Inner)

    class Outer(Method):
        def __call__(self, value: Ref[ConcreteDefinition]) -> Ref[ConcreteDefinition]:
            return inner(value)

    from dryml.core import signatures

    compiled = []
    original_compile = signatures.compile_signature

    def observe_compile(target, **kwargs):
        compiled.append(target)
        return original_compile(target, **kwargs)

    value = _definition()
    monkeypatch.setattr(signatures, "compile_signature", observe_compile)

    assert object.__new__(Leaf)(value) == value
    assert len(compiled) == 1
    assert object.__new__(Outer)(value) == value
    assert len(compiled) == 3


def test_selected_signature_keeps_target_control_keywords_and_descriptor_forms():
    """Explicit context controls never consume target keywords or alter native binding."""

    class Controls(Method):
        @traits()
        def generic(
            self,
            value,
            *,
            repo=None,
            cache=None,
            reuse_live=None,
        ):
            assert (repo, cache, reuse_live) == (1, 2, 3)
            return value

    value = _definition()
    tensor = np.ones((2,), dtype=np.float32)
    assert object.__new__(Controls)(tensor, repo=1, cache=2, reuse_live=3) is tensor
    assert isinstance(Mat(value), DefLink)
