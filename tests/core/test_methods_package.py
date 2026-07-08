from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from dryml.core2.backend import Backend
from dryml.core2.methods import BatchMode, CompilerInfo, Method, Traits, traits
from dryml.core2.methods.compiler_info import CompilerInfo as CompilerInfoFromModule
from dryml.core2.methods.method import Method as MethodFromModule
from dryml.core2.methods.traits import Traits as TraitsFromModule
from dryml.core2.tensor_spec import TensorSpec


class Echo(Method):
    @traits(backend=None)
    def call_default(self, value):
        return ("default", value)

    @traits(backend="numpy")
    def call_numpy(self, value):
        return ("numpy", value)


class BatchAware(Method):
    @traits(backend=None, batch_mode="element")
    def call_element(self, value):
        return "element"

    @traits(backend=None, batch_mode="batched")
    def call_batched(self, value):
        return "batched"


class ManualCall(Method):
    def __call__(self, value):
        return ("manual", value)

    @traits(backend=None)
    def call_default(self, value):
        return ("default", value)


class ChildEcho(Echo):
    @traits(backend="numpy", batch_mode="batched")
    def call_numpy_batched(self, value):
        return ("numpy_batched", value)


class Ambiguous(Method):
    @traits(backend=None)
    def first(self, value):
        return value

    @traits(backend=None)
    def second(self, value):
        return value


class NumpyOnly(Method):
    @traits(backend="numpy")
    def call_numpy(self, value):
        return value


def test_new_methods_imports_and_top_level_core2_exports():
    from dryml.core2 import CompilerInfo as CoreCompilerInfo
    from dryml.core2 import Method as CoreMethod
    from dryml.core2 import Traits as CoreTraits
    from dryml.core2 import traits as core_traits

    assert Method is MethodFromModule is CoreMethod
    assert Traits is TraitsFromModule is CoreTraits
    assert CompilerInfo is CompilerInfoFromModule is CoreCompilerInfo
    assert traits is core_traits


def test_traits_defaults_conversion_matching_specificity_and_hashability():
    default = Traits()
    numpy = Traits(backend="numpy")
    batched = Traits(batch_mode="batched")
    both = Traits(backend="numpy", batch_mode="batched")

    assert default.backend is None
    assert default.batch_mode is None
    assert numpy.backend is Backend.numpy
    assert batched.batch_mode is BatchMode.batched
    assert both.backend is Backend.numpy
    assert both.batch_mode is BatchMode.batched
    assert default.match(numpy)
    with pytest.raises(AttributeError):
        default.match(numpy, strict=True)
    assert not Traits(backend="numpy").match(Traits(backend="tf"))
    assert not Traits(batch_mode="batched").match(Traits(batch_mode="element"))
    assert [default.specificity, numpy.specificity, batched.specificity, both.specificity] == [0, 1, 1, 2]
    assert hash(both) == hash(Traits(backend=Backend.numpy, batch_mode=BatchMode.batched))
    with pytest.raises(NotImplementedError):
        default.match(object())


def test_traits_decorator_attaches_metadata_and_preserves_callable():
    selected = Traits(backend="numpy")

    @traits(traits=selected)
    def call_numpy(self, value):
        return value

    assert call_numpy.__dryml_traits__ is selected
    assert call_numpy(None, "value") == "value"


def test_method_subclass_collection_dispatch_and_user_call_preservation():
    assert len(Echo.__trait_impls__) == 2
    assert Echo().__call__ is not Method._dispatch_call
    assert Echo().resolve_impl(Traits(backend="numpy"))("x") == ("numpy", "x")
    assert Echo().resolve_impl(Traits(backend=None))("x") == ("numpy", "x")
    assert ManualCall()("x") == ("manual", "x")
    assert len(ManualCall.__trait_impls__) == 1
    assert len(ChildEcho.__trait_impls__) == 3


def test_method_resolution_errors_and_helpers():
    assert Echo().get_impl("numpy")("x") == ("numpy", "x")
    assert Echo().get_impl("tf")("x") == ("default", "x")
    assert Echo.get_impl_func("numpy") is Echo.call_numpy
    assert Echo.get_impl_func("tf") is Echo.call_default
    with pytest.raises(NotImplementedError):
        NumpyOnly().resolve_impl(Traits(backend="tf"))
    with pytest.raises(ValueError):
        Ambiguous().resolve_impl(Traits())


def test_batch_mode_resolution_from_hint_and_input_spec():
    method = BatchAware()

    assert method.resolve_impl_for("value", _hint_batched=True)("value") == "batched"
    assert method.resolve_impl_for("value", _hint_batched=False)("value") == "element"
    assert method.resolve_impl_for("value", input_spec=TensorSpec("float32", shape=(2,), batch=4))("value") == "batched"
    assert method.resolve_impl_for("value", input_spec=TensorSpec("float32", shape=(2,)))("value") == "element"


def test_infer_output_spec_default_raises():
    with pytest.raises(NotImplementedError, match="infer_output_spec"):
        Method().infer_output_spec()


def test_compiler_info_defaults_custom_values_and_frozen_behavior():
    info = CompilerInfo()
    custom = CompilerInfo(pure=False, elementwise=True, shape_preserving=True, opaque=True, static_argnames=("x",), tags=frozenset({"tag"}))

    assert info.pure is True
    assert info.elementwise is False
    assert info.shape_preserving is False
    assert info.opaque is False
    assert info.static_argnames == ()
    assert info.tags == frozenset()
    assert custom.static_argnames == ("x",)
    assert custom.tags == frozenset({"tag"})
    with pytest.raises(FrozenInstanceError):
        custom.pure = True


def test_core2_methods_import_does_not_import_dryml_code():
    script = """
import sys
import dryml.core2.methods
assert 'dryml.code' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
