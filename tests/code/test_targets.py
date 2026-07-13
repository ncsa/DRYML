from __future__ import annotations

import json
import sys
import types

import dryml.code as code


class CallableInstance:
    def __call__(self, value):
        return value


class HostileDescriptor:
    def __get__(self, instance, owner):
        raise AssertionError("target construction must not bind descriptors")


class HostileDescriptorOwner:
    method = HostileDescriptor()


class HostileTruthinessDescriptor:
    def __bool__(self):
        raise AssertionError("analysis must not truth-test raw descriptors")


class HostileTruthinessOwner:
    method = HostileTruthinessDescriptor()


class HostileBoundMeta(type):
    def __getattribute__(cls, name):
        if name == "__dict__":
            raise AssertionError("bound method normalization must not read owner.__dict__ dynamically")
        return super().__getattribute__(name)


class HostileBoundOwner(metaclass=HostileBoundMeta):
    def method(self):
        return None


class BoundMethodBase:
    def inherited(self):
        return None


class BoundMethodChild(BoundMethodBase):
    pass


def test_import_path_target_resolves(requirement_targets):
    target = code.target_from_import_path("dryml_requirement_targets:plain_importable_function")

    assert target.spec.kind == "import_path"
    assert target.obj is requirement_targets.plain_importable_function


def test_malformed_and_unresolved_import_paths_produce_diagnostics():
    malformed = code.target_from_import_path("not-an-import-path")
    unresolved = code.target_from_import_path("dryml_requirement_targets:missing")

    assert malformed.diagnostics[0].code == "dryml.code.import_path_invalid"
    assert unresolved.diagnostics[0].code == "dryml.code.qualname_resolution_failed"


def test_allow_import_false_keeps_import_path_spec_only():
    target = code.target_from_import_path("dryml_requirement_targets:plain_importable_function", allow_import=False)

    assert target.obj is None
    assert target.spec.import_path == "dryml_requirement_targets:plain_importable_function"


def test_function_lambda_local_callable_class_and_methods(requirement_targets):
    local = requirement_targets.make_local_training_function()


    assert code.normalize_target(requirement_targets.plain_importable_function).spec.kind == "function"
    assert code.normalize_target(requirement_targets.local_lambda_with_annotation).spec.kind == "lambda"
    assert code.normalize_target(local).spec.kind == "local_function"
    assert code.normalize_target(CallableInstance()).spec.kind == "callable_instance"
    assert code.normalize_target(requirement_targets.LightningModel).spec.kind == "class"
    assert code.normalize_target(requirement_targets.LightningModel().train).spec.kind == "bound_method"
    assert code.normalize_target(requirement_targets.LightningModel.train).spec.kind == "unbound_method"
    assert code.normalize_target(requirement_targets.ClassMethodTargets.inner_decorated).spec.kind == "class_method"
    assert code.normalize_target(requirement_targets.StaticMethodTargets.inner_decorated).spec.kind == "unbound_method"


def test_codetarget_and_spec_inputs_round_trip(requirement_targets):
    original = code.normalize_target(requirement_targets.plain_importable_function)
    spec = code.CodeTargetSpec.from_data(original.spec.to_data())

    assert code.normalize_target(original) is original
    assert code.normalize_target(spec).spec.to_data() == spec.to_data()
    json.dumps(original.spec.to_data())


def test_source_spec_and_unknown_targets_are_serializable():
    spec = code.CodeTargetSpec("source_spec", source_spec={"kind": "function", "source": "lambda x: x"})
    unknown = code.normalize_target(object())

    assert spec.to_data()["source_spec"]["kind"] == "function"
    assert unknown.spec.kind == "unknown"
    json.dumps(unknown.spec.to_data())


def test_source_spec_only_targets_report_source_unavailable_without_reconstruction():
    spec = code.CodeTargetSpec("source_spec", source_spec={"kind": "function", "source": "lambda x: x"})

    result = code.analyze(spec, algorithms=("source", "ast_access", "static_calls"))

    assert not result.facts
    assert result.diagnostics_of_code("dryml.code.source_unavailable")


def test_class_attribute_target_preserves_raw_descriptors(requirement_targets):
    classmethod_target = code.target_from_class_attribute(requirement_targets.ClassMethodTargets, "outer_decorated")
    staticmethod_target = code.target_from_class_attribute(requirement_targets.StaticMethodTargets, "outer_decorated")

    assert classmethod_target.spec.kind == "class_method"
    assert staticmethod_target.spec.kind == "static_method"
    assert classmethod_target.raw_descriptor is not classmethod_target.obj
    assert staticmethod_target.raw_descriptor is not staticmethod_target.obj
    assert classmethod_target.spec.method_name == "outer_decorated"
    assert staticmethod_target.spec.metadata["owner_qualname"] == "StaticMethodTargets"


def test_missing_class_attribute_target_diagnostic(requirement_targets):
    target = code.target_from_class_attribute(requirement_targets.StaticMethodTargets, "missing")

    assert target.spec.kind == "unknown"
    assert target.diagnostics[0].code == "dryml.code.class_attribute_missing"


def test_class_attribute_and_definition_targets_do_not_bind_descriptors():
    class_attribute = code.target_from_class_attribute(HostileDescriptorOwner, "method")
    definition_method = code.target_from_definition_method("subject", HostileDescriptorOwner, "method")

    assert class_attribute.raw_descriptor is HostileDescriptorOwner.__dict__["method"]
    assert class_attribute.obj is class_attribute.raw_descriptor
    assert definition_method.raw_descriptor is HostileDescriptorOwner.__dict__["method"]
    assert definition_method.obj is definition_method.raw_descriptor


def test_bound_method_normalization_avoids_metaclass_hooks_and_preserves_inherited_descriptor():
    hostile = code.normalize_target(HostileBoundOwner().method)
    inherited = code.normalize_target(BoundMethodChild().inherited)

    assert hostile.raw_descriptor is hostile.unwrapped
    assert inherited.owner is BoundMethodChild
    assert inherited.raw_descriptor is BoundMethodBase.__dict__["inherited"]


def test_import_path_resolution_does_not_invoke_module_or_descriptor_hooks(monkeypatch):
    module = types.ModuleType("hostile_code_target_module")

    def module_getattr(name):
        raise AssertionError("import-path resolution must not invoke module __getattr__")

    module.__getattr__ = module_getattr
    module.Owner = HostileDescriptorOwner
    monkeypatch.setitem(sys.modules, module.__name__, module)

    missing = code.target_from_import_path(f"{module.__name__}:missing")
    descriptor = code.target_from_import_path(f"{module.__name__}:Owner.method")

    assert missing.diagnostics[0].code == "dryml.code.qualname_resolution_failed"
    assert descriptor.raw_descriptor is HostileDescriptorOwner.__dict__["method"]


def test_import_disabled_source_analysis_reports_unavailable():
    result = code.analyze(
        "dryml_requirement_targets:plain_importable_function",
        algorithms=("source",),
        context=code.CodeAnalysisContext(allow_import=False),
    )

    assert result.diagnostics_of_code("dryml.code.source_unavailable")


def test_source_analyzers_do_not_truth_test_raw_descriptors():
    target = code.target_from_class_attribute(HostileTruthinessOwner, "method")

    for algorithm in ("source", "ast_access", "static_calls", "symbol_capture"):
        result = code.analyze(target, algorithms=(algorithm,))
        assert not result.diagnostics_of_code("dryml.code.algorithm_failed")


def test_direct_analysis_of_bound_method_preserves_owner_metadata():
    result = code.analyze(BoundMethodChild().inherited, algorithms=("callables",))
    fact = result.facts_of_kind("callable")[0]

    assert fact.data["is_bound_method"] is True
    assert fact.data["owner_qualname"] == "BoundMethodChild"
