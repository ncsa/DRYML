from __future__ import annotations

import json

import dryml.code as code


class CallableInstance:
    def __call__(self, value):
        return value


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
