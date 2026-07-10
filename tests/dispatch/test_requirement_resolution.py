from __future__ import annotations

import json
import os

import dryml

from dryml.dispatch import NormalizedDispatchTarget, normalize_user_operation, resolve_dispatch_plan
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.utils.general import pickle_save
from dryml.formats.refs import format_cdef_id
from dryml.operations import make_function_call_spec, make_method_call_spec


@dryml.env.req(requirements=("resolution-test-package>=1",))
@dryml.world.default(cpus=1)
@dryml.runtime.default(device_visibility={"policy": "assigned"})
def annotated_target():
    return None


def test_live_callable_resolution_preserves_all_namespace_requirements_and_sources():
    normalized = normalize_user_operation(annotated_target, allow_pickle=True)
    resolution = resolve_dispatch_plan(normalized, requirement_policy="ignore")

    data = resolution.requirements.to_data()
    assert data["environment_requirement"]["requirements"] == ["resolution-test-package>=1"]
    assert data["world_default"]["roles"]["main"]["replicas"] == 1
    assert data["runtime_default"]["device_visibility"] == {"policy": "assigned"}
    assert data["source_traces"]
    assert json.loads(json.dumps(resolution.to_data()))["requirement_policy"] == "ignore"


def test_definition_method_resolution_uses_authoritative_class_method_collection():
    @dryml.env.req(requirements=("class-requirement>=1",))
    class Target:
        @dryml.env.req(requirements=("method-requirement>=1",))
        def train(self):
            return None

    normalized = NormalizedDispatchTarget(
        make_function_call_spec("operator:truth"),
        subject_class=Target,
        method_name="train",
        transport="method_call",
    )
    # The resolver consumes the normalized subject class and delegates all MRO
    # and descriptor handling to annotations.fragments_for_method.
    resolution = resolve_dispatch_plan(normalized, requirement_policy="ignore")
    assert resolution.requirements.environment_requirement.requirements == (
        "class-requirement>=1",
        "method-requirement>=1",
    )


def test_explicit_method_operation_recovers_stored_definition_requirements(tmp_path):
    @dryml.env.req(requirements=("stored-class>=1",))
    class Target:
        @dryml.env.req(requirements=("stored-method>=1",))
        def train(self):
            return None

    cdef = ConcreteDefinition(Target)
    subject = format_cdef_id(cdef.stable_hash())

    class Store:
        def object_dir_for_cdef_id(self, value):
            assert value == subject
            return os.fspath(tmp_path)

    pickle_save(cdef, tmp_path / "def.pkl")
    normalized = normalize_user_operation(make_method_call_spec(subject, "train"), store=Store())
    resolution = resolve_dispatch_plan(normalized, requirement_policy="ignore")

    assert normalized.definition_target is not None
    assert normalized.subject_class.__name__ == "Target"
    assert resolution.requirements.environment_requirement.requirements == ("stored-class>=1", "stored-method>=1")


def test_live_target_always_includes_code_analysis_facts():
    resolution = resolve_dispatch_plan(normalize_user_operation(annotated_target, allow_pickle=True), requirement_policy="ignore")

    assert resolution.code_analysis is not None
    assert resolution.code_analysis.facts
