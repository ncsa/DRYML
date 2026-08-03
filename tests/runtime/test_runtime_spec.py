import pytest

import dryml.runtime as runtime
from dryml.runtime.errors import RuntimeSpecError


def test_runtime_spec_canonical_id_stability():
    left = runtime.attach_runtime_id(
        runtime.make_runtime_spec(mode="worker", device_visibility={"policy": "assigned", "accelerators": ["gpu"]}, frameworks={"plain": {"num_threads": 8}})
    )
    right = runtime.attach_runtime_id(
        runtime.make_runtime_spec(frameworks={"plain": {"num_threads": 8}}, device_visibility={"accelerators": ["gpu"], "policy": "assigned"}, mode=runtime.RuntimeMode.WORKER)
    )

    assert left["id"].startswith("runtime-v1-")
    assert left["id"] == right["id"]
    assert runtime.compute_runtime_id(left) == left["id"]


def test_runtime_spec_id_changes_with_semantic_content_and_rejects_bad_payload():
    left = runtime.attach_runtime_id(runtime.make_runtime_spec(mode="worker", device_visibility={"policy": "assigned"}))
    changed = runtime.attach_runtime_id(runtime.make_runtime_spec(mode="inline", device_visibility={"policy": "explicit", "devices": {"gpu": [0]}}))
    assert left["id"] != changed["id"]

    bad = runtime.make_runtime_spec()
    bad["payload"] = {"mode": "bad"}
    with pytest.raises(RuntimeSpecError):
        runtime.validate_runtime_spec(bad)


def test_runtime_spec_rejects_non_mapping_framework_config():
    with pytest.raises(RuntimeSpecError):
        runtime.make_runtime_spec(frameworks={"torch": "bad"})

    with pytest.raises(RuntimeSpecError):
        runtime.RuntimeContextSpec.from_data({"frameworks": {"torch": "bad"}})


def test_runtime_spec_serializes_explicit_none_without_reinterpreting_legacy_payloads():
    no_role = runtime.RuntimeContextSpec.from_data({"mode": "none"})
    legacy = runtime.RuntimeContextSpec.from_data({})

    assert no_role.to_data()["mode"] == "none"
    assert legacy.mode is runtime.RuntimeMode.ORCHESTRATOR
