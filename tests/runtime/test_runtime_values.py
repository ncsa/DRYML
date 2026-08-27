import pytest

from dryml.runtime import RuntimeContextSpec, RuntimeMode, check_runtime_spec_satisfies_requirement
from dryml.runtime.errors import RuntimeSpecError


def test_runtime_context_identity_excludes_metadata_and_validates_allocation_association():
    first = RuntimeContextSpec(RuntimeMode.INLINE, {"policy": "assigned"}, {"torch": {"threads": 1}}, {"memory": "1GiB"}, {"A": "x"}, {"note": "one"}, "worldalloc-v1.1-" + "0" * 64)
    second = RuntimeContextSpec(RuntimeMode.INLINE, {"policy": "assigned"}, {"torch": {"threads": 1}}, {"memory": "1GiB"}, {"A": "x"}, {"note": "two"}, "worldalloc-v1.1-" + "0" * 64)
    assert first.semantic_id == second.semantic_id
    data = first.to_data()
    assert set(data["payload"]) == {"mode", "device_visibility", "frameworks", "limits", "env", "metadata", "world_allocation_id"}
    assert RuntimeContextSpec.from_data(data) == first
    with pytest.raises(RuntimeSpecError, match="envelope"):
        RuntimeContextSpec.from_data(data["payload"])
    with pytest.raises(RuntimeSpecError):
        RuntimeContextSpec(world_allocation_id="worldalloc-v1.1-not-an-id")


def test_runtime_compatibility_is_a_declaration_check_only():
    value = RuntimeContextSpec(RuntimeMode.ORCHESTRATOR, framework={"torch": {"threads": 1}})
    assert check_runtime_spec_satisfies_requirement(value, {"framework": {"torch": {"threads": 1}}}).ok
    assert not check_runtime_spec_satisfies_requirement(value, {"framework": {"torch": {"threads": 2}}}).ok
