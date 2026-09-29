from dryml.runtime import ControlStatus, DeviceVisibilityPolicy, RuntimeMode, build_control_plan, build_device_visibility_plan


def test_control_categories_are_independent_and_visibility_is_planned_before_imports():
    plan = build_control_plan(RuntimeMode.ORCHESTRATOR)
    assert plan.statuses["visibility"] == ControlStatus.PENDING_IMPORT
    assert plan.statuses["affinity"] == ControlStatus.NOT_APPLICABLE
    visibility = build_device_visibility_plan(mode=RuntimeMode.ORCHESTRATOR)
    assert visibility.policy is DeviceVisibilityPolicy.NONE
    assert visibility.env_updates["CUDA_VISIBLE_DEVICES"] == ""
