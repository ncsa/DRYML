import os
import json
import subprocess
import sys

import pytest

import dryml.runtime as runtime
from dryml.runtime.enforcement import default_enforcement_from_env
from dryml.runtime.errors import FrameworkImportSafetyError, NoAllocationError, RuntimeTransitionError


@pytest.fixture(autouse=True)
def reset_runtime_state():
    runtime.reset_runtime()
    yield
    runtime.reset_runtime()


def test_default_enforcement_is_off():
    assert runtime.enforcement() is runtime.RuntimeEnforcement.OFF
    assert runtime.active_runtime().enforcement is runtime.RuntimeEnforcement.OFF


def test_runtime_enforcement_values_are_stable():
    assert runtime.RuntimeEnforcement.STRICT.value == "strict"
    assert runtime.RuntimeEnforcement.WARN.value == "warn"
    assert runtime.RuntimeEnforcement.OFF.value == "off"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("strict", runtime.RuntimeEnforcement.STRICT),
        ("warn", runtime.RuntimeEnforcement.WARN),
        ("off", runtime.RuntimeEnforcement.OFF),
        ("STRICT", runtime.RuntimeEnforcement.STRICT),
        (runtime.RuntimeEnforcement.STRICT, runtime.RuntimeEnforcement.STRICT),
    ],
)
def test_set_enforcement_accepts_strings_and_enum(value, expected):
    before = runtime.active_runtime()

    assert runtime.set_enforcement(value) is expected

    after = runtime.active_runtime()
    assert after.enforcement is expected
    assert after.mode is before.mode
    assert after.allocation is before.allocation


@pytest.mark.parametrize("value", ["disabled", "ignore", "false", None])
def test_set_enforcement_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="strict, warn, or off"):
        runtime.set_enforcement(value)


def test_enable_and_disable_helpers():
    assert runtime.disable() is runtime.RuntimeEnforcement.OFF
    assert runtime.enforcement() is runtime.RuntimeEnforcement.OFF
    assert runtime.enable() is runtime.RuntimeEnforcement.STRICT
    assert runtime.enforcement() is runtime.RuntimeEnforcement.STRICT


def test_disabled_context_restores_process_baseline_after_normal_exit():
    before = runtime.active_runtime()

    with runtime.disabled():
        assert runtime.enforcement() is runtime.RuntimeEnforcement.OFF
        assert runtime.active_runtime().mode is before.mode
        assert runtime.active_runtime().allocation is before.allocation

    assert runtime.enforcement() is runtime.RuntimeEnforcement.OFF


def test_disabled_context_restores_warn_and_nested_contexts():
    runtime.set_enforcement("warn")

    with runtime.disabled():
        assert runtime.enforcement() is runtime.RuntimeEnforcement.OFF
        runtime.set_enforcement("warn")
        with runtime.disabled():
            assert runtime.enforcement() is runtime.RuntimeEnforcement.OFF
        assert runtime.enforcement() is runtime.RuntimeEnforcement.WARN

    assert runtime.enforcement() is runtime.RuntimeEnforcement.WARN


def test_disabled_context_restores_after_exception():
    runtime.set_enforcement("warn")

    with pytest.raises(RuntimeError, match="boom"):
        with runtime.disabled():
            assert runtime.enforcement() is runtime.RuntimeEnforcement.OFF
            raise RuntimeError("boom")

    assert runtime.enforcement() is runtime.RuntimeEnforcement.WARN


@pytest.mark.parametrize(("value", "expected"), [("off", "off"), ("warn", "warn"), ("strict", "strict"), ("OFF", "off")])
def test_environment_variable_initializes_default_enforcement(value, expected):
    env = dict(os.environ, DRYML_RUNTIME_ENFORCEMENT=value)
    proc = subprocess.run(
        [sys.executable, "-c", "import dryml.runtime as runtime; print(runtime.enforcement().value)"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert proc.stdout.strip() == expected


def test_fresh_python_baseline_leaves_inherited_visibility_unchanged():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="inherited-device")
    env.pop("DRYML_RUNTIME_ENFORCEMENT", None)
    env.pop(runtime.BOOTSTRAP_MARKER_ENV, None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, os; import dryml.runtime as runtime; "
            "print(json.dumps({'mode': runtime.active_runtime().mode.value, "
            "'enforcement': runtime.enforcement().value, "
            "'visibility': os.environ['CUDA_VISIBLE_DEVICES'], "
            "'bootstrap': runtime.BOOTSTRAP_MARKER_ENV in os.environ}))",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert json.loads(proc.stdout) == {
        "mode": "orchestrator",
        "enforcement": "off",
        "visibility": "inherited-device",
        "bootstrap": False,
    }


def test_invalid_environment_variable_warns_and_falls_back_to_off():
    with pytest.warns(RuntimeWarning, match="falling back to off"):
        assert default_enforcement_from_env({"DRYML_RUNTIME_ENFORCEMENT": "disabled"}) is runtime.RuntimeEnforcement.OFF


def test_guards_raise_in_strict_mode():
    runtime.enable()
    with pytest.raises(NoAllocationError):
        runtime.require_allocation("training")
    with runtime.enter_runtime(runtime.RuntimeMode.WORKER, runtime.RuntimeAllocationView(cpus=(0,))):
        with pytest.raises(RuntimeTransitionError):
            runtime.assert_no_workload_allocation()


def test_guard_warns_in_warn_mode_and_returns_neutral_value():
    runtime.set_enforcement("warn")

    with pytest.warns(RuntimeWarning, match="workload allocation is required"):
        assert runtime.require_allocation("training") is runtime.NoAllocation
    with pytest.warns(RuntimeWarning, match="framework import requires active runtime bootstrap"):
        runtime.assert_framework_import_configured("fakeframework")


def test_guard_bypasses_in_off_mode_without_creating_allocation():
    runtime.disable()

    assert runtime.require_allocation("training") is runtime.NoAllocation
    assert runtime.require_workload_allocation("materialize") is runtime.NoAllocation
    runtime.assert_framework_import_safe("fakeframework")


def test_off_mode_does_not_bypass_non_enforcement_errors():
    runtime.disable()

    with pytest.raises(ModuleNotFoundError):
        runtime.import_configured_framework("definitely_missing_dryml_test_framework")
