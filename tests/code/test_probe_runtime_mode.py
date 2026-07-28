from __future__ import annotations

import dryml.code as code
import dryml.environments as environments
import dryml.runtime as runtime
import dryml.worlds as worlds
from dryml.code.analysis import CodeAnalysisResult, CodeAnalysisContext, FunctionAnalyzer, register_analyzer
from dryml.code.facts import CodeFact
from dryml.code.targets import CodeTarget
from dryml.runtime.context import set_runtime


TARGET = "probe_targets:plain_function"


def _runtime_analyzer(target: CodeTarget, context: CodeAnalysisContext) -> CodeAnalysisResult:
    state = runtime.active_runtime()
    return CodeAnalysisResult(
        target=target.spec,
        facts=(CodeFact(
            kind="runtime_state",
            data={
                "mode": state.mode.value,
                "allocation_is_no_allocation": state.allocation is runtime.NoAllocation,
                "enforcement": state.enforcement.value,
                "allow_dynamic_execution": context.allow_dynamic_execution,
            },
        ),),
    )


register_analyzer(FunctionAnalyzer("probe_runtime_state", _runtime_analyzer), replace=True)


def test_probe_runtime_mode_and_no_allocation():
    result = code.probe_target(TARGET, algorithms=("probe_runtime_state",), include_environment_record=False)
    fact = result.analysis.facts_of_kind("runtime_state")[0]

    assert result.ok
    assert fact.data["mode"] == "probe"
    assert fact.data["allocation_is_no_allocation"] is True
    assert fact.data["allow_dynamic_execution"] is False


def test_probe_does_not_require_world_or_mutate_current_contexts():
    env_spec = environments.CurrentEnvironmentSpec()
    world_marker = object()

    with environments.use(env_spec), worlds.use(world_marker):
        result = code.probe_target(TARGET, include_environment_record=False)
        assert result.ok
        assert environments.current() is env_spec
        assert worlds.current() is world_marker


def test_in_process_probe_restores_previous_runtime_state():
    with runtime.plain() as state:
        result = code.probe_target(TARGET, algorithms=("probe_runtime_state",), include_environment_record=False)
        assert result.ok
        assert runtime.active_runtime() is state


def test_probe_uses_explicit_strict_enforcement_under_runtime_overrides():
    for policy in (runtime.RuntimeEnforcement.STRICT, runtime.RuntimeEnforcement.WARN, runtime.RuntimeEnforcement.OFF):
        token = set_runtime(runtime.RuntimeState(enforcement=policy))
        try:
            result = code.probe_target(TARGET, algorithms=("probe_runtime_state",), include_environment_record=False)
        finally:
            runtime.reset_runtime(token)
        fact = result.analysis.facts_of_kind("runtime_state")[0]
        assert result.ok
        assert fact.data["enforcement"] == runtime.RuntimeEnforcement.STRICT.value
