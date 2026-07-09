from __future__ import annotations

import dryml

from dryml.dispatch import normalize_user_operation, resolve_dispatch_plan
from dryml.environments import CurrentEnvironmentSpec, PythonExecutableSpec, use


@dryml.env.default(CurrentEnvironmentSpec())
def target_with_environment_default():
    return None


def test_environment_selection_precedence_and_consideration_trace():
    explicit = PythonExecutableSpec("/usr/bin/python3")
    normalized = normalize_user_operation(target_with_environment_default, allow_pickle=True)
    with use(PythonExecutableSpec("/bin/python3")):
        resolution = resolve_dispatch_plan(normalized, environment=explicit, requirement_policy="ignore")

    selection = resolution.environment_selection
    assert selection.source == "explicit"
    assert selection.candidate["executable"] == "/usr/bin/python3"
    assert [item.status for item in selection.considered] == ["selected", "not_selected", "not_selected", "not_selected"]


def test_environment_default_wins_before_current_and_fallback():
    normalized = normalize_user_operation(target_with_environment_default, allow_pickle=True)
    resolution = resolve_dispatch_plan(normalized, requirement_policy="ignore")

    assert resolution.environment_selection.source == "annotation_default"
    assert resolution.environment_selection.candidate["kind"] == "current"
