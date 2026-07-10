from __future__ import annotations

from dataclasses import replace

from dryml.environments import CurrentEnvironmentSpec, EnvironmentProbeResult, EnvironmentRegistry, EnvironmentRequirement, PythonExecutableSpec, inspect_current, resolve


def test_resolve_without_requirement_selects_first_candidate_without_probe():
    calls = []

    result = resolve(
        None,
        candidates=(PythonExecutableSpec("/first/python"),),
        include_current=False,
        probe_runner=lambda *args, **kwargs: calls.append(args),
    )

    assert result.ok
    assert result.selected_source == "candidate"
    assert calls == []


def test_resolve_uses_registry_name_order_for_no_requirement_candidates():
    registry = EnvironmentRegistry()
    registry.register("zeta", PythonExecutableSpec("/z/python"))
    registry.register("alpha", PythonExecutableSpec("/a/python"))

    result = resolve(None, registry=registry, include_current=False)

    assert result.selected_name == "alpha"


def test_resolve_deduplicates_after_selected_candidate():
    spec = CurrentEnvironmentSpec()

    result = resolve(None, candidates=(spec, spec.to_data()), include_current=False)

    assert result.selected == spec


def test_resolve_records_incompatible_duplicate_then_selected_and_reuses_record():
    calls = []
    requirement = EnvironmentRequirement(tags=("selected",))
    first = PythonExecutableSpec("/a/python")
    second = PythonExecutableSpec("/b/python")

    def runner(spec, *, timeout):
        calls.append((spec.executable, timeout))
        record = replace(inspect_current(), tags=("selected",) if spec.executable.endswith("b/python") else ())
        return EnvironmentProbeResult(spec, True, record=record)

    result = resolve(requirement, candidates=(first, first.to_data(), second), include_current=False, probe_runner=runner)

    assert [attempt.status for attempt in result.attempts] == ["incompatible", "duplicate", "selected"]
    assert result.selected == second
    assert result.selected_record is not None and result.selected_record.tags == ("selected",)
    assert [call[0] for call in calls] == ["/a/python", "/b/python"]


def test_resolve_prefilters_registry_labels_and_enforces_candidate_limit():
    registry = EnvironmentRegistry()
    registry.register("ignored", PythonExecutableSpec("/ignored/python"), tags=("other",))
    calls = []
    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(PythonExecutableSpec("/one/python"), PythonExecutableSpec("/two/python")),
        registry=registry,
        include_current=False,
        max_candidates=1,
        probe_runner=lambda spec, *, timeout: calls.append(spec) or EnvironmentProbeResult(spec, False),
    )

    assert [attempt.status for attempt in result.attempts] == ["probe_failed", "not_considered_limit", "not_considered_limit"]
    assert len(calls) == 1


def test_resolve_stops_lazy_candidates_after_requirement_free_selection():
    def candidates():
        yield CurrentEnvironmentSpec()
        raise AssertionError("resolver consumed a candidate after selection")

    result = resolve(None, candidates=candidates(), include_current=False)

    assert result.ok


def test_resolver_metadata_redacts_environment_overrides():
    result = resolve(None, candidates=(PythonExecutableSpec("/python", env={"TOKEN": "secret"}),), include_current=False)

    assert "TOKEN" not in str(result.to_data())
