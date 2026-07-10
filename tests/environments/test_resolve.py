from __future__ import annotations

from dataclasses import replace

import pytest

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

    assert [attempt.status for attempt in result.attempts] == ["probe_failed", "not_considered_limit"]
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


def test_total_timeout_bounds_a_probe_without_an_explicit_probe_timeout():
    observed = []
    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(PythonExecutableSpec("/first/python"),),
        include_current=False,
        probe_timeout=None,
        total_timeout=0.25,
        probe_runner=lambda spec, *, timeout: observed.append(timeout) or EnvironmentProbeResult(spec, False),
    )

    assert result.status == "no_match"
    assert observed and 0 < observed[0] <= 0.25


def test_resolver_metadata_bounds_large_candidate_values():
    result = resolve(None, candidates=(PythonExecutableSpec("x" * 5000),), include_current=False)

    assert len(result.to_data()["selected"]["executable"]) == 4096


def test_resolver_continues_after_the_bounded_duplicate_trace():
    requirement = EnvironmentRequirement(tags=("selected",))
    rejected = PythonExecutableSpec("/rejected/python")
    selected = PythonExecutableSpec("/selected/python")

    def runner(spec, *, timeout):
        return EnvironmentProbeResult(spec, True, record=replace(inspect_current(), tags=("selected",) if spec == selected else ()))

    result = resolve(
        requirement,
        candidates=(rejected, *((rejected.to_data(),) * 32), selected),
        include_current=False,
        probe_runner=runner,
    )

    assert result.selected == selected
    assert len(result.attempts) == 32
    assert any(issue.code == "resolver_trace_truncated" for issue in result.diagnostics)


def test_resolver_continues_after_a_invalid_probe_runner_result():
    requirement = EnvironmentRequirement(tags=("selected",))
    rejected = PythonExecutableSpec("/invalid/python")
    selected = PythonExecutableSpec("/selected/python")

    result = resolve(
        requirement,
        candidates=(rejected, selected),
        include_current=False,
        probe_runner=lambda spec, *, timeout: None if spec == rejected else EnvironmentProbeResult(spec, True, record=replace(inspect_current(), tags=("selected",))),
    )

    assert result.selected == selected
    assert result.attempts[0].status == "probe_failed"
    assert result.attempts[0].diagnostics[0].message == "environment probe raised TypeError"


def test_resolver_does_not_serialize_probe_exception_secrets():
    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(PythonExecutableSpec("/first/python", env={"TOKEN": "secret"}),),
        include_current=False,
        probe_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("TOKEN=secret")),
    )

    assert "TOKEN" not in str(result.to_data())
    assert "secret" not in str(result.to_data())


def test_resolver_bounds_duplicate_only_candidate_input():
    rejected = PythonExecutableSpec("/rejected/python")

    def candidates():
        yield rejected
        for _ in range(33):
            yield rejected.to_data()
        raise AssertionError("resolver consumed an unbounded duplicate input")

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=candidates(),
        include_current=False,
        max_candidates=1,
        probe_runner=lambda spec, *, timeout: EnvironmentProbeResult(spec, False),
    )

    assert result.status == "no_match"
    assert any(issue.code == "resolver_candidates_truncated" for issue in result.diagnostics)


def test_resolver_rejects_invalid_candidate_before_running_any_probe():
    calls = []

    with pytest.raises(ValueError, match="invalid environment resolver candidate"):
        resolve(
            EnvironmentRequirement(tags=("wanted",)),
            candidates=(PythonExecutableSpec("/valid/python"), object()),
            include_current=False,
            probe_runner=lambda *args, **kwargs: calls.append(args),
        )

    assert calls == []


def test_resolver_rejects_malformed_injected_probe_result():
    spec = PythonExecutableSpec("/candidate/python")
    malformed = EnvironmentProbeResult(spec, "true")  # type: ignore[arg-type]

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(spec,),
        include_current=False,
        probe_runner=lambda *_args, **_kwargs: malformed,
    )

    assert result.status == "no_match"
    assert result.attempts[0].status == "probe_failed"
