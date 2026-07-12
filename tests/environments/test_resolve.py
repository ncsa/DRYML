from __future__ import annotations

from dataclasses import replace
import json

import pytest

from dryml.environments import CompatibilityIssue, CondaEnvironmentSpec, ContainerEnvironmentSpec, CurrentEnvironmentSpec, EnvironmentProbeResult, EnvironmentRegistry, EnvironmentRequirement, PythonExecutableSpec, inspect_current, resolve
from dryml.environments.compatibility import report_from_issues


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


def test_resolve_without_requirement_skips_unsupported_container_candidate():
    result = resolve(
        None,
        candidates=(ContainerEnvironmentSpec("example/image"), CurrentEnvironmentSpec()),
        include_current=False,
    )

    assert isinstance(result.selected, CurrentEnvironmentSpec)
    assert result.attempts[0].status == "unsupported"


def test_resolve_without_requirement_skips_unlaunchable_conda_candidate():
    result = resolve(
        None,
        candidates=(CondaEnvironmentSpec(name="only-name"), CurrentEnvironmentSpec()),
        include_current=False,
    )

    assert isinstance(result.selected, CurrentEnvironmentSpec)
    assert result.attempts[0].status == "unsupported"


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
    assert result.status == "incomplete"


def test_resolve_orders_candidates_then_registry_and_prefilters_registry_labels():
    registry = EnvironmentRegistry()
    ignored = PythonExecutableSpec("/ignored/python")
    selected = PythonExecutableSpec("/selected/python")
    registry.register("ignored", ignored, tags=("other",))
    registry.register("selected", selected, tags=("wanted",))
    rejected = PythonExecutableSpec("/candidate/python")
    calls = []

    def runner(spec, *, timeout):
        calls.append(spec)
        return EnvironmentProbeResult(
            spec,
            True,
            record=replace(inspect_current(), tags=("wanted",) if spec == selected else ()),
        )

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(rejected,),
        registry=registry,
        include_current=True,
        probe_runner=runner,
    )

    assert result.selected == selected
    assert [(attempt.source, attempt.status) for attempt in result.attempts] == [
        ("candidate", "incompatible"),
        ("registry", "label_mismatch"),
        ("registry", "selected"),
    ]
    assert calls == [rejected, selected]


def test_resolve_records_an_all_incompatible_no_match_trace():
    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(PythonExecutableSpec("/candidate/python"),),
        include_current=False,
        probe_runner=lambda spec, *, timeout: EnvironmentProbeResult(spec, True, record=replace(inspect_current(), tags=())),
    )

    assert result.status == "no_match"
    assert [attempt.status for attempt in result.attempts] == ["incompatible"]


def test_resolve_stops_lazy_candidates_after_requirement_free_selection():
    def candidates():
        yield CurrentEnvironmentSpec()
        raise AssertionError("resolver consumed a candidate after selection")

    result = resolve(None, candidates=candidates(), include_current=False)

    assert result.ok


def test_resolve_defers_registry_iterator_until_needed():
    class UnusedRegistry:
        def iter_entries(self):
            raise AssertionError("resolver invoked an unused registry callback")

    result = resolve(
        None,
        candidates=(CurrentEnvironmentSpec(),),
        registry=UnusedRegistry(),
        include_current=False,
    )

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


def test_resolver_requires_a_finite_managed_probe_deadline():
    with pytest.raises(ValueError, match="finite deadline"):
        resolve(
            EnvironmentRequirement(tags=("wanted",)),
            candidates=(PythonExecutableSpec("/first/python"),),
            include_current=False,
            probe_timeout=None,
            total_timeout=None,
        )


def test_resolver_metadata_bounds_large_candidate_values():
    result = resolve(None, candidates=(PythonExecutableSpec("x" * 5000),), include_current=False)

    assert len(result.to_data()["selected"]["executable"]) == 4096


def test_resolver_reports_incomplete_after_bounded_duplicate_sequence():
    requirement = EnvironmentRequirement(tags=("selected",))
    rejected = PythonExecutableSpec("/rejected/python")
    selected = PythonExecutableSpec("/selected/python")

    def runner(spec, *, timeout):
        return EnvironmentProbeResult(spec, True, record=replace(inspect_current(), tags=("selected",) if spec == selected else ()))

    result = resolve(
        requirement,
        candidates=(rejected, *((rejected.to_data(),) * 300), selected),
        include_current=False,
        probe_runner=runner,
    )

    assert result.status == "incomplete"
    assert result.selected is None
    assert len(result.attempts) == 32
    assert result.attempt_count == 264
    assert result.probe_count == 1
    assert result.probe_duration_s >= 0
    assert result.to_data()["attempt_count"] == 264
    assert any(issue.code == "resolver_trace_truncated" for issue in result.diagnostics)
    assert any(issue.code == "resolver_candidates_truncated" for issue in result.diagnostics)
    assert any(issue.code == "resolver_input_truncated" for issue in result.diagnostics)


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
        for _ in range(300):
            yield rejected.to_data()
        raise AssertionError("resolver consumed an unbounded duplicate input")

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=candidates(),
        include_current=False,
        max_candidates=1,
        probe_runner=lambda spec, *, timeout: EnvironmentProbeResult(spec, False),
    )

    assert result.status == "incomplete"
    assert any(issue.code == "resolver_candidates_truncated" for issue in result.diagnostics)


def test_resolver_reports_incomplete_after_bounded_registry_aliases():
    registry = EnvironmentRegistry()
    for index in range(300):
        registry.register(f"alias-{index:02}", CurrentEnvironmentSpec())
    registry.register("selected", PythonExecutableSpec("/selected/python"))

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        registry=registry,
        include_current=False,
        max_candidates=8,
        probe_runner=lambda spec, *, timeout: EnvironmentProbeResult(
            spec,
            True,
            record=replace(inspect_current(), tags=("wanted",))
            if isinstance(spec, PythonExecutableSpec) and spec.executable == "/selected/python"
            else None,
        )
        if isinstance(spec, PythonExecutableSpec) and spec.executable == "/selected/python"
        else EnvironmentProbeResult(spec, False),
    )

    assert result.status == "incomplete"
    assert result.selected_name is None
    assert any(issue.code == "resolver_registry_truncated" for issue in result.diagnostics)
    assert any(issue.code == "resolver_input_truncated" for issue in result.diagnostics)


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


def test_resolver_rejects_duck_typed_registry_candidates_before_running_any_probe():
    class InvalidEntry:
        name = "invalid"
        spec = object()

    calls = []
    with pytest.raises(ValueError, match="invalid environment resolver candidate"):
        resolve(
            EnvironmentRequirement(tags=("wanted",)),
            candidates=(InvalidEntry(),),
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


def test_resolver_rejects_malformed_injected_probe_report():
    spec = PythonExecutableSpec("/candidate/python")
    malformed = EnvironmentProbeResult(spec, False, report=object())  # type: ignore[arg-type]

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(spec,),
        include_current=False,
        probe_runner=lambda *_args, **_kwargs: malformed,
    )

    assert result.status == "no_match"
    assert result.attempts[0].status == "probe_failed"


def test_resolver_rejects_non_mapping_injected_probe_report_details():
    from dryml.environments.compatibility import CompatibilityReport

    spec = PythonExecutableSpec("/candidate/python")
    malformed = EnvironmentProbeResult(spec, False, report=CompatibilityReport("compatible", details=object()))  # type: ignore[arg-type]

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(spec,),
        include_current=False,
        probe_runner=lambda *_args, **_kwargs: malformed,
    )

    assert result.status == "no_match"
    assert result.attempts[0].status == "probe_failed"


def test_total_timeout_includes_candidate_normalization():
    ticks = iter((0.0, 0.0, 1.0))

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(PythonExecutableSpec("/candidate/python"),),
        include_current=False,
        total_timeout=0.5,
        clock=lambda: next(ticks),
        probe_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )

    assert result.status == "incomplete"
    assert any(issue.code == "resolver_candidate_input_timeout" for issue in result.diagnostics)
    assert any(issue.code == "resolver_input_truncated" for issue in result.diagnostics)


def test_resolver_records_probe_duration():
    spec = PythonExecutableSpec("/candidate/python")
    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(spec,),
        include_current=False,
        probe_runner=lambda candidate, *, timeout: EnvironmentProbeResult(candidate, False),
    )

    assert result.attempts[0].probe_duration_s is not None
    assert result.attempts[0].to_data()["probe_duration_s"] >= 0


def test_resolver_does_not_double_count_malformed_probe_duration():
    spec = PythonExecutableSpec("/candidate/python")
    state = {"now": 0.0}

    def runner(candidate, *, timeout):
        state["now"] = 2.0
        return EnvironmentProbeResult(candidate, "true")  # type: ignore[arg-type]

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(spec,),
        include_current=False,
        probe_runner=runner,
        clock=lambda: state["now"],
    )

    assert result.probe_count == 1
    assert result.probe_duration_s == 2.0


def test_resolver_marks_a_probe_that_exhausts_total_timeout_as_incomplete():
    spec = PythonExecutableSpec("/candidate/python")
    state = {"now": 0.0}

    def runner(candidate, *, timeout):
        state["now"] = 1.0
        return EnvironmentProbeResult(candidate, True, record=replace(inspect_current(), tags=("wanted",)))

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(spec,),
        include_current=False,
        total_timeout=0.5,
        clock=lambda: state["now"],
        probe_runner=runner,
    )

    attempt = result.attempts[0]
    assert result.status == "incomplete"
    assert attempt.status == "probe_failed"
    assert attempt.probe_duration_s == 1.0
    assert attempt.probe is not None and not attempt.probe.ok
    assert attempt.probe.report is not None
    assert attempt.probe.report.issues[-1].code == "resolver_total_timeout"


@pytest.mark.parametrize("failure", (RuntimeError(), TypeError()))
def test_resolver_marks_deadline_exhausting_probe_failures_as_incomplete(failure):
    spec = PythonExecutableSpec("/candidate/python")
    state = {"now": 0.0}

    def runner(_candidate, *, timeout):
        state["now"] = 1.0
        raise failure

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(spec,),
        include_current=False,
        total_timeout=0.5,
        clock=lambda: state["now"],
        probe_runner=runner,
    )

    assert result.status == "incomplete"
    assert result.attempts[0].status == "probe_failed"
    assert result.attempts[0].probe_duration_s == 1.0


def test_resolver_continues_after_malformed_protocol_record():
    malformed = PythonExecutableSpec("/malformed-protocol/python")
    selected = PythonExecutableSpec("/selected/python")

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(malformed, selected),
        include_current=False,
        probe_runner=lambda spec, *, timeout: EnvironmentProbeResult(
            spec,
            False,
            report=report_from_issues((CompatibilityIssue("probe_failed", "error", "environment probe record could not be decoded"),)),
        )
        if spec == malformed
        else EnvironmentProbeResult(spec, True, record=replace(inspect_current(), tags=("wanted",))),
    )

    assert result.selected == selected
    assert result.attempts[0].status == "probe_failed"
    assert result.attempts[0].probe is not None
    assert result.attempts[0].probe.report.issues[0].code == "probe_failed"


def test_total_timeout_is_checked_after_candidate_conversion():
    ticks = iter((0.0, 0.0, 0.0, 0.0, 1.0))
    candidate = CurrentEnvironmentSpec().to_data()

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(candidate,),
        include_current=False,
        total_timeout=0.5,
        clock=lambda: next(ticks),
        probe_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )

    assert result.status == "incomplete"
    assert any(issue.code == "resolver_candidate_input_timeout" for issue in result.diagnostics)


def test_resolver_does_not_fall_through_after_candidate_input_timeout():
    ticks = iter((0.0, 0.0, 1.0))
    registry = EnvironmentRegistry()
    registry.register("lower-precedence", PythonExecutableSpec("/registry/python"))

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(PythonExecutableSpec("/candidate/python"),),
        registry=registry,
        include_current=True,
        total_timeout=0.5,
        clock=lambda: next(ticks),
        probe_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )

    assert result.status == "incomplete"
    assert all(attempt.source == "candidate" for attempt in result.attempts)


def test_resolver_serialization_redacts_non_json_injected_probe_evidence():
    from dryml.environments.compatibility import CompatibilityReport

    spec = PythonExecutableSpec("/candidate/python")
    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(spec,),
        include_current=False,
        probe_runner=lambda candidate, *, timeout: EnvironmentProbeResult(
            candidate,
            False,
            report=CompatibilityReport("incompatible", details={"opaque": b"not-json"}),
        ),
    )

    data = result.to_data()
    assert json.loads(json.dumps(data))
    assert data["attempts"][0]["probe"]["report"]["details"] == {"redacted": True}


@pytest.mark.parametrize("aliases", [list, tuple])
def test_resolver_bounds_finite_list_and_tuple_aliases(aliases):
    rejected = PythonExecutableSpec("/rejected/python")
    selected = PythonExecutableSpec("/selected/python")

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=aliases([rejected, *((rejected.to_data(),) * 300), selected]),
        include_current=False,
        max_candidates=1,
        probe_runner=lambda spec, *, timeout: EnvironmentProbeResult(spec, False),
    )

    assert result.status == "incomplete"
    assert result.selected is None
    assert any(issue.code == "resolver_candidates_truncated" for issue in result.diagnostics)


def test_resolver_does_not_select_lower_precedence_registry_after_candidate_truncation():
    rejected = PythonExecutableSpec("/rejected/python")
    registry = EnvironmentRegistry()
    registry.register("selected", PythonExecutableSpec("/selected/python"))

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(rejected, *((rejected.to_data(),) * 300)),
        registry=registry,
        include_current=False,
        max_candidates=1,
        probe_runner=lambda spec, *, timeout: EnvironmentProbeResult(
            spec,
            True,
            record=replace(
                inspect_current(),
                tags=("wanted",) if spec.executable == "/selected/python" else (),
            ),
        ),
    )

    assert result.status == "incomplete"
    assert result.selected is None
    assert all(attempt.source != "registry" for attempt in result.attempts)


def test_resolver_does_not_select_current_after_registry_truncation():
    rejected = CurrentEnvironmentSpec()
    registry = EnvironmentRegistry()
    for index in range(300):
        registry.register(f"alias-{index:03}", rejected)

    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        registry=registry,
        include_current=True,
        max_candidates=1,
        probe_runner=lambda spec, *, timeout: EnvironmentProbeResult(
            spec,
            True,
            record=replace(inspect_current(), tags=()),
        ),
    )

    assert result.status == "incomplete"
    assert result.selected is None
    assert all(attempt.source != "current" for attempt in result.attempts)


def test_resolver_serialization_redacts_injected_report_detail_secrets():
    from dryml.environments.compatibility import CompatibilityReport

    secret = "TOKEN=secret"
    spec = PythonExecutableSpec("/candidate/python")
    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(spec,),
        include_current=False,
        probe_runner=lambda candidate, *, timeout: EnvironmentProbeResult(
            candidate,
            False,
            report=CompatibilityReport("incompatible", details={"TOKEN": secret}),
        ),
    )

    assert "TOKEN" not in str(result.to_data())
    assert secret not in str(result.to_data())


def test_resolver_serialization_redacts_injected_issue_secrets():
    from dryml.environments.compatibility import CompatibilityReport

    secret = "TOKEN=secret"
    spec = PythonExecutableSpec("/candidate/python")
    result = resolve(
        EnvironmentRequirement(tags=("wanted",)),
        candidates=(spec,),
        include_current=False,
        probe_runner=lambda candidate, *, timeout: EnvironmentProbeResult(
            candidate,
            False,
            report=CompatibilityReport(
                "incompatible",
                issues=(CompatibilityIssue("runner_secret", "error", secret, expected=secret, observed=secret),),
            ),
        ),
    )

    data = result.to_data()
    issue = data["attempts"][0]["probe"]["report"]["issues"][0]
    assert secret not in str(data)
    assert issue["message"] == "environment compatibility issue: runner_secret"
    assert issue["expected"] == {"redacted": True}
    assert issue["observed"] == {"redacted": True}
