import json
import os

import dryml
import dryml.annotations as ann
import dryml.environments as envs
import dryml.operations as ops
import dryml.providers as providers
import dryml.reporting as reporting


def teardown_function():
    dryml.reset_config()


def operation_spec():
    return ops.make_function_call_spec("providers.fake_provider:target_fn")


def provider_registry():
    registry = providers.ProviderRegistry()
    registry.register_ref(providers.ProviderRef("fake", "providers.fake_provider"))
    return registry


def test_env_reporting_defaults(monkeypatch):
    monkeypatch.setenv("DRYML_REPORT", "details")
    monkeypatch.setenv("DRYML_REPORT_STREAM", "stderr")
    monkeypatch.setenv("DRYML_REPORT_FORMAT", "json")

    cfg = dryml.reset_config()

    assert cfg.reporting.level == "details"
    assert dryml.status()["reporting"]["stream"] == "stderr"
    assert dryml.status()["reporting"]["format"] == "json"


def test_quiet_mode_emits_nothing(capsys):
    dryml.configure(reporting="quiet")

    reporting.step("dryml.test.step", "Should not render")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_steps_mode_renders_step_not_detail(capsys):
    dryml.configure(reporting="steps")

    reporting.step("dryml.test.step", "Doing work")
    reporting.detail("dryml.test.detail", "Hidden detail", data={"x": 1})

    captured = capsys.readouterr()
    assert "DRYML: Doing work" in captured.out
    assert "Hidden detail" not in captured.out


def test_details_mode_renders_compact_data(capsys):
    dryml.configure(reporting={"level": "details", "stream": "stdout", "include_timing": False})

    reporting.detail("dryml.test.detail", "Merged requirements", operation_id="op-v1-test", data={"fragments": 3})

    captured = capsys.readouterr()
    assert "DRYML: Merged requirements" in captured.out
    assert "fragments: 3" in captured.out
    assert "operation_id: op-v1-test" in captured.out


def test_json_format_outputs_event_data(capsys):
    dryml.configure(reporting={"level": "debug", "format": "json", "include_timing": False})

    reporting.debug("dryml.test.debug", "Debug payload", data={"cache_key": "abc"})

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "dryml.test.debug"
    assert payload["data"]["cache_key"] == "abc"


def test_capture_reporter_collects_events_without_stdout(capsys):
    capture = reporting.CaptureReporter()
    dryml.configure(reporting={"level": "debug", "reporter": capture})

    reporting.step("dryml.test.step", "Captured")
    reporting.debug("dryml.test.debug", "Captured debug")

    assert [event.name for event in capture.events] == ["dryml.test.step", "dryml.test.debug"]
    assert capsys.readouterr().out == ""


def test_config_context_temporarily_overrides_reporting(capsys):
    dryml.configure(reporting="quiet")

    with dryml.config(reporting="steps"):
        reporting.step("dryml.test.step", "Inside context")
    reporting.step("dryml.test.step", "Outside context")

    captured = capsys.readouterr()
    assert "Inside context" in captured.out
    assert "Outside context" not in captured.out


def test_annotations_resolve_emits_boundary_events():
    capture = reporting.CaptureReporter()
    dryml.configure(reporting={"level": "details", "reporter": capture})

    def target():
        return None

    ann.resolve(target)

    assert [event.name for event in capture.events] == ["dryml.annotations.resolve", "dryml.annotations.resolve.result"]


def test_provider_run_probe_emits_boundary_events(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.path.abspath("tests"))
    capture = reporting.CaptureReporter()
    dryml.configure(reporting={"level": "details", "reporter": capture})

    report = providers.probe_operation(operation_spec(), environment=envs.CurrentEnvironmentSpec(), providers=("fake",), registry=provider_registry(), timeout=30)

    assert report.status == "ok"
    names = [event.name for event in capture.events]
    assert "dryml.providers.probe.start" in names
    assert "dryml.providers.probe.refs" in names
    assert "dryml.providers.probe.complete" in names
    assert "dryml.providers.probe.result" in names


def test_worker_protocol_stdout_is_json_only_when_reporting_enabled(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.path.abspath("tests"))
    monkeypatch.setenv("DRYML_REPORT", "steps")
    dryml.reset_config()

    report = providers.probe_operation(operation_spec(), environment=envs.CurrentEnvironmentSpec(), providers=("fake",), registry=provider_registry(), provider_options={"fake": {"noisy": True}}, timeout=30)

    assert report.status == "ok"
    assert "hello from provider" in report.reports[0].stdout
    assert all("DRYML:" not in text for text in (report.reports[0].stdout or "", report.reports[0].stderr or ""))
