import json
import logging
import os

import pytest

import dryml
import dryml.annotations as ann
import dryml.environments as envs
import dryml.operations as ops
import dryml.dispatch as dispatch
import dryml.providers as providers
import dryml.reporting as reporting
from dryml.core2.store.dir import DirStore
from dryml.records import ExecutionRecord, RecordStoreIO, make_record, plan_record_closure


def teardown_function():
    dryml.reset_config()


def operation_spec():
    return ops.make_function_call_spec("providers.fake_provider:target_fn")


def provider_registry():
    registry = providers.ProviderRegistry()
    registry.register_ref(providers.ProviderRef("fake", "providers.fake_provider"))
    return registry


class FailingReporter(reporting.Reporter):
    def emit(self, event, config):
        raise RuntimeError("reporter failed")


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


def test_direct_reporter_config_defaults_to_debug_when_base_is_quiet(capsys):
    capture = reporting.CaptureReporter()

    dryml.configure(reporting=capture)
    reporting.debug("dryml.test.debug", "Captured through direct reporter")

    assert dryml.status()["reporting"]["level"] == "debug"
    assert [event.name for event in capture.events] == ["dryml.test.debug"]
    assert capsys.readouterr().out == ""


def test_boolean_reporting_config_parsing():
    dryml.configure(reporting={"level": "details", "include_ids": "false", "include_timing": "0", "strict": "yes"})

    status = dryml.status()["reporting"]
    assert status["include_ids"] is False
    assert status["include_timing"] is False
    assert status["strict"] is True

    with pytest.raises(ValueError, match="include_ids"):
        dryml.configure(reporting={"include_ids": "sometimes"})


def test_logging_reporter_does_not_accumulate_null_handlers():
    logger = logging.getLogger("dryml.reporting")
    before = sum(isinstance(handler, logging.NullHandler) for handler in logger.handlers)
    dryml.configure(reporting={"level": "steps", "stream": "logging"})

    for index in range(3):
        reporting.step("dryml.test.logging", f"Logging event {index}")

    after = sum(isinstance(handler, logging.NullHandler) for handler in logger.handlers)
    assert after <= max(before, 1)


def test_reporting_emit_is_fail_soft_by_default():
    dryml.configure(reporting="details")

    assert reporting.detail("dryml.test.bad", "Bad payload", data={"object": object()}) is None

    dryml.configure(reporting={"level": "steps", "reporter": FailingReporter()})
    assert reporting.step("dryml.test.fail", "Reporter failure") is None


def test_reporting_strict_mode_raises_event_and_reporter_errors():
    dryml.configure(reporting={"level": "details", "strict": True})
    with pytest.raises(ValueError, match="JSON-ready"):
        reporting.detail("dryml.test.bad", "Bad payload", data={"object": object()})

    dryml.configure(reporting={"level": "steps", "strict": True, "reporter": FailingReporter()})
    with pytest.raises(RuntimeError, match="reporter failed"):
        reporting.step("dryml.test.fail", "Reporter failure")


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


def test_dispatch_and_execution_reporting_events(tmp_path):
    capture = reporting.CaptureReporter()
    dryml.configure(reporting={"level": "details", "reporter": capture})
    operation = ops.attach_operation_id(operation_spec())
    dispatch_spec = dispatch.attach_dispatch_id(dispatch.make_dispatch_spec(operation_id=operation["id"]))
    recipe = dispatch.attach_recipe_id(dispatch.make_execution_recipe(dispatch_id=dispatch_spec["id"], operation_id=operation["id"], backend={"name": "dryml.fake"}))
    store = DirStore(tmp_path / "store")
    io = RecordStoreIO(store)
    io.write_spec(operation, family="operation")
    io.write_spec(dispatch_spec, family="dispatch")
    io.write_spec(recipe, family="execution_recipe")
    seed = io.write_record(make_record(kind="stored_state", payload={"subject_cdef_id": "cdef-v4-" + "a" * 64}))
    execution = ExecutionRecord(
        execution_kind="python",
        operation_id=operation["id"],
        dispatch_id=dispatch_spec["id"],
        recipe_id=recipe["id"],
        backend={"name": "dryml.fake"},
        status="ok",
        consumed_records=(seed.record_id,),
    )
    io.write_record(execution.to_envelope())
    io.find_execution_records(operation_id=operation["id"])
    plan_record_closure(store, seed_records=[seed.record_id], policy="provenance")

    names = [event.name for event in capture.events]
    assert "dryml.dispatch.spec.build" in names
    assert "dryml.dispatch.recipe.build" in names
    assert "dryml.records.execution.write" in names
    assert "dryml.records.execution.query" in names
    assert "dryml.records.execution.export" in names
