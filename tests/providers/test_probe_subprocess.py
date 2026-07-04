import json
import os
import stat
import sys

import dryml.environments as envs
import dryml.operations as ops
import dryml.providers as providers
from dryml.providers.probe_worker import handle_json_request


def operation_spec():
    return ops.make_function_call_spec("providers.fake_provider:target_fn")


def registry(ref=None):
    reg = providers.ProviderRegistry()
    reg.register_ref(ref or providers.ProviderRef("fake", "providers.fake_provider"))
    return reg


def test_worker_protocol_serialization():
    request = providers.OperationInspectionRequest(operation_spec=operation_spec())
    payload = {
        "schema": "dryml.provider_probe_request.v1",
        "schema_version": 1,
        "request": request.to_data(),
        "providers": [providers.ProviderRef("fake", "providers.fake_provider").to_data()],
        "probe_policy": request.probe_policy.to_data(),
    }

    response = handle_json_request(json.dumps(payload))
    assert response["ok"] is True
    report = providers.ProbeReport.from_data(response["probe_report"])
    assert report.reports[0].status == "ok"


def test_current_environment_and_python_executable_subprocess_probe(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.path.abspath("tests"))
    current = providers.probe_operation(operation_spec(), environment=envs.CurrentEnvironmentSpec(), providers=("fake",), registry=registry(), timeout=30)
    python = providers.probe_operation(operation_spec(), environment=envs.PythonExecutableSpec(sys.executable, pythonpath_policy="inherit"), providers=("fake",), registry=registry(), timeout=30)

    assert current.status == "ok"
    assert python.status == "ok"
    metadata = current.reports[0].metadata
    assert metadata["runtime_mode"] == "probe"
    assert metadata["allocation"] == "NoAllocation"
    assert metadata["cuda_visible_devices"] == ""
    assert metadata["hip_visible_devices"] == ""
    assert metadata["xla_visible_devices"] == ""


def test_provider_failure_and_import_error_are_structured(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.path.abspath("tests"))
    failing = providers.probe_operation(operation_spec(), providers=("fake",), registry=registry(), provider_options={"fake": {"fail": True}}, timeout=30)
    missing_ref = providers.ProviderRef("missing", "providers.missing_provider")
    import_error = providers.probe_operation(operation_spec(), providers=(missing_ref,), registry=registry(missing_ref), timeout=30)

    assert failing.status == "failed"
    assert failing.reports[0].issues[0].code == "provider_failed"
    assert import_error.status == "failed"
    assert import_error.reports[0].issues[0].code == "provider_failed"


def test_timeout_nonzero_and_malformed_worker_output(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.path.abspath("tests"))
    timeout = providers.probe_operation(operation_spec(), providers=("fake",), registry=registry(), provider_options={"fake": {"sleep": 2}}, timeout=0.05)
    assert timeout.diagnostics[0].code == "probe_timeout"

    bad = tmp_path / "bad-python"
    bad.write_text("#!/bin/sh\nprintf '%s\n' 'not json'\n")
    bad.chmod(bad.stat().st_mode | stat.S_IXUSR)
    malformed = providers.probe_operation(operation_spec(), environment=envs.PythonExecutableSpec(str(bad)), providers=("fake",), registry=registry(), timeout=30)
    assert malformed.diagnostics[0].code in {"malformed_worker_output", "probe_failed"}

    nonzero = tmp_path / "nonzero-python"
    nonzero.write_text("#!/bin/sh\nprintf '%s\n' 'oops'\nexit 7\n")
    nonzero.chmod(nonzero.stat().st_mode | stat.S_IXUSR)
    failed = providers.probe_operation(operation_spec(), environment=envs.PythonExecutableSpec(str(nonzero)), providers=("fake",), registry=registry(), timeout=30)
    assert failed.status == "failed"
