from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import dryml.code as code


def _request(import_path="probe_targets:plain_function", algorithms=("callables",)):
    return code.CodeProbeRequest(
        target=code.CodeTargetSpec.from_import_path(import_path),
        algorithms=algorithms,
        include_environment_record=False,
    )


def _run_worker(payload: str):
    env = dict(os.environ)
    fixture_dir = str(Path(__file__).parents[1] / "fixtures")
    env["PYTHONPATH"] = os.pathsep.join((fixture_dir, env.get("PYTHONPATH", ""))).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", "dryml.code.probe_worker", "--json"],
        input=payload,
        text=True,
        capture_output=True,
        env=env,
    )


def _result(completed) -> code.CodeProbeResult:
    return code.CodeProbeResult.from_data(json.loads(completed.stdout))


def test_worker_accepts_valid_json_and_emits_round_trippable_result():
    completed = _run_worker(json.dumps(_request().to_data()))

    assert completed.returncode == 0
    result = _result(completed)
    assert result.ok
    assert result.analysis is not None
    assert result.analysis.facts_of_kind("callable")
    assert code.CodeProbeResult.from_data(result.to_data()).ok


def test_worker_handles_invalid_json_with_structured_result():
    completed = _run_worker("{not json")

    result = _result(completed)

    assert completed.returncode == 0
    assert not result.ok
    assert result.diagnostics[0].code == "code_probe.invalid_json"


def test_worker_handles_missing_target_field():
    completed = _run_worker(json.dumps({"schema_version": 1}))

    result = _result(completed)

    assert completed.returncode == 0
    assert not result.ok
    assert result.diagnostics[0].code == "code_probe.invalid_request"


def test_worker_rejects_unsupported_schema_version():
    payload = _request().to_data()
    payload["schema_version"] = 999

    completed = _run_worker(json.dumps(payload))
    result = _result(completed)

    assert completed.returncode == 0
    assert not result.ok
    assert result.diagnostics[0].code == "code_probe.invalid_request"


def test_worker_handles_unknown_algorithm():
    completed = _run_worker(json.dumps(_request(algorithms=("missing_algorithm",)).to_data()))

    result = _result(completed)
    assert completed.returncode == 0
    assert not result.ok
    assert {item.code for item in result.diagnostics} == {"code_probe.unknown_algorithm"}


def test_worker_rejects_orchestrator_only_analyzer():
    code.register_analyzer(
        code.FunctionAnalyzer("orchestrator_only", lambda target, context: code.CodeAnalysisResult(target.spec)),
        replace=True,
    )
    completed = _run_worker(json.dumps(_request(algorithms=("orchestrator_only",)).to_data()))
    result = _result(completed)

    assert completed.returncode == 0
    assert not result.ok
    assert {item.code for item in result.diagnostics} == {"code_probe.unknown_algorithm"}


def test_worker_rejects_source_spec_only_targets():
    request = code.CodeProbeRequest(
        target=code.CodeTargetSpec("source_spec", source_spec={"kind": "function", "source": "lambda: None"}),
        include_environment_record=False,
    )
    completed = _run_worker(json.dumps(request.to_data()))
    result = _result(completed)

    assert completed.returncode == 0
    assert not result.ok
    assert result.analysis is None
    assert result.diagnostics[0].code == "code_probe.source_spec_reconstruction_unavailable"


def test_worker_runs_explicit_builtin_static_calls():
    completed = _run_worker(json.dumps(_request(algorithms=("static_calls",)).to_data()))
    result = _result(completed)

    assert completed.returncode == 0
    assert result.ok
    assert result.analysis is not None
    assert result.analysis.facts_of_kind("static_call_summary")


def test_worker_reports_import_failure_diagnostic():
    completed = _run_worker(json.dumps(_request("probe_import_failure:target").to_data()))
    result = _result(completed)
    codes = {item.code for item in result.diagnostics}

    assert completed.returncode == 0
    assert not result.ok
    assert "code_probe.import_error" in codes


def test_worker_captures_user_stdout_and_stderr_without_corrupting_protocol():
    completed = _run_worker(json.dumps(_request("probe_prints_on_import:noisy_target").to_data()))
    result = _result(completed)

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.strip().startswith("{")
    assert "probe fixture stdout" in (result.stdout or "")
    assert "probe fixture stderr" in (result.stderr or "")


def test_worker_usage_errors_exit_2():
    completed = subprocess.run(
        [sys.executable, "-m", "dryml.code.probe_worker"],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
