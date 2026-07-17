import io
import json
import xml.etree.ElementTree as ET

import pytest

from tools import measure_suite


def write_phase_artifacts(output_dir, phase="smoke", *, nodeid="tests/x.py::test_x"):
    (output_dir / f"timing-{phase}.json").write_text(json.dumps({
        "version": 2,
        "phase": phase,
        "session_wall_seconds": 0.1,
        "collection_seconds": 0.01,
        "counts": {
            "collected": 1,
            "selected": 1,
            "executed": 1,
            "deselected": 0,
            "outcomes": {"passed": 1},
        },
        "records": [{"nodeid": nodeid}],
    }))
    (output_dir / f"junit-{phase}.xml").write_text(
        '<testsuites><testsuite tests="1" errors="0" failures="0" skipped="0" time="0.1"/></testsuites>'
    )


def test_fresh_output_dir_rejects_repository_and_existing_paths(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        measure_suite.fresh_output_dir(str(measure_suite.ROOT))
    with pytest.raises(ValueError, match="fresh"):
        measure_suite.fresh_output_dir(str(tmp_path))
    assert measure_suite.fresh_output_dir(str(tmp_path / "new-output")) == (tmp_path / "new-output").resolve()


def test_measure_writes_versioned_non_mutating_artifacts(tmp_path, monkeypatch):
    output_dir = tmp_path / "measurement"

    def fake_run_phase(**kwargs):
        phase = kwargs["phase"]
        write_phase_artifacts(kwargs["output_dir"], phase)
        return {
            "phase": phase,
            "returncode": 0,
            "wall_seconds": 0.1,
            "stdout": {},
            "stderr": {},
            "timing_artifact": f"timing-{phase}.json",
            "junit_artifact": f"junit-{phase}.xml",
        }

    monkeypatch.setattr(measure_suite, "run_phase", fake_run_phase)
    monkeypatch.setattr(measure_suite, "_candidate_identity", lambda: {
        "nested_commit": "nested-sha",
        "parent_commit": None,
        "parent_repository_available": False,
        "nested_tracked_clean": True,
        "parent_tracked_clean": None,
    })

    assert measure_suite.measure(["--output-dir", str(output_dir), "smoke", "-q"]) == 0
    run = json.loads((output_dir / "run.json").read_text())
    nodes = json.loads((output_dir / "nodes.json").read_text())
    assert run["schema"] == 2
    assert run["status"] == "success"
    assert run["coverage"]["enabled"] is False
    assert run["candidate"]["parent_repository_available"] is False
    assert run["candidate"]["parent_commit"] is None
    assert run["counts"]["outcomes"] == {"passed": 1}
    assert nodes["records"] == [{"nodeid": "tests/x.py::test_x"}]
    assert ET.parse(output_dir / "junit.xml").getroot().attrib["tests"] == "1"


def test_measure_fails_before_overwriting_existing_output(tmp_path):
    output_dir = tmp_path / "measurement"
    output_dir.mkdir()

    with pytest.raises(SystemExit) as error:
        measure_suite.measure(["--output-dir", str(output_dir), "smoke"])

    assert error.value.code == 2


def test_bounded_capture_records_original_size_without_retaining_all_output():
    capture = measure_suite._BoundedCapture(io.BytesIO(b"x" * (measure_suite.LOG_LIMIT_BYTES + 1)))

    capture.read()

    assert capture.original_bytes == measure_suite.LOG_LIMIT_BYTES + 1
    assert len(capture.buffer) == measure_suite.LOG_LIMIT_BYTES


def test_status_precedence_is_fail_closed():
    assert measure_suite.classify_status() == "success"
    assert measure_suite.classify_status(invalidated=True) == "invalidated"
    assert measure_suite.classify_status(incomplete=True, invalidated=True) == "incomplete"
    assert measure_suite.classify_status(failure=True, incomplete=True) == "failure"
    assert measure_suite.classify_status(unsupported=True, failure=True) == "unsupported"


def test_measure_reports_missing_artifacts_as_incomplete(tmp_path, monkeypatch):
    output_dir = tmp_path / "measurement"

    def fake_run_phase(**kwargs):
        return {
            "phase": kwargs["phase"],
            "returncode": 0,
            "timing_artifact": "missing.json",
            "junit_artifact": "missing.xml",
        }

    monkeypatch.setattr(measure_suite, "run_phase", fake_run_phase)

    assert measure_suite.measure(["--output-dir", str(output_dir), "smoke"]) == 1
    run = json.loads((output_dir / "run.json").read_text())
    assert run["status"] == "incomplete"
    assert run["artifact_errors"]


def test_measure_preserves_explicit_invalidation(tmp_path, monkeypatch):
    output_dir = tmp_path / "measurement"

    def fake_run_phase(**kwargs):
        phase = kwargs["phase"]
        write_phase_artifacts(kwargs["output_dir"], phase)
        return {
            "phase": phase,
            "returncode": 0,
            "timing_artifact": f"timing-{phase}.json",
            "junit_artifact": f"junit-{phase}.xml",
        }

    monkeypatch.setattr(measure_suite, "run_phase", fake_run_phase)

    assert measure_suite.measure([
        "--output-dir", str(output_dir), "--invalidate-reason", "host pressure", "smoke",
    ]) == 1
    assert json.loads((output_dir / "run.json").read_text())["status"] == "invalidated"


def test_full_no_coverage_preserves_both_behavioral_phases():
    specs = measure_suite._phase_specs("full", False)

    assert [spec[0] for spec in specs] == ["medium", "heavy"]
    assert [spec[3:] for spec in specs] == [(False, False), (False, False)]


def test_coverage_reports_are_deferred_until_the_final_phase():
    arguments = ["-q", "--cov-report=xml", "--cov-report", "html"]
    output_dir = measure_suite.ROOT.parent / "external-measurement"

    assert measure_suite._phase_pytest_args(
        arguments, defer_coverage_reports=True, output_dir=output_dir,
    ) == ["-q", "--cov-report="]
    assert measure_suite._phase_pytest_args(
        arguments, defer_coverage_reports=False, output_dir=output_dir,
    ) == [
        "-q",
        f"--cov-report=xml:{output_dir / 'coverage.xml'}",
        f"--cov-report=html:{output_dir / 'htmlcov'}",
    ]


@pytest.mark.parametrize(
    "argument",
    [
        "tests/untracked",
        "tests/x.py::test_x",
        "--junitxml=tests/test_tiers.json",
        "--dryml-timing-output=/tmp/other.json",
        "--debug=tests/test_tiers.json",
        "--cov=dryml.other",
        "--basetemp=/tmp/pytest",
        "--rootdir=/tmp/other",
        "-m=other_marker",
    ],
)
def test_measurement_rejects_test_paths_and_runner_output_overrides(tmp_path, argument):
    output_dir = tmp_path / "measurement"

    with pytest.raises(SystemExit) as error:
        measure_suite.measure(["--output-dir", str(output_dir), "smoke", argument])

    assert error.value.code == 2
    assert not output_dir.exists()


def test_coverage_report_metadata_omits_caller_destinations():
    arguments = ["--cov-report=xml:/private/result.xml", "--cov-report", "html:/private/html"]

    measure_suite._validate_pytest_args(arguments)

    assert measure_suite._coverage_reports(arguments) == ["xml", "html"]


def test_coverage_core_is_explicit_and_phase_appropriate():
    assert measure_suite._coverage_core(
        coverage=False, append_coverage=False, version_info=(3, 12),
        coverage_version="7.13.5",
    ) is None
    assert measure_suite._coverage_core(
        coverage=True, append_coverage=False, version_info=(3, 11),
        coverage_version="7.13.5",
    ) == "ctrace"
    assert measure_suite._coverage_core(
        coverage=True, append_coverage=False, version_info=(3, 12),
        coverage_version="7.13.5",
    ) == "sysmon"
    assert measure_suite._coverage_core(
        coverage=True, append_coverage=True, version_info=(3, 12),
        coverage_version="7.13.5",
    ) == "ctrace"
    assert measure_suite._coverage_core(
        coverage=True, append_coverage=False, version_info=(3, 12),
        coverage_version="7.3.4",
    ) == "ctrace"


def test_run_phase_sets_and_records_coverage_core(tmp_path, monkeypatch):
    observed = {}

    class FakeProcess:
        stdout = io.BytesIO(b"")
        stderr = io.BytesIO(b"")

        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(measure_suite, "selected_files", lambda _tiers: ["tests/x.py"])
    monkeypatch.setattr(measure_suite, "_coverage_core", lambda **_kwargs: "sysmon")
    monkeypatch.setattr(measure_suite.subprocess, "Popen", fake_popen)

    result = measure_suite.run_phase(
        output_dir=tmp_path,
        phase="medium",
        tiers=["smoke", "medium"],
        markexpr="speed_smoke or speed_medium",
        coverage=True,
        append_coverage=False,
        pytest_args=[],
    )

    assert observed["environment"]["COVERAGE_CORE"] == "sysmon"
    assert observed["environment"]["COVERAGE_FILE"] == str(tmp_path / ".coverage")
    assert observed["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert observed["command"][:6] == [
        measure_suite.sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--cov=dryml",
    ]
    assert result["coverage"] == {
        "enabled": True,
        "append": False,
        "core": "sysmon",
    }


def test_heavy_phase_does_not_publish_inert_bootstrap_contract(tmp_path, monkeypatch):
    observed = {}
    monkeypatch.delenv("DRYML_TEST_BOOTSTRAP_CONTEXTS", raising=False)

    class FakeProcess:
        stdout = io.BytesIO(b"")
        stderr = io.BytesIO(b"")

        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(measure_suite, "selected_files", lambda _tiers: ["tests/x.py"])
    monkeypatch.setattr(measure_suite.subprocess, "Popen", fake_popen)

    result = measure_suite.run_phase(
        output_dir=tmp_path,
        phase="heavy",
        tiers=["heavy"],
        markexpr="speed_heavy",
        coverage=False,
        append_coverage=False,
        pytest_args=[],
    )

    assert "DRYML_TEST_BOOTSTRAP_CONTEXTS" not in observed["environment"]
    assert "--no-cov" in observed["command"]
    assert result["selection"]["selected_files"] == ["tests/x.py"]


def test_memory_is_explicitly_unavailable_without_resource(monkeypatch):
    monkeypatch.setattr(measure_suite, "resource", None)

    assert measure_suite._peak_child_memory() == {
        "bytes": None,
        "available": False,
        "reason": "unsupported_platform",
    }


def test_git_identity_rejects_an_unrelated_enclosing_repository(tmp_path, monkeypatch):
    expected_root = tmp_path / "standalone-source"
    expected_root.mkdir()

    def fake_git_output(_root, *arguments):
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        raise AssertionError("identity lookup must stop after the root mismatch")

    monkeypatch.setattr(measure_suite, "_git_output", fake_git_output)

    assert measure_suite._git_repository_identity(expected_root) is None


def test_git_identity_accepts_only_the_exact_repository_root(tmp_path, monkeypatch):
    expected_root = tmp_path / "repository"
    expected_root.mkdir()

    def fake_git_output(_root, *arguments):
        return {
            ("rev-parse", "--show-toplevel"): str(expected_root),
            ("rev-parse", "HEAD"): "candidate-sha",
            ("status", "--porcelain=v1", "--untracked-files=no"): "",
        }[arguments]

    monkeypatch.setattr(measure_suite, "_git_output", fake_git_output)

    assert measure_suite._git_repository_identity(expected_root) == ("candidate-sha", True)


def test_logs_and_commands_redact_credentials_and_private_paths(tmp_path):
    output = tmp_path / "measure"
    output.mkdir()
    secret = "gho_exampletoken"
    raw = f"TOKEN={secret} Authorization: Bearer abc {measure_suite.ROOT} {output}"

    measure_suite._write_sanitized_log(output / "stdout.log", raw.encode(), output)
    command = measure_suite._sanitize_command(["tool", "--password", "value", raw], output)
    encoded = (output / "stdout.log").read_text() + repr(command)

    assert secret not in encoded
    assert "Bearer abc" not in encoded
    assert " abc" not in encoded
    assert str(measure_suite.ROOT) not in encoded
    assert str(output) not in encoded
    assert "value" not in repr(command)


def test_log_redaction_preserves_windows_newlines_and_size_bound(tmp_path):
    output = tmp_path / "measure"
    output.mkdir()
    raw = b'line\r\n"api_key": "private"\r\n' + b"x" * measure_suite.LOG_LIMIT_BYTES

    measure_suite._write_sanitized_log(output / "stdout.log", raw, output)
    retained = (output / "stdout.log").read_bytes()

    assert b"private" not in retained
    assert b"\r\n" in retained
    assert len(retained) <= measure_suite.LOG_LIMIT_BYTES


def test_ci_workflow_uses_one_measurement_and_bounded_artifacts_per_job():
    workflow = (measure_suite.ROOT / ".github/workflows/tests.yaml").read_text()

    assert workflow.count("actions/upload-artifact@v4") == 2
    assert workflow.count("retention-days: 14") == 2
    assert "measure --output-dir" in workflow
    assert "matrix.python-version == '3.12'" in workflow


def test_full_runner_defers_coverage_reports_until_after_append():
    runner = (measure_suite.ROOT / "tests.sh").read_text()

    medium_command, heavy_command = (
        line for line in runner.splitlines()
        if "pytest --cov=dryml" in line and "_selected[@]" in line
    )
    assert "--cov-report=" in medium_command
    assert "coverage_stripped_args" in medium_command
    assert 'COVERAGE_CORE="$medium_coverage_core"' in medium_command
    assert "--cov-append" in heavy_command
    assert "stripped_args" in heavy_command
    assert "COVERAGE_CORE=ctrace" in heavy_command


def test_runner_does_not_publish_inert_bootstrap_contract():
    runner = (measure_suite.ROOT / "tests.sh").read_text()

    assert "DRYML_TEST_BOOTSTRAP_CONTEXTS" not in runner
