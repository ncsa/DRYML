import ast
import io
import json
import re
import subprocess
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
    pytest_temp_dirs = []

    def fake_run_phase(**kwargs):
        phase = kwargs["phase"]
        pytest_temp_dir = kwargs["pytest_temp_dir"]
        pytest_temp_dirs.append(pytest_temp_dir)
        payload = pytest_temp_dir / "store" / "objects"
        payload.mkdir(parents=True)
        (payload / "def.pkl").write_bytes(b"test payload")
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
    assert pytest_temp_dirs and all(not path.exists() for path in pytest_temp_dirs)
    assert not list(output_dir.rglob("*.pkl"))
    assert not (output_dir / "pytest-temp").exists()


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
        "--config-file=/tmp/pytest.ini",
        "-c=/tmp/pytest.ini",
        "@/tmp/pytest-args.txt",
        "-m=other_marker",
    ],
)
def test_measurement_rejects_test_paths_and_runner_output_overrides(tmp_path, argument):
    output_dir = tmp_path / "measurement"

    with pytest.raises(SystemExit) as error:
        measure_suite.measure(["--output-dir", str(output_dir), "smoke", argument])

    assert error.value.code == 2
    assert not output_dir.exists()


def test_measurement_rejects_config_and_plugin_bypasses(tmp_path):
    arguments = [
        ["--override-ini=log_file=README.md"],
        ["--override-i=addopts=--log-file=README.md"],
        ["--override-ini", "log_file=README.md"],
        ["-p", "unsafe_plugin"],
        ["-p=unsafe_plugin"],
        ["-punsafe_plugin"],
        ["-olog_file=README.md"],
        ["-c/tmp/pytest.ini"],
    ]

    for index, pytest_args in enumerate(arguments):
        output_dir = tmp_path / f"measurement-{index}"
        with pytest.raises(SystemExit) as error:
            measure_suite.measure(["--output-dir", str(output_dir), "smoke", *pytest_args])
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
    monkeypatch.setenv("PYTEST_ADDOPTS", "--log-file=README.md")
    monkeypatch.setenv("PYTEST_DEBUG", "1")
    monkeypatch.setenv("PYTEST_PLUGINS", "unsafe_plugin")
    for variable in (
        "COVERAGE_CORE", "COVERAGE_DEBUG", "COVERAGE_DEBUG_FILE",
        "COVERAGE_FILE", "COVERAGE_PROCESS_START", "COVERAGE_RCFILE",
    ):
        monkeypatch.setenv(variable, "README.md")

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
        pytest_temp_dir=tmp_path.parent / "pytest-temp",
    )

    assert observed["environment"]["COVERAGE_CORE"] == "sysmon"
    assert observed["environment"]["COVERAGE_FILE"] == str(tmp_path / ".coverage")
    assert observed["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert all(
        name not in observed["environment"]
        for name in (
            "PYTEST_ADDOPTS", "PYTEST_DEBUG", "PYTEST_PLUGINS",
            "COVERAGE_DEBUG", "COVERAGE_DEBUG_FILE", "COVERAGE_PROCESS_START",
            "COVERAGE_RCFILE",
        )
    )
    assert observed["command"][:8] == [
        measure_suite.sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "--cov=dryml",
    ]
    assert f"--basetemp={tmp_path.parent / 'pytest-temp'}" in observed["command"]
    assert "--basetemp=<pytest-temp>" in result["command"]
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
        pytest_temp_dir=tmp_path.parent / "pytest-temp",
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

    expected_actions = {
        "actions/checkout": "08c6903cd8c0fde910a37f88322edcfb5dd907a8",
        "actions/setup-python": "e797f83bcb11b83ae66e0230d6156d7c80228e7c",
        "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
        "codecov/codecov-action": "57e3a136b779b570ffcdbf80b3bdc90e7fab3de2",
    }
    for action, revision in expected_actions.items():
        assert f"{action}@{revision}" in workflow
    assert not re.search(r"uses:\s+[^\s]+@v\d", workflow)
    assert workflow.count("actions/upload-artifact@") == 2
    assert workflow.count("retention-days: 14") == 2
    assert "measure --output-dir" in workflow
    assert "matrix.python-version == '3.12'" in workflow
    assert "files: ${{ runner.temp }}/dryml-measure/coverage.xml" in workflow
    assert "disable_search: true" in workflow
    assert "fail_ci_if_error: true" in workflow
    assert workflow.count("if-no-files-found: error") == 2
    assert workflow.count("timeout-minutes:") == 2
    assert workflow.count("Verify Bounded Timing Artifact") == 2
    assert workflow.count("assert not missing") == 2
    assert workflow.count('["status"] == "success"') == 2
    assert workflow.count('run["schema"] == 2') == 2
    assert workflow.count('run["candidate"]["nested_commit"] == expected_sha') == 2
    assert workflow.count('run["candidate"]["nested_tracked_clean"] is True') == 2
    assert workflow.count('run["counts"]["complete"] is True') == 2
    assert workflow.count('[phase["phase"] for phase in run["phases"]] == ["medium", "heavy"]') == 2
    assert workflow.count('run["environment"]["os"] == expected_os') == 2
    assert workflow.count('ci["python"]["version"].startswith(expected_python + ".")') == 2
    assert workflow.count("len(files) <= 20_000") == 2
    assert workflow.count("100 * 1024 * 1024") == 2
    assert "ci-metadata.json" in workflow
    assert "runner.os" in workflow
    assert "runner.arch" in workflow
    assert "ImageOS" in workflow
    assert "ImageVersion" in workflow
    assert workflow.count('"setuptools"') == 4
    assert workflow.count('"wheel"') == 4
    assert workflow.count("--extra-index-url https://download.pytorch.org/whl/cpu") == 2
    assert "--extra-index-url" not in (
        measure_suite.ROOT / "test_requirements.txt"
    ).read_text()


def test_current_runtime_enforcement_limitation_is_the_only_strict_xfail():
    tracked_tests = subprocess.check_output(
        ["git", "ls-files", "--", "tests/*.py"],
        cwd=measure_suite.ROOT,
        text=True,
    ).splitlines()
    xfails = []
    for relative in tracked_tests:
        if relative.startswith(("tests/old/", "tests/dev/")):
            continue
        tree = ast.parse((measure_suite.ROOT / relative).read_text(), filename=relative)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if not (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "xfail"
            ):
                continue
            owner = parents.get(call)
            while owner is not None and not isinstance(
                owner, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                owner = parents.get(owner)
            nodeid = (
                f"{relative}::{owner.name}"
                if owner is not None
                else f"{relative}:line-{call.lineno}"
            )
            keywords = {
                item.arg: ast.literal_eval(item.value)
                for item in call.keywords
            }
            xfails.append((
                nodeid,
                keywords.get("strict"),
                keywords.get("reason"),
            ))

    expected_reason = (
        "currently unsupported: runtime_enforcement OFF does not bypass "
        "dispatch planning and launch requirements"
    )
    pytest_config = (measure_suite.ROOT / "pyproject.toml").read_text()

    assert xfails == [(
        "tests/runtime/test_future_enforcement_xfail.py::test_dispatch_respects_runtime_enforcement_off",
        True,
        expected_reason,
    )]
    assert "current_limitation: current unsupported behavior" in pytest_config


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


def test_measure_entrypoint_clears_inherited_configuration_before_python_startup():
    runner = (measure_suite.ROOT / "tests.sh").read_text()

    assert "unset PYTEST_ADDOPTS PYTEST_DEBUG PYTEST_PLUGINS" in runner
    assert "unset COVERAGE_CORE COVERAGE_DEBUG COVERAGE_DEBUG_FILE COVERAGE_FILE" in runner
    assert "unset COVERAGE_PROCESS_START COVERAGE_RCFILE" in runner
