"""Focused regression tests for the DRYML test runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.tools import test_buckets


ROOT = Path(__file__).resolve().parents[2]


def _run_tests_sh(
    tmp_path: Path, *args: str, pytest_statuses: tuple[int, ...] = ()
) -> tuple[subprocess.CompletedProcess, list[list[str]], list[str]]:
    """Run the shell with stubbed pytest/selection and real argument parsing."""

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("tests.sh integration requires Bash")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "pytest-calls.jsonl"
    bucket_log_path = tmp_path / "bucket-calls.txt"
    python_stub = bin_dir / "python"
    python_stub.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"./tests/tools/test_buckets.py\" ]]; then\n"
        "  case \"$2\" in\n"
        "    runner-args) exec \"$DRYML_REAL_PYTHON\" \"$@\" ;;\n"
        "    select|select-profile)\n"
        "      \"$DRYML_REAL_PYTHON\" -c 'import os, sys; "
        "open(os.environ[\"DRYML_BUCKET_LOG\"], \"a\").write("
        "sys.argv[1] + \"\\n\")' \"$2\"\n"
        "      if [[ \" $* \" == *\" heavy \"* ]]; then\n"
        "        printf '%s\\n' ./tests/tf/test_one.py\n"
        "      else\n"
        "        printf '%s\\n' ./tests/session/test_one.py "
        "./tests/core/test_one.py\n"
        "      fi\n"
        "      exit 0 ;;\n"
        "    select-category)\n"
        "      printf '%s\\n' ./tests/package/test_one.py\n"
        "      exit 0 ;;\n"
        "    update|summary)\n"
        "      \"$DRYML_REAL_PYTHON\" -c 'import os, sys; "
        "open(os.environ[\"DRYML_BUCKET_LOG\"], \"a\").write("
        "sys.argv[1] + \"\\n\")' \"$2\"\n"
        "      exit 0 ;;\n"
        "  esac\n"
        "fi\n"
        "exec \"$DRYML_REAL_PYTHON\" \"$@\"\n"
    )
    python_stub.chmod(0o755)
    pytest_stub = bin_dir / "pytest"
    # Record shell arguments before Git Bash translates paths for native Python.
    pytest_stub.write_text(
        "#!/usr/bin/env bash\n"
        "MSYS2_ARG_CONV_EXCL='*' \"$DRYML_REAL_PYTHON\" -c 'import json, os, sys; "
        "path=os.environ[\"DRYML_PYTEST_LOG\"]; "
        "lines=open(path, encoding=\"utf-8\").readlines() if "
        "os.path.exists(path) else []; "
        "open(path, \"a\", encoding=\"utf-8\").write("
        "json.dumps(sys.argv[1:]) + \"\\n\"); "
        "statuses=[int(value) for value in "
        "os.environ.get(\"DRYML_PYTEST_STATUSES\", \"\").split(\",\") "
        "if value]; sys.exit(statuses[len(lines)] if len(lines) < "
        "len(statuses) else 0)' \"$@\"\n"
    )
    pytest_stub.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
    env["DRYML_PYTEST_LOG"] = str(log_path)
    env["DRYML_BUCKET_LOG"] = str(bucket_log_path)
    env["DRYML_REAL_PYTHON"] = sys.executable
    env["DRYML_PYTEST_STATUSES"] = ",".join(
        str(status) for status in pytest_statuses
    )
    result = subprocess.run(
        [bash, "./tests.sh", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = []
    if log_path.exists():
        calls = [
            json.loads(line) for line in log_path.read_text().splitlines()
        ]
    bucket_calls = []
    if bucket_log_path.exists():
        bucket_calls = bucket_log_path.read_text().splitlines()
    return result, calls, bucket_calls


def test_nested_runner_accepts_an_inherited_native_pythonpath(tmp_path, monkeypatch):
    """Nested shells preserve native path lists instead of mixing separators."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(ROOT), str(tmp_path / "with spaces"))))
    result, calls, _ = _run_tests_sh(tmp_path, "good-enough")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2


def test_runner_parser_preserves_path_options_and_removes_only_suite_root():
    args = [
        "--ignore",
        "tests/old",
        "--basetemp",
        "tests",
        "--junitxml",
        "tests/results.xml",
        "-o",
        "cache_dir=tests",
        "-k",
        "example",
        "-m",
        "category_core",
        "tests",
    ]

    root_index = test_buckets.runner_suite_root_index(args)

    assert root_index == len(args) - 1
    assert args[:root_index] + args[root_index + 1:] == args[:-1]


def test_runner_parser_recognizes_dryml_options_without_suite_root():
    assert test_buckets.runner_suite_root_index([
        "--dryml-tier-baseline", "tests/test_tiers.json",
        "--dryml-profile-policy", "tests/test_profiles.json",
        "--dryml-timing-summary",
    ]) is None


def test_runner_parser_honors_environment_and_pyproject_addopts(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        'addopts = "-o cache_dir=tests --ignore tests"\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_ADDOPTS", "--basetemp tests")
    args = ["--junitxml", "tests/results.xml", "tests"]

    root_index = test_buckets.runner_suite_root_index(args)

    assert root_index == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    "args",
    [
        ["tests/core/test_repo_save_load.py"],
        ["-x", "tests/core/test_repo_save_load.py::test_missing"],
    ],
)
def test_runner_parser_rejects_explicit_selections(args):
    with pytest.raises(ValueError, match="put focused paths first"):
        test_buckets.runner_suite_root_index(args)


@pytest.mark.parametrize("suite_prefix", [("full",), ("exhaustive",)])
def test_exhaustive_runner_preserves_options_and_intersects_user_filters(
    tmp_path, suite_prefix
):
    result, calls, bucket_calls = _run_tests_sh(
        tmp_path,
        *suite_prefix,
        "--ignore",
        "tests/old",
        "--ignore",
        "tests/dev",
        "--basetemp",
        "tests",
        "--junitxml",
        "tests/results.xml",
        "-o",
        "cache_dir=tests",
        "-k",
        "example",
        "-m",
        "category_core",
        "-x",
        "tests",
    )

    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    assert bucket_calls == ["select", "select"]
    for call in calls:
        assert ["--ignore", "tests/old"] == call[
            call.index("--ignore"):call.index("--ignore") + 2
        ]
        assert "--basetemp" in call
        assert call[call.index("--basetemp") + 1] == "tests"
        assert call.count("tests") == 1
        assert call[call.index("-m") + 1] == "category_core"
        assert "-x" in call
        assert "--dryml-runner-tiers" in call
        assert "--dryml-test-profile" not in call
        assert "--no-cov" in call
        assert "--cov=dryml" not in call
    phase_paths = [
        {arg for arg in call if arg.startswith("./tests/")} for call in calls
    ]
    assert phase_paths == [
        {"./tests/session/test_one.py"},
        {"./tests/core/test_one.py"},
        {"./tests/tf/test_one.py"},
    ]
    assert set().union(*phase_paths) == {
        "./tests/session/test_one.py",
        "./tests/core/test_one.py",
        "./tests/tf/test_one.py",
    }
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(phase_paths)
        for right in phase_paths[index + 1:]
    )


@pytest.mark.parametrize("suite_prefix", [(), ("good-enough",), ("-x",)])
def test_good_enough_is_default_and_excludes_heavy_and_coverage(
    tmp_path, suite_prefix
):
    result, calls, bucket_calls = _run_tests_sh(tmp_path, *suite_prefix)

    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    assert bucket_calls == ["select-profile"]
    assert all("--no-cov" in call for call in calls)
    assert all("--cov=dryml" not in call for call in calls)
    assert all("--dryml-test-profile" in call for call in calls)
    assert all("./tests/tf/test_one.py" not in call for call in calls)


def test_coverage_is_the_only_mode_that_implicitly_combines_coverage(tmp_path):
    result, calls, bucket_calls = _run_tests_sh(tmp_path, "coverage")

    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    assert bucket_calls == ["select", "select"]
    assert all("--cov=dryml" in call for call in calls)
    assert "--cov-append" not in calls[0]
    assert all("--cov-append" in call for call in calls[1:])
    assert all("--no-cov" not in call for call in calls)


@pytest.mark.parametrize(
    "args",
    [
        ("full", "tests/core/test_repo_save_load.py"),
        ("good-enough", "tests/core/test_repo_save_load.py"),
        ("coverage", "tests/core/test_repo_save_load.py"),
        ("medium", "tests/core/test_repo_save_load.py::test_missing"),
        ("-x", "tests/core/test_repo_save_load.py"),
        ("profile", "tests/core/test_repo_save_load.py"),
    ],
)
def test_named_and_option_first_suites_reject_selection_before_pytest(
    tmp_path, args
):
    result, calls, bucket_calls = _run_tests_sh(tmp_path, *args)

    assert result.returncode == 2
    assert calls == []
    assert bucket_calls == []
    assert "put focused paths first" in result.stderr.lower()


def test_focused_path_first_still_runs_directly(tmp_path):
    path = "tests/core/test_repo_save_load.py"

    result, calls, bucket_calls = _run_tests_sh(
        tmp_path, path, "-x"
    )

    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert bucket_calls == []
    assert path in calls[0]
    assert "--no-cov" in calls[0]


def test_focused_path_accepts_explicit_coverage_without_no_cov(tmp_path):
    path = "tests/core/test_repo_save_load.py"

    result, calls, bucket_calls = _run_tests_sh(
        tmp_path, path, "--cov=dryml", "-x"
    )

    assert result.returncode == 0, result.stderr
    assert bucket_calls == []
    assert calls == [[path, "--cov=dryml", "-x"]]


def test_package_mode_owns_package_files_without_implicit_coverage(tmp_path):
    result, calls, bucket_calls = _run_tests_sh(tmp_path, "package")

    assert result.returncode == 0, result.stderr
    assert bucket_calls == []
    assert len(calls) == 1
    assert "./tests/package/test_one.py" in calls[0]
    assert "--no-cov" in calls[0]


def test_profile_remains_exhaustive_no_cov_and_writes_timings_under_tmp(tmp_path):
    result, calls, bucket_calls = _run_tests_sh(tmp_path, "profile")

    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    assert bucket_calls == ["select", "select", "update", "summary"]
    assert all("--no-cov" in call for call in calls)
    assert all("--dryml-test-profile" not in call for call in calls)
    outputs = [call[call.index("--dryml-timing-output") + 1] for call in calls]
    assert all(path.startswith("/tmp/dryml/profile/") for path in outputs)
    assert all(not path.startswith("./tests/") for path in outputs)


def test_empty_early_phase_does_not_hide_later_matching_tests(tmp_path):
    result, calls, _ = _run_tests_sh(
        tmp_path, pytest_statuses=(5, 0)
    )

    assert result.returncode == 0, result.stderr
    assert len(calls) == 2


def test_all_empty_phases_return_pytest_no_tests(tmp_path):
    result, calls, _ = _run_tests_sh(
        tmp_path, pytest_statuses=(5, 5)
    )

    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED
    assert len(calls) == 2
    assert "no tests matched" in result.stderr.lower()


def test_phase_failure_stops_without_double_execution(tmp_path):
    result, calls, _ = _run_tests_sh(tmp_path, pytest_statuses=(1,))

    assert result.returncode == 1
    assert len(calls) == 1


def test_real_pytest_intersects_user_filters_with_runner_tiers(tmp_path):
    test_path = tmp_path / "test_mixed.py"
    test_path.write_text(
        "import pytest\n"
        "def test_smoke_match(): pass\n"
        "def test_smoke_other(): pass\n"
        "@pytest.mark.speed_heavy\n"
        "def test_explicit_heavy_match(): pass\n"
        "def test_baseline_heavy_match(): pass\n"
    )
    baseline_path = tmp_path / "tiers.json"
    baseline_path.write_text(
        json.dumps(
            {
                "default_tier": "medium",
                "path_tiers": {"test_mixed.py": "smoke"},
                "node_tiers": {
                    "test_mixed.py::test_baseline_heavy_match": "heavy"
                },
            }
        )
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("PYTEST_ADDOPTS", None)
    cases = [
        ("smoke", {"test_mixed.py::test_smoke_match"}),
        (
            "heavy",
            {
                "test_mixed.py::test_explicit_heavy_match",
                "test_mixed.py::test_baseline_heavy_match",
            },
        ),
    ]

    for tier, expected in cases:
        timing_path = tmp_path / f"{tier}.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "tests.timing_plugin",
                "--no-cov",
                "--dryml-tier-baseline",
                str(baseline_path),
                "--dryml-runner-tiers",
                tier,
                "--dryml-timing-output",
                str(timing_path),
                "-m",
                "speed_smoke or speed_heavy",
                "-k",
                "match",
                test_path.name,
            ],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        records = json.loads(timing_path.read_text())["records"]
        assert {record["nodeid"] for record in records} == expected


def test_real_profile_is_stable_and_filters_cannot_resample_matrix(tmp_path):
    (tmp_path / "test_matrix.py").write_text(
        "import pytest\n"
        "@pytest.fixture(params=['fa', 'fb'])\n"
        "def shared(request): return request.param\n"
        "@pytest.mark.parametrize(('number', 'label'), "
        "[(0, 'x0'), (1, 'x1'), (2, 'x2')])\n"
        "@pytest.mark.parametrize('flavor', ['red', 'blue', 'green'])\n"
        "def test_tuple_matrix(shared, number, label, flavor): pass\n"
        "@pytest.mark.parametrize('left', ['A0', 'A1', 'A2'])\n"
        "@pytest.mark.parametrize('right', ["
        "'B0', pytest.param('B1', marks=pytest.mark.focus), 'B2'])\n"
        "def test_filter_matrix(left, right): pass\n"
        "@pytest.mark.parametrize('invalid', [0, 1, 2, 3, 4])\n"
        "def test_single_axis(invalid): pass\n"
        "@pytest.mark.exhaustive_only\n"
        "@pytest.mark.parametrize('case', [0, 1])\n"
        "def test_extra_matrix(case): pass\n"
    )
    (tmp_path / "test_safety.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('left', [0, 1, 2])\n"
        "@pytest.mark.parametrize('right', [0, 1, 2])\n"
        "def test_race(left, right): pass\n"
    )
    policy_path = tmp_path / "profiles.json"
    policy_path.write_text(json.dumps({
        "version": 1,
        "profiles": {
            "toy": {
                "sample_multi_axis_at": 4,
                "sampled_functions": {
                    "test_matrix.py::test_tuple_matrix": "genuine product",
                    "test_matrix.py::test_filter_matrix": "genuine product",
                },
                "exhaustive_paths": {
                    "test_safety.py": "safety proof",
                },
            },
        },
    }))
    baseline_path = tmp_path / "tiers.json"
    baseline_path.write_text(json.dumps({"default_tier": "medium"}))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("PYTEST_ADDOPTS", None)

    def run(output: Path, *extra: str, representative=True) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "tests.timing_plugin",
                "--no-cov",
                "--dryml-tier-baseline",
                str(baseline_path),
                *(["--dryml-test-profile", "toy"] if representative else []),
                "--dryml-profile-policy",
                str(policy_path),
                "--dryml-timing-output",
                str(output),
                *extra,
                "test_matrix.py",
                "test_safety.py",
            ],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = run(first_path)
    second = run(second_path)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_ids = [
        record["nodeid"]
        for record in json.loads(first_path.read_text())["records"]
    ]
    second_ids = [
        record["nodeid"]
        for record in json.loads(second_path.read_text())["records"]
    ]
    assert first_ids == second_ids
    assert not any("test_extra_matrix" in nodeid for nodeid in first_ids)
    complete_path = tmp_path / "complete.json"
    complete = run(complete_path, representative=False)
    assert complete.returncode == 0, complete.stderr
    assert {
        record["nodeid"]
        for record in json.loads(complete_path.read_text())["records"]
        if "test_extra_matrix" in record["nodeid"]
    } == {"test_matrix.py::test_extra_matrix[0]", "test_matrix.py::test_extra_matrix[1]"}
    assert sum(nodeid.startswith("test_safety.py::") for nodeid in first_ids) == 9
    assert sum("test_single_axis" in nodeid for nodeid in first_ids) == 5
    tuple_ids = [nodeid for nodeid in first_ids if "test_tuple_matrix" in nodeid]
    assert len(tuple_ids) < 18
    for value in ("fa", "fb", "x0", "x1", "x2", "red", "blue", "green"):
        assert any(value in nodeid for nodeid in tuple_ids)

    filtered = run(tmp_path / "filtered.json", "-k", "A1 and B1", "-m", "focus")
    assert filtered.returncode == pytest.ExitCode.NO_TESTS_COLLECTED
