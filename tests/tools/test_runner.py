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
    tmp_path: Path, *args: str
) -> tuple[subprocess.CompletedProcess, list[list[str]], list[str]]:
    """Run the real shell runner while replacing only its pytest executable."""

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
        "    select)\n"
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
    pytest_stub.write_text(
        "#!/usr/bin/env bash\n"
        "\"$DRYML_REAL_PYTHON\" -c 'import json, os, sys; "
        "open(os.environ[\"DRYML_PYTEST_LOG\"], \"a\", "
        "encoding=\"utf-8\").write(json.dumps(sys.argv[1:]) + "
        "\"\\n\")' \"$@\"\n"
    )
    pytest_stub.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
    env["DRYML_PYTEST_LOG"] = str(log_path)
    env["DRYML_BUCKET_LOG"] = str(bucket_log_path)
    env["DRYML_REAL_PYTHON"] = sys.executable
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


@pytest.mark.parametrize("suite_prefix", [("full",), ()])
def test_full_runner_preserves_options_and_intersects_user_marker_filter(
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


@pytest.mark.parametrize(
    "args",
    [
        ("full", "tests/core/test_repo_save_load.py"),
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
        tmp_path, path, "--no-cov", "-x"
    )

    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert bucket_calls == []
    assert path in calls[0]
    assert "--no-cov" in calls[0]


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
