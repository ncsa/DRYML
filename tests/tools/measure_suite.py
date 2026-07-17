#!/usr/bin/env python3
"""Run bounded, non-mutating DRYML test-suite measurements."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.version import Version

try:
    import resource
except ImportError:  # pragma: no cover - exercised on Windows CI
    resource = None


ROOT = Path(__file__).resolve().parents[2]
PARENT_ROOT = ROOT.parent
LOG_LIMIT_BYTES = 1024 * 1024
VALID_MODES = ("smoke", "medium", "heavy", "full")
DEPENDENCY_PACKAGES = ("pytest", "pytest-cov", "numpy", "dill", "packaging")
_COVERAGE_REPORT_DESTINATIONS = {
    "annotate": "coverage-annotate",
    "html": "htmlcov",
    "json": "coverage.json",
    "lcov": "coverage.lcov",
    "markdown": "coverage.md",
    "markdown-append": "coverage.md",
    "xml": "coverage.xml",
}
_COVERAGE_REPORT_TYPES = {"", "term", "term-missing", *_COVERAGE_REPORT_DESTINATIONS}
_FORBIDDEN_PYTEST_OPTION_PREFIXES = (
    "--basetemp",
    "--cache",
    "--cov",
    "--debug",
    "--dryml-timing-",
    "--html",
    "--json-report",
    "--junit",
    "--log-file",
    "--output",
    "--report-log",
    "--result-log",
    "--rootdir",
)
_PYTEST_OPTIONS_WITH_VALUES = {
    "-k", "--capture", "--color", "--deselect", "--durations",
    "--durations-min", "--ignore", "--ignore-glob", "--log-level",
    "--maxfail", "--show-capture", "--tb", "--verbosity",
}
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?\b(?:token|password|passwd|secret|api[_-]?key|authorization)\b[\"']?)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_GITHUB_TOKEN = re.compile(r"\bgh[oprsu]_[A-Za-z0-9_]+\b")
_SECRET_OPTION = re.compile(
    r"(?i)(--(?:token|password|passwd|secret|api-key|authorization)\s+)([^\s,;]+)"
)


class _BoundedCapture:
    """Read one subprocess stream while retaining no more than one MiB."""

    def __init__(self, stream) -> None:
        self.stream = stream
        self.buffer = bytearray()
        self.original_bytes = 0

    def read(self) -> None:
        while chunk := self.stream.read(65536):
            self.original_bytes += len(chunk)
            remaining = LOG_LIMIT_BYTES - len(self.buffer)
            if remaining > 0:
                self.buffer.extend(chunk[:remaining])


def fresh_output_dir(value: str) -> Path:
    """Validate an external output directory that cannot overwrite existing data."""
    path = Path(value).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("--output-dir must be outside the repository")
    if path.exists():
        raise ValueError("--output-dir must name a fresh, non-existent directory")
    return path


def selected_files(tiers: list[str]) -> list[str]:
    """Return existing bucket selections without updating their manifest."""
    return subprocess.check_output(
        [sys.executable, "tests/tools/test_buckets.py", "select", *tiers],
        cwd=ROOT,
        text=True,
    ).splitlines()


def run_phase(
    *, output_dir: Path, phase: str, tiers: list[str], markexpr: str,
    coverage: bool, append_coverage: bool, pytest_args: list[str],
) -> dict[str, Any]:
    """Run one selected pytest phase and write bounded phase logs."""
    timing_path = output_dir / f"timing-{phase}.json"
    junit_path = output_dir / f"junit-{phase}.xml"
    files = selected_files(tiers)
    command = [
        sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=",
    ]
    if coverage:
        command.append("--cov=dryml")
        if append_coverage:
            command.append("--cov-append")
    else:
        command.append("--no-cov")
    command.extend(["-m", markexpr, *files])
    command.extend([
        "--dryml-timing-output", str(timing_path),
        "--dryml-timing-phase", phase,
        f"--junitxml={junit_path}",
        *pytest_args,
    ])
    started = time.monotonic()
    environment = os.environ.copy()
    for variable in ("PYTEST_ADDOPTS", "PYTEST_DEBUG", "PYTEST_PLUGINS"):
        environment.pop(variable, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    coverage_core = _coverage_core(coverage=coverage, append_coverage=append_coverage)
    if coverage_core is not None:
        environment["COVERAGE_CORE"] = coverage_core
        environment["COVERAGE_FILE"] = str(output_dir / ".coverage")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None and process.stderr is not None
    stdout = _BoundedCapture(process.stdout)
    stderr = _BoundedCapture(process.stderr)
    readers = [threading.Thread(target=capture.read) for capture in (stdout, stderr)]
    for reader in readers:
        reader.start()
    returncode = process.wait()
    for reader in readers:
        reader.join()
    for stream, capture in (("stdout", stdout), ("stderr", stderr)):
        _write_sanitized_log(output_dir / f"{phase}.{stream}.log", capture.buffer, output_dir)
    return {
        "phase": phase,
        "command": _sanitize_command(command, output_dir),
        "returncode": returncode,
        "wall_seconds": time.monotonic() - started,
        "coverage": {
            "enabled": coverage,
            "append": append_coverage,
            "core": coverage_core,
        },
        "selection": {
            "tiers": tiers,
            "marker_expression": markexpr,
            "selected_file_count": len(files),
            "selected_files": files,
        },
        "timing_artifact": timing_path.name,
        "junit_artifact": junit_path.name,
        "stdout": {
            "original_bytes": stdout.original_bytes,
            "retained_bytes": (output_dir / f"{phase}.stdout.log").stat().st_size,
            "truncated": stdout.original_bytes > LOG_LIMIT_BYTES,
        },
        "stderr": {
            "original_bytes": stderr.original_bytes,
            "retained_bytes": (output_dir / f"{phase}.stderr.log").stat().st_size,
            "truncated": stderr.original_bytes > LOG_LIMIT_BYTES,
        },
    }


def measure(argv: list[str] | None = None) -> int:
    """Run one requested measurement mode and emit versioned JSON artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    coverage_group = parser.add_mutually_exclusive_group()
    coverage_group.add_argument("--coverage", action="store_true", dest="coverage", default=None)
    coverage_group.add_argument("--no-coverage", action="store_false", dest="coverage")
    parser.add_argument("--invalidate-reason")
    parser.add_argument("mode", choices=VALID_MODES)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        output_dir = fresh_output_dir(args.output_dir)
        _validate_pytest_args(args.pytest_args)
    except ValueError as error:
        parser.error(str(error))
    output_dir.mkdir(mode=0o700, parents=True)

    coverage_enabled = args.mode == "full" if args.coverage is None else args.coverage
    phase_specs = _phase_specs(args.mode, coverage_enabled)
    phases: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    artifact_errors: list[str] = []
    run_error: str | None = None
    memory_before = _peak_child_memory()
    host_before = _host_snapshot()
    try:
        for phase, tiers, markexpr, coverage, append in phase_specs:
            phase_pytest_args = _phase_pytest_args(
                args.pytest_args,
                defer_coverage_reports=coverage and len(phase_specs) > 1 and not append,
                output_dir=output_dir,
            )
            result = run_phase(
                output_dir=output_dir,
                phase=phase,
                tiers=tiers,
                markexpr=markexpr,
                coverage=coverage,
                append_coverage=append,
                pytest_args=phase_pytest_args,
            )
            phases.append(result)
            if result["returncode"]:
                break
    except Exception as error:  # Preserve diagnostics and partial artifacts.
        run_error = _sanitize_text(f"{type(error).__name__}: {error}", output_dir)[:2000]

    for phase in phases:
        timing_path = output_dir / phase["timing_artifact"]
        try:
            timing = _load_timing(timing_path, phase["phase"])
        except (OSError, ValueError, json.JSONDecodeError) as error:
            artifact_errors.append(f"{phase['phase']} timing artifact: {type(error).__name__}")
        else:
            phase["counts"] = timing["counts"]
            phase["session_wall_seconds"] = timing["session_wall_seconds"]
            phase["collection_seconds"] = timing["collection_seconds"]
            nodes.extend(timing["records"])
        try:
            ET.parse(output_dir / phase["junit_artifact"])
        except (OSError, ET.ParseError) as error:
            artifact_errors.append(f"{phase['phase']} JUnit artifact: {type(error).__name__}")

    if phases and not artifact_errors:
        try:
            _combine_junit(output_dir, [phase["phase"] for phase in phases])
        except (OSError, ValueError, ET.ParseError) as error:
            artifact_errors.append(f"combined JUnit artifact: {type(error).__name__}")

    process_failed = run_error is not None or any(phase["returncode"] != 0 for phase in phases)
    incomplete = len(phases) != len(phase_specs) or bool(artifact_errors)
    status = classify_status(
        failure=process_failed,
        incomplete=incomplete,
        invalidated=bool(args.invalidate_reason),
    )
    aggregate_counts = _aggregate_counts(phases)
    coverage_reports = _coverage_reports(args.pytest_args)
    candidate = _candidate_identity()
    if candidate["nested_commit"] is None:
        artifact_errors.append("candidate identity unavailable")
        if status == "success":
            status = "incomplete"
    run = {
        "schema": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "mode": args.mode,
        "candidate": candidate,
        "environment": {
            "python": sys.version,
            "python_executable": "<python>",
            "implementation": platform.python_implementation(),
            "os": platform.system(),
            "architecture": platform.machine(),
            "dependencies": _dependency_versions(),
        },
        "host": {"before": host_before, "after": _host_snapshot()},
        "coverage": {
            "enabled": coverage_enabled,
            "target": "dryml" if coverage_enabled else None,
            "reports_requested": coverage_reports,
        },
        "counts": aggregate_counts,
        "phases": phases,
        "peak_child_memory": {"before": memory_before, "after": _peak_child_memory()},
        "invalidation_reason": (
            _sanitize_text(args.invalidate_reason, output_dir)[:2000]
            if args.invalidate_reason else None
        ),
        "error": run_error,
        "artifact_errors": artifact_errors,
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    (output_dir / "nodes.json").write_text(json.dumps({
        "schema": 2,
        "counts": aggregate_counts,
        "records": nodes,
    }, indent=2, sort_keys=True) + "\n")
    return 0 if status == "success" else 1


def classify_status(
    *, failure: bool = False, incomplete: bool = False,
    invalidated: bool = False, unsupported: bool = False,
) -> str:
    """Apply the authoritative measurement outcome precedence."""
    if unsupported:
        return "unsupported"
    if failure:
        return "failure"
    if incomplete:
        return "incomplete"
    if invalidated:
        return "invalidated"
    return "success"


def _phase_specs(mode: str, coverage: bool):
    if mode == "full":
        return (
            ("medium", ["smoke", "medium"], "speed_smoke or speed_medium", coverage, False),
            ("heavy", ["heavy"], "speed_heavy", coverage, coverage),
        )
    tiers, markexpr = {
        "smoke": (["smoke"], "speed_smoke"),
        "medium": (["smoke", "medium"], "speed_smoke or speed_medium"),
        "heavy": (["heavy"], "speed_heavy"),
    }[mode]
    return ((mode, tiers, markexpr, coverage, False),)


def _coverage_core(
    *, coverage: bool, append_coverage: bool, version_info=None,
    coverage_version: str | None = None,
) -> str | None:
    """Select the deterministic coverage core for one isolated phase."""
    if not coverage:
        return None
    if version_info is None:
        version_info = sys.version_info
    if coverage_version is None:
        coverage_version = metadata.version("coverage")
    if (
        not append_coverage
        and version_info >= (3, 12)
        and Version(coverage_version) >= Version("7.4")
    ):
        return "sysmon"
    return "ctrace"


def _load_timing(path: Path, phase: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("version") != 2 or payload.get("phase") != phase:
        raise ValueError("incompatible timing artifact")
    if not isinstance(payload.get("records"), list) or not isinstance(payload.get("counts"), dict):
        raise ValueError("incomplete timing artifact")
    return payload


def _aggregate_counts(phases: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"collected": 0, "selected": 0, "executed": 0, "deselected": 0}
    outcomes: Counter[str] = Counter()
    complete = True
    for phase in phases:
        counts = phase.get("counts")
        if not isinstance(counts, dict):
            complete = False
            continue
        for name in totals:
            totals[name] += int(counts.get(name, 0))
        outcomes.update(counts.get("outcomes", {}))
    return {**totals, "outcomes": dict(sorted(outcomes.items())), "complete": complete}


def _combine_junit(output_dir: Path, phases: list[str]) -> None:
    suites = ET.Element("testsuites")
    totals: Counter[str] = Counter()
    total_time = 0.0
    for phase in phases:
        root = ET.parse(output_dir / f"junit-{phase}.xml").getroot()
        phase_suites = list(root) if root.tag == "testsuites" else [root]
        for suite in phase_suites:
            suites.append(suite)
            for name in ("tests", "errors", "failures", "skipped"):
                totals[name] += int(suite.attrib.get(name, 0))
            total_time += float(suite.attrib.get("time", 0.0))
    for name in ("tests", "errors", "failures", "skipped"):
        suites.set(name, str(totals[name]))
    suites.set("time", f"{total_time:.6f}")
    ET.ElementTree(suites).write(output_dir / "junit.xml", encoding="utf-8", xml_declaration=True)


def _write_sanitized_log(path: Path, data: bytes, output_dir: Path) -> None:
    text = data.decode("utf-8", errors="replace")
    sanitized = _sanitize_text(text, output_dir).encode("utf-8")[:LOG_LIMIT_BYTES]
    path.write_bytes(sanitized)


def _sanitize_command(command: list[str], output_dir: Path) -> list[str]:
    sanitized = []
    redact_next = False
    for argument in command:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        text = str(argument)
        if text.lower().rstrip("=") in {
            "--token", "--password", "--passwd", "--secret", "--api-key", "--authorization",
        }:
            sanitized.append(text)
            redact_next = True
            continue
        sanitized.append(_sanitize_text(text, output_dir))
    return sanitized


def _sanitize_text(value: str, output_dir: Path) -> str:
    text = str(value)
    replacements = (
        (str(output_dir), "<output-dir>"),
        (str(ROOT), "<repository>"),
        (str(Path.home()), "<home>"),
        (sys.executable, "<python>"),
    )
    for original, replacement in replacements:
        if original:
            text = text.replace(original, replacement)
    text = _BEARER_TOKEN.sub("Bearer <redacted>", text)
    text = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text)
    text = _SECRET_OPTION.sub(lambda match: f"{match.group(1)}<redacted>", text)
    return _GITHUB_TOKEN.sub("<redacted>", text)


def _peak_child_memory() -> dict[str, Any]:
    if resource is None:
        return {"bytes": None, "available": False, "reason": "unsupported_platform"}
    value = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    byte_value = int(value if sys.platform == "darwin" else value * 1024)
    return {"bytes": byte_value, "available": True, "reason": None}


def _host_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "memory_available_bytes": None,
        "swap_free_bytes": None,
        "io_pressure_full_avg10": None,
    }
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values = {}
        for line in meminfo.read_text().splitlines():
            name, raw = line.split(":", 1)
            fields = raw.split()
            if fields:
                values[name] = int(fields[0]) * 1024
        result["memory_available_bytes"] = values.get("MemAvailable")
        result["swap_free_bytes"] = values.get("SwapFree")
    pressure = Path("/proc/pressure/io")
    if pressure.exists():
        for line in pressure.read_text().splitlines():
            if line.startswith("full "):
                fields = dict(field.split("=", 1) for field in line.split()[1:])
                result["io_pressure_full_avg10"] = float(fields["avg10"])
                break
    return result


def _git_output(root: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_repository_identity(root: Path) -> tuple[str, bool] | None:
    top_level = _git_output(root, "rev-parse", "--show-toplevel")
    resolved_top = os.path.normcase(str(Path(top_level).resolve())) if top_level is not None else None
    if resolved_top != os.path.normcase(str(root.resolve())):
        return None
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=no")
    if commit is None or status is None:
        return None
    return commit, status == ""


def _candidate_identity() -> dict[str, Any]:
    nested = _git_repository_identity(ROOT)
    parent = _git_repository_identity(PARENT_ROOT)
    return {
        "nested_commit": nested[0] if nested is not None else None,
        "parent_commit": parent[0] if parent is not None else None,
        "parent_repository_available": parent is not None,
        "nested_tracked_clean": nested[1] if nested is not None else None,
        "parent_tracked_clean": parent[1] if parent is not None else None,
    }


def _dependency_versions() -> dict[str, str | None]:
    versions = {}
    for package in DEPENDENCY_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _coverage_reports(arguments: list[str]) -> list[str]:
    reports = []
    for index, argument in enumerate(arguments):
        if argument.startswith("--cov-report="):
            reports.append(argument.split("=", 1)[1].split(":", 1)[0])
        elif argument == "--cov-report" and index + 1 < len(arguments):
            reports.append(arguments[index + 1].split(":", 1)[0])
    return reports


def _phase_pytest_args(
    arguments: list[str], *, defer_coverage_reports: bool, output_dir: Path,
) -> list[str]:
    filtered: list[str] = []
    skip_value = False
    for index, argument in enumerate(arguments):
        if skip_value:
            skip_value = False
            continue
        if argument == "--cov-report":
            value = arguments[index + 1]
            skip_value = True
        elif argument.startswith("--cov-report="):
            value = argument.split("=", 1)[1]
        else:
            filtered.append(argument)
            continue
        if not defer_coverage_reports:
            filtered.append(f"--cov-report={_external_coverage_report(value, output_dir)}")
    return [*filtered, "--cov-report="] if defer_coverage_reports else filtered


def _external_coverage_report(value: str, output_dir: Path) -> str:
    report_type = value.split(":", 1)[0]
    if report_type in {"", "term", "term-missing"}:
        return value
    return f"{report_type}:{output_dir / _COVERAGE_REPORT_DESTINATIONS[report_type]}"


def _validate_pytest_args(arguments: list[str]) -> None:
    consume_value = False
    for index, argument in enumerate(arguments):
        if consume_value:
            consume_value = False
            continue
        option = argument.split("=", 1)[0]
        if option == "--cov-report":
            if "=" in argument:
                value = argument.split("=", 1)[1]
            elif index + 1 < len(arguments):
                value = arguments[index + 1]
                consume_value = True
            else:
                raise ValueError("--cov-report requires a value")
            if value.split(":", 1)[0] not in _COVERAGE_REPORT_TYPES:
                raise ValueError("unsupported coverage report type")
            continue
        if option in {
            "-c", "--config-file", "-m", "--pyargs", "--collect-only", "--co", "-o",
        } or option.startswith(
            _FORBIDDEN_PYTEST_OPTION_PREFIXES
        ):
            raise ValueError(f"pytest argument is not safe for measurement: {option}")
        if argument == "--" or argument.startswith("@"):
            raise ValueError("pytest test paths are not supported by measurement")
        if argument.startswith("-"):
            consume_value = "=" not in argument and option in _PYTEST_OPTIONS_WITH_VALUES
            continue
        path_text = argument.split("::", 1)[0]
        candidate = Path(path_text)
        if (
            candidate.is_absolute()
            or "/" in argument
            or "\\" in argument
            or "::" in argument
            or path_text.endswith(".py")
            or (ROOT / candidate).exists()
        ):
            raise ValueError("pytest test paths are not supported by measurement")


if __name__ == "__main__":
    raise SystemExit(measure())
