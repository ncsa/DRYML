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

try:
    import resource
except ImportError:  # pragma: no cover - exercised on Windows CI
    resource = None


ROOT = Path(__file__).resolve().parents[2]
PARENT_ROOT = ROOT.parent
LOG_LIMIT_BYTES = 1024 * 1024
VALID_MODES = ("smoke", "medium", "heavy", "full")
DEPENDENCY_PACKAGES = ("pytest", "pytest-cov", "numpy", "dill", "packaging")
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
    command = [sys.executable, "-m", "pytest"]
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
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
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
        "coverage": {"enabled": coverage, "append": append_coverage},
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
    if None in (candidate["nested_commit"], candidate["parent_commit"]):
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


def _candidate_identity() -> dict[str, Any]:
    nested_status = _git_output(ROOT, "status", "--porcelain=v1", "--untracked-files=no")
    parent_status = _git_output(PARENT_ROOT, "status", "--porcelain=v1", "--untracked-files=no")
    return {
        "nested_commit": _git_output(ROOT, "rev-parse", "HEAD"),
        "parent_commit": _git_output(PARENT_ROOT, "rev-parse", "HEAD"),
        "nested_tracked_clean": nested_status == "" if nested_status is not None else None,
        "parent_tracked_clean": parent_status == "" if parent_status is not None else None,
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
            reports.append(argument.split("=", 1)[1])
        elif argument == "--cov-report" and index + 1 < len(arguments):
            reports.append(arguments[index + 1])
    return reports


def _phase_pytest_args(arguments: list[str], *, defer_coverage_reports: bool) -> list[str]:
    if not defer_coverage_reports:
        return list(arguments)
    filtered = []
    skip_value = False
    for argument in arguments:
        if skip_value:
            skip_value = False
            continue
        if argument == "--cov-report":
            skip_value = True
        elif not argument.startswith("--cov-report="):
            filtered.append(argument)
    return [*filtered, "--cov-report="]


if __name__ == "__main__":
    raise SystemExit(measure())
