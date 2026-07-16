#!/usr/bin/env python3
"""Run bounded, non-mutating DRYML test-suite measurements."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOG_LIMIT_BYTES = 1024 * 1024
VALID_MODES = ("smoke", "medium", "heavy", "full")


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
    command = [sys.executable, "-m", "pytest"]
    if coverage:
        command.append("--cov=dryml")
        if append_coverage:
            command.append("--cov-append")
    else:
        command.append("--no-cov")
    command.extend(["-m", markexpr, *selected_files(tiers)])
    command.extend([
        "--dryml-timing-output", str(timing_path),
        "--dryml-timing-phase", phase,
        *pytest_args,
    ])
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        (output_dir / f"{phase}.{stream}.log").write_bytes(capture.buffer)
    return {
        "phase": phase,
        "command": command,
        "returncode": returncode,
        "wall_seconds": time.monotonic() - started,
        "stdout": {"original_bytes": stdout.original_bytes, "truncated": stdout.original_bytes > LOG_LIMIT_BYTES},
        "stderr": {"original_bytes": stderr.original_bytes, "truncated": stderr.original_bytes > LOG_LIMIT_BYTES},
    }


def measure(argv: list[str] | None = None) -> int:
    """Run one requested measurement mode and emit versioned JSON artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("mode", choices=VALID_MODES)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        output_dir = fresh_output_dir(args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    output_dir.mkdir(mode=0o700, parents=True)
    if args.mode == "full":
        phase_specs = (
            ("medium", ["smoke", "medium"], "speed_smoke or speed_medium", True, False),
            ("heavy", ["heavy"], "speed_heavy", True, True),
        )
    else:
        tiers, markexpr = {
            "smoke": (["smoke"], "speed_smoke"),
            "medium": (["smoke", "medium"], "speed_smoke or speed_medium"),
            "heavy": (["heavy"], "speed_heavy"),
        }[args.mode]
        phase_specs = ((args.mode, tiers, markexpr, False, False),)
    phases = []
    before_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    for phase, tiers, markexpr, coverage, append in phase_specs:
        result = run_phase(
            output_dir=output_dir, phase=phase, tiers=tiers, markexpr=markexpr,
            coverage=coverage, append_coverage=append, pytest_args=args.pytest_args,
        )
        phases.append(result)
        if result["returncode"]:
            break
    nodes = []
    for timing_path in sorted(output_dir.glob("timing-*.json")):
        nodes.extend(json.loads(timing_path.read_text()).get("records", ()))
    status = "success" if phases and all(phase["returncode"] == 0 for phase in phases) else "failure"
    run = {
        "schema": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "mode": args.mode,
        "candidate": {"nested_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
        "environment": {"python": sys.version, "platform": platform.platform()},
        "coverage": args.mode == "full",
        "phases": phases,
        "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "peak_child_rss_kib_before": before_rss,
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    (output_dir / "nodes.json").write_text(json.dumps({"schema": 1, "records": nodes}, indent=2, sort_keys=True) + "\n")
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(measure())
