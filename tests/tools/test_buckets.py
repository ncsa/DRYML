#!/usr/bin/env python3
"""Manage DRYML pytest speed buckets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = ROOT / "tests"
DEFAULT_BASELINE = TESTS_DIR / "test_tiers.json"
VALID_TIERS = ("smoke", "medium", "heavy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select and update DRYML test speed buckets"
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="test tier baseline JSON path",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser(
        "select", help="print test files for a tier set"
    )
    select.add_argument("tiers", nargs="+", choices=VALID_TIERS + ("full",))

    summary = subparsers.add_parser("summary", help="print bucket summary")
    summary.add_argument(
        "--all-files",
        action="store_true",
        help="include files not listed in the baseline",
    )

    update = subparsers.add_parser(
        "update", help="update node tiers from timing output"
    )
    update.add_argument(
        "timings",
        nargs="+",
        help="JSON files written by pytest --dryml-timing-output",
    )
    update.add_argument(
        "--output",
        default=None,
        help="updated baseline path; defaults to --baseline",
    )

    prune = subparsers.add_parser(
        "prune", help="remove overrides for no-longer-collected tests"
    )
    prune.add_argument(
        "--output",
        default=None,
        help="updated baseline path; defaults to --baseline",
    )

    args = parser.parse_args(argv)
    baseline = load_baseline(Path(args.baseline))
    if args.command == "select":
        tiers = set(VALID_TIERS if "full" in args.tiers else args.tiers)
        for path in select_files(baseline, tiers):
            print(path)
        return 0
    if args.command == "summary":
        print_summary(baseline, include_all=args.all_files)
        return 0
    if args.command == "update":
        updated = update_from_timings(
            baseline, [Path(path) for path in args.timings]
        )
        output = Path(args.output) if args.output else Path(args.baseline)
        output.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
        return 0
    if args.command == "prune":
        updated = prune_stale_node_tiers(baseline, collected_test_nodeids())
        output = Path(args.output) if args.output else Path(args.baseline)
        output.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
        return 0
    return 2


def load_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "default_tier": "medium",
            "category_tiers": {},
            "path_tiers": {},
            "node_tiers": {},
        }
    return json.loads(path.read_text())


def iter_test_files() -> list[Path]:
    files = []
    for path in TESTS_DIR.rglob("test_*.py"):
        rel = path.relative_to(ROOT)
        parts = rel.parts
        if len(parts) > 1 and parts[1] in {"old", "dev"}:
            continue
        files.append(rel)
    return sorted(files)


def select_files(baseline: dict[str, Any], tiers: set[str]) -> list[str]:
    selected = []
    for rel in iter_test_files():
        rel_text = rel.as_posix()
        if tier_for_path(rel_text, baseline) in tiers or has_node_tier(
            rel_text, baseline, tiers
        ):
            selected.append("./" + rel_text)
    return selected


def has_node_tier(
    path: str, baseline: dict[str, Any], tiers: set[str]
) -> bool:
    prefix = path + "::"
    return any(
        nodeid.startswith(prefix) and validate_tier(tier) in tiers
        for nodeid, tier in baseline.get("node_tiers", {}).items()
    )


def category_for_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 3 and parts[0] == "tests":
        return parts[1]
    return "uncategorized"


def tier_for_path(path: str, baseline: dict[str, Any]) -> str:
    if path in baseline.get("path_tiers", {}):
        return validate_tier(baseline["path_tiers"][path])
    category = category_for_path(path)
    if category in baseline.get("category_tiers", {}):
        return validate_tier(baseline["category_tiers"][category])
    return validate_tier(baseline.get("default_tier", "medium"))


def validate_tier(value: str) -> str:
    value = str(value).strip().lower()
    return value if value in VALID_TIERS else "medium"


def print_summary(baseline: dict[str, Any], *, include_all: bool) -> None:
    counts: dict[str, dict[str, int]] = {}
    for rel in iter_test_files():
        rel_text = rel.as_posix()
        tier = (
            tier_for_path(rel_text, baseline)
            if include_all or rel_text in baseline.get("path_tiers", {})
            else None
        )
        if tier is None:
            continue
        category = category_for_path(rel_text)
        counts.setdefault(category, {tier: 0 for tier in VALID_TIERS})[
            tier
        ] += 1
    print("category smoke medium heavy")
    for category in sorted(counts):
        row = counts[category]
        print(f"{category} {row['smoke']} {row['medium']} {row['heavy']}")


def update_from_timings(
    baseline: dict[str, Any], timing_paths: list[Path]
) -> dict[str, Any]:
    thresholds = baseline.get(
        "thresholds", {"smoke_seconds": 0.25, "medium_seconds": 2.0}
    )
    smoke_seconds = float(thresholds.get("smoke_seconds", 0.25))
    medium_seconds = float(thresholds.get("medium_seconds", 2.0))
    durations: dict[str, list[float]] = {}
    for timings_path in timing_paths:
        timings = json.loads(timings_path.read_text())
        for record in timings.get("records", ()):
            if record.get("outcome") != "passed":
                continue
            durations.setdefault(record["nodeid"], []).append(
                float(record["duration_seconds"])
            )
    node_tiers = dict(baseline.get("node_tiers", {}))
    for nodeid, values in durations.items():
        duration = median(values)
        if duration <= smoke_seconds:
            node_tiers[nodeid] = "smoke"
        elif duration <= medium_seconds:
            node_tiers[nodeid] = "medium"
        else:
            node_tiers[nodeid] = "heavy"
    updated = dict(baseline)
    updated["node_tiers"] = dict(sorted(node_tiers.items()))
    return updated


def collected_test_nodeids() -> set[str]:
    """Return maintained pytest node IDs using import-safe collection mode."""

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--import-mode=importlib",
            "tests",
            "--ignore=tests/old",
            "--ignore=tests/dev",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    }


def prune_stale_node_tiers(
    baseline: dict[str, Any], nodeids: set[str]
) -> dict[str, Any]:
    """Return baseline metadata without overrides for retired pytest nodes."""

    updated = dict(baseline)
    updated["node_tiers"] = {
        nodeid: tier
        for nodeid, tier in sorted(baseline.get("node_tiers", {}).items())
        if nodeid in nodeids
    }
    return updated


def test_every_maintained_category_has_an_explicit_tier() -> None:
    """Keep newly added test categories visible to tier selection."""

    baseline = load_baseline(DEFAULT_BASELINE)
    categories = {
        category_for_path(path.as_posix()) for path in iter_test_files()
    }
    assert categories <= set(baseline["category_tiers"])


def test_tier_metadata_references_existing_test_files() -> None:
    """Reject stale file and node overrides in the checked-in baseline."""

    baseline = load_baseline(DEFAULT_BASELINE)
    references = set(baseline.get("path_tiers", {}))
    references.update(
        nodeid.split("::", 1)[0] for nodeid in baseline.get("node_tiers", {})
    )
    missing = sorted(
        path for path in references if not (ROOT / path).is_file()
    )
    assert missing == []


def test_tier_metadata_references_collected_test_nodes() -> None:
    """Reject stale overrides in modules collectable with installed extras.

    Optional-framework modules may skip collection in the lightweight CI
    environment. Their file paths remain covered by the preceding existence
    gate and their node IDs are validated whenever those extras are installed.
    """

    baseline = load_baseline(DEFAULT_BASELINE)
    collected = collected_test_nodeids()
    collected_paths = {path_for_nodeid(nodeid) for nodeid in collected}
    stale = sorted(
        nodeid
        for nodeid in baseline.get("node_tiers", {})
        if path_for_nodeid(nodeid) in collected_paths
        and nodeid not in collected
    )
    assert stale == []


def test_every_maintained_test_has_an_intentional_tier() -> None:
    """Require a node, path, or category tier instead of the default fallback."""

    baseline = load_baseline(DEFAULT_BASELINE)
    node_tiers = baseline.get("node_tiers", {})
    path_tiers = baseline.get("path_tiers", {})
    category_tiers = baseline.get("category_tiers", {})
    missing = sorted(
        nodeid
        for nodeid in collected_test_nodeids()
        if nodeid not in node_tiers
        and path_for_nodeid(nodeid) not in path_tiers
        and category_for_path(path_for_nodeid(nodeid)) not in category_tiers
    )
    assert missing == []


def path_for_nodeid(nodeid: str) -> str:
    """Return the test-file component of a pytest node ID."""

    return nodeid.split("::", 1)[0]


def test_all_tiers_cover_every_maintained_test_file() -> None:
    """Ensure the complete tier union cannot silently omit a test file."""

    baseline = load_baseline(DEFAULT_BASELINE)
    expected = {"./" + path.as_posix() for path in iter_test_files()}
    assert set(select_files(baseline, set(VALID_TIERS))) == expected


if __name__ == "__main__":
    raise SystemExit(main())
