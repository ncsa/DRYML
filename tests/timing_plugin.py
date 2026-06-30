"""Pytest support for DRYML test speed tiers and timing reports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest


VALID_TIERS = {"smoke", "medium", "heavy"}
DEFAULT_BASELINE = Path(__file__).with_name("test_tiers.json")


def pytest_addoption(parser):
    group = parser.getgroup("dryml-test-tiers")
    group.addoption(
        "--dryml-tier-baseline",
        action="store",
        default=str(DEFAULT_BASELINE),
        help="Path to the DRYML test tier baseline JSON file.",
    )
    group.addoption(
        "--dryml-timing-output",
        action="store",
        default=None,
        help="Write per-test duration data to this JSON file.",
    )
    group.addoption(
        "--dryml-timing-summary",
        action="store_true",
        help="Print a DRYML test timing summary at the end of the run.",
    )


def pytest_configure(config):
    for tier in sorted(VALID_TIERS):
        config.addinivalue_line("markers", f"speed_{tier}: auto-applied DRYML speed tier")
    config.addinivalue_line("markers", "category(name): auto-applied DRYML test category")
    config.addinivalue_line("markers", "timed: test duration was recorded by the DRYML timing plugin")
    baseline = _load_baseline(Path(config.getoption("--dryml-tier-baseline")))
    categories = set(baseline.get("category_tiers", {}))
    categories.update(category_for_path(path) for path in baseline.get("path_tiers", {}))
    for category in sorted(categories):
        config.addinivalue_line("markers", f"category_{category}: auto-applied DRYML category")
    config._dryml_tier_baseline = baseline
    config._dryml_timing_records = []


def pytest_collection_modifyitems(config, items):
    baseline = getattr(config, "_dryml_tier_baseline", {})
    for item in items:
        category = category_for_nodeid(item.nodeid)
        tier = tier_for_nodeid(item.nodeid, baseline)
        item.add_marker(f"speed_{tier}")
        if category:
            item.add_marker("category", category)
            item.add_marker(f"category_{category}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    config = item.config
    records = getattr(config, "_dryml_timing_records", None)
    if records is None:
        return
    records.append(
        {
            "nodeid": report.nodeid,
            "path": path_for_nodeid(report.nodeid),
            "category": category_for_nodeid(report.nodeid),
            "duration_seconds": float(report.duration),
            "outcome": report.outcome,
        }
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    records = getattr(config, "_dryml_timing_records", [])
    timing_output = config.getoption("--dryml-timing-output")
    if timing_output:
        payload = {
            "version": 1,
            "records": sorted(records, key=lambda item: item["nodeid"]),
        }
        output_path = Path(timing_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if config.getoption("--dryml-timing-summary"):
        _write_timing_summary(terminalreporter, records)


def _load_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def path_for_nodeid(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def category_for_nodeid(nodeid: str) -> str:
    path = path_for_nodeid(nodeid)
    return category_for_path(path)


def category_for_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 2 and parts[0] == "tests":
        return parts[1]
    return "uncategorized"


def tier_for_nodeid(nodeid: str, baseline: dict[str, Any]) -> str:
    explicit = _explicit_marker_tier(nodeid)
    if explicit is not None:
        return explicit
    node_tiers = baseline.get("node_tiers", {})
    if nodeid in node_tiers:
        return _validated_tier(node_tiers[nodeid])
    path = path_for_nodeid(nodeid)
    path_tiers = baseline.get("path_tiers", {})
    if path in path_tiers:
        return _validated_tier(path_tiers[path])
    category = category_for_nodeid(nodeid)
    category_tiers = baseline.get("category_tiers", {})
    if category in category_tiers:
        return _validated_tier(category_tiers[category])
    return _validated_tier(baseline.get("default_tier", "medium"))


def _explicit_marker_tier(nodeid: str) -> str | None:
    # Placeholder for future static parsing; runtime pytest markers are handled
    # by user-provided -m expressions after this plugin applies baseline tiers.
    return None


def _validated_tier(value: str) -> str:
    tier = str(value).strip().lower()
    if tier not in VALID_TIERS:
        return "medium"
    return tier


def _write_timing_summary(terminalreporter, records: list[dict[str, Any]]) -> None:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
    terminalreporter.write_sep("-", "DRYML timing summary")
    for category in sorted(by_category):
        total = sum(record["duration_seconds"] for record in by_category[category])
        count = len(by_category[category])
        terminalreporter.write_line(f"{category}: {count} tests, {total:.2f}s total")
    slowest = sorted(records, key=lambda item: item["duration_seconds"], reverse=True)[:10]
    if slowest:
        terminalreporter.write_line("slowest tests:")
        for record in slowest:
            terminalreporter.write_line(f"  {record['duration_seconds']:.2f}s {record['nodeid']}")
