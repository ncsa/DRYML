"""Pytest support for DRYML test profiles, speed tiers, and timings."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest


VALID_TIERS = {"smoke", "medium", "heavy"}
DEFAULT_BASELINE = Path(__file__).with_name("test_tiers.json")
DEFAULT_PROFILES = Path(__file__).with_name("test_profiles.json")


def pytest_addoption(parser):
    """Register DRYML's public profiling and internal runner options."""

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
    group.addoption(
        "--dryml-timing-unknown-only",
        action="store_true",
        help="When profiling, run only tests missing from baseline node_tiers.",
    )
    group.addoption(
        "--dryml-runner-tiers",
        action="store",
        default=None,
        help="Internal tests.sh tier constraint, as a comma-separated list.",
    )
    group.addoption(
        "--dryml-test-profile",
        action="store",
        default=None,
        help="Internal tests.sh representative selection profile.",
    )
    group.addoption(
        "--dryml-profile-policy",
        action="store",
        default=str(DEFAULT_PROFILES),
        help="Path to the DRYML representative profile policy JSON file.",
    )


def pytest_configure(config):
    """Load selection metadata and register generated DRYML markers."""

    for tier in sorted(VALID_TIERS):
        config.addinivalue_line("markers", f"speed_{tier}: auto-applied DRYML speed tier")
    config.addinivalue_line("markers", "category(name): auto-applied DRYML test category")
    config.addinivalue_line("markers", "timed: test duration was recorded by the DRYML timing plugin")
    config.addinivalue_line(
        "markers", "exhaustive_only: omitted from representative profiles; retained in explicit full selections"
    )
    baseline = _load_baseline(Path(config.getoption("--dryml-tier-baseline")))
    categories = {"uncategorized", *baseline.get("category_tiers", {})}
    categories.update(category_for_path(path) for path in baseline.get("path_tiers", {}))
    for category in sorted(categories):
        config.addinivalue_line("markers", f"category_{category}: auto-applied DRYML category")
    config._dryml_tier_baseline = baseline
    config._dryml_timing_records = []
    config._dryml_profile_policy = _load_baseline(
        Path(config.getoption("--dryml-profile-policy"))
    )
    config._dryml_profile_summary = None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Apply representative selection before user ``-k`` and ``-m`` filters."""

    collected_items = list(items)
    baseline = getattr(config, "_dryml_tier_baseline", {})
    runner_tiers = _runner_tiers(config)
    selected = []
    deselected = []
    for item in items:
        category = category_for_nodeid(item.nodeid)
        tier = tier_for_item(item, baseline)
        if not _has_marker(item, f"speed_{tier}"):
            item.add_marker(f"speed_{tier}")
        if category:
            item.add_marker(pytest.mark.category(category))
            item.add_marker(f"category_{category}")
        if runner_tiers is None or tier in runner_tiers:
            selected.append(item)
        else:
            deselected.append(item)
    items[:] = selected
    profile = _runner_profile(config)
    if profile is not None:
        profile_selected, profile_deselected, summary = select_representatives(
            items, profile, collected_items=collected_items
        )
        items[:] = profile_selected
        deselected.extend(profile_deselected)
        config._dryml_profile_summary = summary
    if _unknown_only_enabled(config):
        known = set(baseline.get("node_tiers", {}))
        selected = [item for item in items if item.nodeid not in known]
        deselected.extend(item for item in items if item.nodeid in known)
        items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)


def pytest_sessionfinish(session, exitstatus):
    """Treat an empty unknown-only timing update as successful."""

    if _unknown_only_enabled(session.config) and exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture call-phase timing records for optional timing output."""

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
    """Write optional timing data and a transparent profile summary."""

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
    summary = getattr(config, "_dryml_profile_summary", None)
    if summary is not None:
        terminalreporter.write_sep("-", "DRYML representative profile")
        terminalreporter.write_line(
            f"{summary['name']}: kept {summary['selected']} of "
            f"{summary['candidates']} candidates; deselected "
            f"{summary['deselected']}"
        )
        for reason, count in summary["reasons"].items():
            terminalreporter.write_line(f"  {reason}: {count}")


def _load_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _runner_profile(config) -> dict[str, Any] | None:
    """Return the configured representative profile, if enabled by tests.sh."""

    try:
        name = config.getoption("--dryml-test-profile")
    except (AttributeError, ValueError):
        return None
    if name is None:
        return None
    policy = getattr(config, "_dryml_profile_policy", {})
    profiles = policy.get("profiles", {})
    if name not in profiles:
        raise pytest.UsageError(f"unknown DRYML test profile: {name}")
    profile = dict(profiles[name])
    profile["name"] = name
    return profile


def select_representatives(
    items: list[Any],
    profile: dict[str, Any],
    *,
    collected_items: list[Any] | None = None,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Select deterministic representatives from collected pytest items.

    Files are complete unless policy names representative functions for that
    file or tests carry the ``exhaustive_only`` marker. A marked mandatory node
    raises UsageError rather than silently dropping a required proof.
    Matrix sampling is separately opt-in and uses callspec values rather
    than pytest row indices, retaining every value on every axis. Stale
    references are rejected only when their owning file was collected, so the
    same policy works across the runner's independent phases.
    """

    minimum_cases = int(profile.get("sample_multi_axis_at", 8))
    if minimum_cases < 2:
        raise pytest.UsageError("sample_multi_axis_at must be at least 2")
    exhaustive_paths = set(profile.get("exhaustive_paths", {}))
    exhaustive_functions = set(profile.get("exhaustive_functions", {}))
    representative_functions = profile.get("representative_functions", {})
    sampled_functions = set(profile.get("sampled_functions", {}))
    must_run = set(profile.get("must_run", {}))
    representative_names: dict[str, set[str]] = {}
    for path, selection in representative_functions.items():
        functions = selection.get("functions", ())
        if not functions:
            raise pytest.UsageError(
                f"representative_functions[{path!r}] must retain a function"
            )
        representative_names[path] = set(functions)
    groups: dict[str, list[Any]] = {}
    group_order: list[str] = []
    exhaustive_only_count = 0
    for item in items:
        if _has_marker(item, "exhaustive_only"):
            if item.nodeid in must_run:
                raise pytest.UsageError(
                    f"mandatory DRYML profile node is exhaustive_only: {item.nodeid}"
                )
            exhaustive_only_count += 1
            continue
        function = _function_nodeid(item)
        if function not in groups:
            groups[function] = []
            group_order.append(function)
        groups[function].append(item)

    collection = items if collected_items is None else collected_items
    collected_functions = {_function_nodeid(item) for item in collection}
    collected_paths = {
        path_for_nodeid(function) for function in collected_functions
    }
    collected_nodeids = {item.nodeid for item in collection}
    referenced_functions = exhaustive_functions | sampled_functions
    stale_functions = sorted(
        function
        for function in referenced_functions
        if path_for_nodeid(function) in collected_paths
        and function not in collected_functions
    )
    stale_representatives = sorted(
        f"{path}::{name}"
        for path, names in representative_names.items()
        if path in collected_paths
        for name in names
        if f"{path}::{name}" not in collected_functions
    )
    stale_nodes = sorted(
        nodeid
        for nodeid in must_run
        if path_for_nodeid(nodeid) in collected_paths
        and nodeid not in collected_nodeids
    )
    if stale_functions or stale_representatives or stale_nodes:
        stale = ", ".join(
            stale_functions + stale_representatives + stale_nodes
        )
        raise pytest.UsageError(f"stale DRYML profile references: {stale}")

    keep: set[int] = set()
    reasons: defaultdict[str, int] = defaultdict(int)
    if exhaustive_only_count:
        reasons["explicit exhaustive-only cases"] = exhaustive_only_count
    for function in group_order:
        group = groups[function]
        path = path_for_nodeid(function)
        selected_names = representative_names.get(path)
        if path in exhaustive_paths or function in exhaustive_functions:
            _keep_items(group, keep)
            reasons["safety/exhaustive pin"] += len(group)
            continue
        if selected_names is not None and _function_name(function) not in selected_names:
            mandatory = [item for item in group if item.nodeid in must_run]
            _keep_items(mandatory, keep)
            if mandatory:
                reasons["mandatory node pin"] += len(mandatory)
            reasons["deferred integration repetitions"] += len(group) - len(mandatory)
            continue
        callspecs = [getattr(item, "callspec", None) for item in group]
        axes = _parameter_axes(callspecs)
        if function not in sampled_functions:
            _keep_items(group, keep)
            if selected_names is not None:
                reasons["policy-selected functions"] += len(group)
            else:
                reasons["ordinary tests"] += len(group)
            continue
        if len(axes) < 2 or len(group) < minimum_cases:
            raise pytest.UsageError(
                f"sampled function is not a qualifying multi-axis product: {function}"
            )

        seen: set[tuple[str, Any]] = set()
        matrix_kept = 0
        for item, callspec in zip(group, callspecs):
            values = {
                (axis, _parameter_value_key(callspec.params[axis]))
                for axis in axes
                if axis in callspec.params
            }
            if values - seen or item.nodeid in must_run:
                keep.add(id(item))
                seen.update(values)
                matrix_kept += 1
        for item in group:
            if item.nodeid in must_run:
                keep.add(id(item))
        reasons["marginal matrix representatives"] += matrix_kept
        reasons["redundant matrix interactions"] += len(group) - matrix_kept

    selected = [item for item in items if id(item) in keep]
    deselected = [item for item in items if id(item) not in keep]
    summary = {
        "name": profile.get("name", "representative"),
        "candidates": len(items),
        "selected": len(selected),
        "deselected": len(deselected),
        "reasons": dict(sorted(reasons.items())),
    }
    return selected, deselected, summary


def _keep_items(items: list[Any], keep: set[int]) -> None:
    """Add item identities to a representative selection set."""

    keep.update(id(item) for item in items)


def _function_nodeid(item: Any) -> str:
    """Return a parameter-independent node ID for one pytest function item."""

    prefix, separator, leaf = item.nodeid.rpartition("::")
    original = getattr(item, "originalname", None)
    if original:
        return f"{prefix}{separator}{original}"
    return f"{prefix}{separator}{leaf.split('[', 1)[0]}"


def _function_name(function_nodeid: str) -> str:
    """Return the leaf function name from a parameter-independent node ID."""

    return function_nodeid.rpartition("::")[-1]


def _parameter_axes(callspecs: list[Any]) -> tuple[str, ...]:
    """Return callspec axes in pytest's stable parameter insertion order."""

    for callspec in callspecs:
        if callspec is not None:
            return tuple(callspec.params)
    return ()


def _parameter_value_key(value: Any) -> Any:
    """Return a stable, hashable identity for a real callspec parameter value."""

    value_type = type(value)
    type_name = f"{value_type.__module__}.{value_type.__qualname__}"
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return type_name, value
    if isinstance(value, (tuple, list)):
        return type_name, tuple(_parameter_value_key(part) for part in value)
    if isinstance(value, dict):
        parts = (
            (_parameter_value_key(key), _parameter_value_key(part))
            for key, part in value.items()
        )
        return type_name, tuple(sorted(parts, key=repr))
    if isinstance(value, (set, frozenset)):
        parts = (_parameter_value_key(part) for part in value)
        return type_name, tuple(sorted(parts, key=repr))
    if isinstance(value, type):
        return "builtins.type", value.__module__, value.__qualname__
    try:
        hash(value)
    except (TypeError, ValueError):
        return type_name, repr(value)
    return type_name, value


def _unknown_only_enabled(config) -> bool:
    getoption = getattr(config, "getoption", None)
    if getoption is None:
        return bool(getattr(config, "_dryml_timing_unknown_only", False))
    try:
        return bool(getoption("--dryml-timing-unknown-only"))
    except (AttributeError, ValueError):
        return bool(getattr(config, "_dryml_timing_unknown_only", False))


def _runner_tiers(config) -> set[str] | None:
    """Return the internal runner tier constraint, if one was supplied."""

    try:
        value = config.getoption("--dryml-runner-tiers")
    except (AttributeError, ValueError):
        return None
    if value is None:
        return None
    tiers = {tier.strip() for tier in value.split(",") if tier.strip()}
    invalid = tiers - VALID_TIERS
    if not tiers or invalid:
        detail = ", ".join(sorted(invalid)) or "empty tier set"
        raise pytest.UsageError(f"invalid tests.sh runner tiers: {detail}")
    return tiers


def path_for_nodeid(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def category_for_nodeid(nodeid: str) -> str:
    path = path_for_nodeid(nodeid)
    return category_for_path(path)


def category_for_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 3 and parts[0] == "tests":
        return parts[1]
    return "uncategorized"


def tier_for_nodeid(nodeid: str, baseline: dict[str, Any]) -> str:
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


def tier_for_item(item, baseline: dict[str, Any]) -> str:
    explicit = _explicit_marker_tier(item)
    if explicit is not None:
        return explicit
    return tier_for_nodeid(item.nodeid, baseline)


def _explicit_marker_tier(item) -> str | None:
    for tier in ("heavy", "medium", "smoke"):
        if _has_marker(item, f"speed_{tier}"):
            return tier
    return None


def _has_marker(item, name: str) -> bool:
    getter = getattr(item, "get_closest_marker", None)
    if getter is not None and getter(name) is not None:
        return True
    iterator = getattr(item, "iter_markers", None)
    if iterator is not None:
        return any(marker.name == name for marker in iterator())
    return False


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
