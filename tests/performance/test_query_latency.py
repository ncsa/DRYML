"""Hard wall-clock canaries for notebook-scale public Repo queries.

One bounded builder prepares the Store for every scenario. Each scenario then
gets a fresh Python process and a hard deadline covering startup, reopen, selector
construction and two real public queries. TimeoutExpired kills/reaps a stalled
worker even inside a native SQLite call; unfinished work never counts as a pass.
"""

import json
import subprocess
import sys
from pathlib import Path
import time

import pytest

from tests.performance.query_workload import (
    BUILD_TIMEOUT_SECONDS, INSTANCES, PLANNING_TIMEOUT_SECONDS, QUERY_TIMEOUT_SECONDS,
    SCENARIO_TIMEOUT_BUDGET_SECONDS, SCENARIOS,
)


pytestmark = [pytest.mark.speed_medium, pytest.mark.timeout(110)]
_ROOT = Path(__file__).resolve().parents[2]


def _worker(action, store, timeout):
    started = time.monotonic()
    command = [sys.executable, "-m", "tests.performance.query_worker", action, str(store)]
    try:
        result = subprocess.run(command, cwd=_ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"{action} exceeded its {timeout:g}s hard wall deadline; worker killed/reaped. "
                    f"stdout={error.stdout!r}; stderr={error.stderr!r}", pytrace=False)
    assert result.returncode == 0, f"{action} worker failed:\n{result.stdout}\n{result.stderr}"
    data = json.loads(result.stdout)
    data["worker_seconds"] = time.monotonic() - started
    return data


def test_query_canary_budgets_remain_bounded():
    """Prevent scenario growth or deadline inflation from consuming the routine gate."""
    assert SCENARIOS and len({case.name for case in SCENARIOS}) == len(SCENARIOS)
    assert BUILD_TIMEOUT_SECONDS <= 90
    assert all(type(case.expected) is int and case.expected >= 0 for case in SCENARIOS)
    assert all(0 < case.timeout_seconds <= QUERY_TIMEOUT_SECONDS <= 8 for case in SCENARIOS)
    assert sum(case.timeout_seconds for case in SCENARIOS) <= SCENARIO_TIMEOUT_BUDGET_SECONDS <= 60
    assert 0 < PLANNING_TIMEOUT_SECONDS <= 60


@pytest.fixture(scope="module")
def query_store(tmp_path_factory):
    """Build one immutable notebook-scale Store under a separate 90-second cap."""
    store = tmp_path_factory.mktemp("query-performance") / "store"
    summary = _worker("build", store, BUILD_TIMEOUT_SECONDS)
    assert summary["experiments"] == INSTANCES == 70
    assert summary["copies"] == 5 and len(summary["widths"]) == 7
    assert summary["histories"] == 28
    assert summary["families"]["query_graph"] >= 400
    assert summary["families"]["discovery"] >= 2000
    assert summary["families"]["facet"] >= 10000
    return store, summary


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda case: case.name)
def test_notebook_queries_finish_within_budget(query_store, scenario, record_property):
    """Require exact cold/repeated answers before the whole worker's deadline."""
    store, summary = query_store
    result = _worker(scenario.name, store, scenario.timeout_seconds)
    assert [item["count"] for item in result["results"]] == [scenario.expected, scenario.expected]
    record_property("fixture_build_seconds", summary["build_seconds"])
    record_property("fixture_graph_blocks", summary["families"]["query_graph"])
    record_property("worker_budget_seconds", scenario.timeout_seconds)
    record_property("worker_seconds", result["worker_seconds"])
    for item in result["results"]:
        record_property(item["phase"] + "_query_seconds", item["seconds"])


@pytest.mark.timeout(160)
def test_notebook_trial_planning_finishes_within_one_minute(query_store, record_property):
    """Bound full selection, seven refinements, status inspection and retry planning."""
    store, summary = query_store
    result = _worker("trial_planning", store, PLANNING_TIMEOUT_SECONDS)
    assert (result["candidates"], result["completed"], result["planned"]) == (35, 25, 10)
    record_property("fixture_build_seconds", summary["build_seconds"])
    record_property("worker_budget_seconds", PLANNING_TIMEOUT_SECONDS)
    record_property("worker_seconds", result["worker_seconds"])
    record_property("planning_seconds", result["seconds"])
