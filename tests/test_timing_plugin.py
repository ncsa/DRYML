from types import SimpleNamespace

import pytest

from tests import timing_plugin
from tests.tools import test_buckets


class FakeItem:
    def __init__(self, nodeid, markers=()):
        self.nodeid = nodeid
        self._markers = [getattr(pytest.mark, marker).mark for marker in markers]

    def add_marker(self, marker, append=True):
        mark = getattr(pytest.mark, marker).mark if isinstance(marker, str) else marker.mark
        if append:
            self._markers.append(mark)
        else:
            self._markers.insert(0, mark)

    def get_closest_marker(self, name):
        return next((marker for marker in reversed(self._markers) if marker.name == name), None)

    def iter_markers(self):
        return iter(self._markers)


class FakeHook:
    def __init__(self):
        self.deselected = []

    def pytest_deselected(self, items):
        self.deselected.extend(items)


class FakeConfig:
    def __init__(self, baseline, *, unknown_only=False):
        self._dryml_tier_baseline = baseline
        self._dryml_timing_unknown_only = unknown_only
        self.hook = FakeHook()

    def getoption(self, option):
        if option == "--dryml-timing-unknown-only":
            return self._dryml_timing_unknown_only
        raise ValueError(option)


def test_timing_plugin_adds_category_marker_with_argument():
    item = FakeItem("tests/core/test_example.py::test_example")
    config = SimpleNamespace(_dryml_tier_baseline={"default_tier": "medium"})

    timing_plugin.pytest_collection_modifyitems(config, [item])

    category = item.get_closest_marker("category")
    assert category is not None
    assert category.args == ("core",)
    assert item.get_closest_marker("category_core") is not None


def test_timing_plugin_honors_explicit_speed_markers():
    item = FakeItem("tests/core/test_example.py::test_example", markers=("speed_heavy",))
    config = SimpleNamespace(_dryml_tier_baseline={"path_tiers": {"tests/core/test_example.py": "smoke"}})

    timing_plugin.pytest_collection_modifyitems(config, [item])

    assert item.get_closest_marker("speed_heavy") is not None
    assert item.get_closest_marker("speed_smoke") is None


def test_timing_plugin_baseline_assigns_expected_tiers():
    baseline = {
        "node_tiers": {"tests/core/test_example.py::test_node": "heavy"},
        "path_tiers": {"tests/core/test_example.py": "smoke"},
        "category_tiers": {"core": "medium"},
    }

    assert timing_plugin.tier_for_nodeid("tests/core/test_example.py::test_node", baseline) == "heavy"
    assert timing_plugin.tier_for_nodeid("tests/core/test_example.py::test_other", baseline) == "smoke"
    assert timing_plugin.tier_for_nodeid("tests/core/test_other.py::test_other", baseline) == "medium"


def test_timing_plugin_unknown_test_defaults_safely():
    assert timing_plugin.tier_for_nodeid("tests/new/test_case.py::test_case", {"default_tier": "bad"}) == "medium"


def test_top_level_test_files_are_uncategorized():
    assert timing_plugin.category_for_path("tests/test_timing_plugin.py") == "uncategorized"
    assert test_buckets.category_for_path("tests/test_timing_plugin.py") == "uncategorized"


def test_unknown_only_deselects_known_nodeids():
    known = FakeItem("tests/core/test_example.py::test_known")
    unknown = FakeItem("tests/core/test_example.py::test_new")
    items = [known, unknown]
    config = FakeConfig(
        {"node_tiers": {known.nodeid: "smoke"}, "path_tiers": {"tests/core/test_example.py": "smoke"}},
        unknown_only=True,
    )

    timing_plugin.pytest_collection_modifyitems(config, items)

    assert items == [unknown]
    assert config.hook.deselected == [known]


def test_unknown_only_keeps_path_tiered_tests_missing_from_node_tiers():
    item = FakeItem("tests/core/test_example.py::test_new")
    items = [item]
    config = FakeConfig(
        {"node_tiers": {}, "path_tiers": {"tests/core/test_example.py": "smoke"}},
        unknown_only=True,
    )

    timing_plugin.pytest_collection_modifyitems(config, items)

    assert items == [item]
    assert item.get_closest_marker("speed_smoke") is not None
    assert config.hook.deselected == []


def test_unknown_only_off_keeps_known_nodeids():
    item = FakeItem("tests/core/test_example.py::test_known")
    items = [item]
    config = FakeConfig({"node_tiers": {item.nodeid: "smoke"}}, unknown_only=False)

    timing_plugin.pytest_collection_modifyitems(config, items)

    assert items == [item]
    assert config.hook.deselected == []


def test_unknown_only_no_tests_collected_exit_is_success():
    session = SimpleNamespace(
        config=FakeConfig({"node_tiers": {}}, unknown_only=True),
        exitstatus=pytest.ExitCode.NO_TESTS_COLLECTED,
    )

    timing_plugin.pytest_sessionfinish(session, pytest.ExitCode.NO_TESTS_COLLECTED)

    assert session.exitstatus == pytest.ExitCode.OK
