from types import SimpleNamespace

import pytest

from tests import timing_plugin
from tests.tools import test_buckets


class FakeItem:

    def __init__(self, nodeid, markers=(), params=None):
        self.nodeid = nodeid
        self.originalname = nodeid.rpartition("::")[-1].split("[", 1)[0]
        if params is not None:
            self.callspec = SimpleNamespace(params=params)
        self._markers = [
            getattr(pytest.mark, marker).mark for marker in markers
        ]

    def add_marker(self, marker, append=True):
        mark = (
            getattr(pytest.mark, marker).mark
            if isinstance(marker, str)
            else marker.mark
        )
        if append:
            self._markers.append(mark)
        else:
            self._markers.insert(0, mark)

    def get_closest_marker(self, name):
        return next(
            (
                marker
                for marker in reversed(self._markers)
                if marker.name == name
            ),
            None,
        )

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
    item = FakeItem(
        "tests/core/test_example.py::test_example", markers=("speed_heavy",)
    )
    config = SimpleNamespace(
        _dryml_tier_baseline={
            "path_tiers": {"tests/core/test_example.py": "smoke"}
        }
    )

    timing_plugin.pytest_collection_modifyitems(config, [item])

    assert item.get_closest_marker("speed_heavy") is not None
    assert item.get_closest_marker("speed_smoke") is None


def test_timing_plugin_baseline_assigns_expected_tiers():
    baseline = {
        "node_tiers": {"tests/core/test_example.py::test_node": "heavy"},
        "path_tiers": {"tests/core/test_example.py": "smoke"},
        "category_tiers": {"core": "medium"},
    }

    assert (
        timing_plugin.tier_for_nodeid(
            "tests/core/test_example.py::test_node", baseline
        )
        == "heavy"
    )
    assert (
        timing_plugin.tier_for_nodeid(
            "tests/core/test_example.py::test_other", baseline
        )
        == "smoke"
    )
    assert (
        timing_plugin.tier_for_nodeid(
            "tests/core/test_other.py::test_other", baseline
        )
        == "medium"
    )


def test_timing_plugin_path_tier_marks_critical_tests_directly():
    baseline = {
        "path_tiers": {"tests/core/test_state_ref_save.py": "medium"},
        "category_tiers": {"core": "smoke"},
    }

    assert (
        timing_plugin.tier_for_nodeid(
            "tests/core/test_state_ref_save.py::test_publication", baseline
        )
        == "medium"
    )


def test_timing_plugin_unknown_test_defaults_safely():
    assert (
        timing_plugin.tier_for_nodeid(
            "tests/new/test_case.py::test_case", {"default_tier": "bad"}
        )
        == "medium"
    )


def test_top_level_test_files_are_uncategorized():
    assert (
        timing_plugin.category_for_path("tests/test_timing_plugin.py")
        == "uncategorized"
    )
    assert (
        test_buckets.category_for_path("tests/test_timing_plugin.py")
        == "uncategorized"
    )


def test_unknown_only_deselects_known_nodeids():
    known = FakeItem("tests/core/test_example.py::test_known")
    unknown = FakeItem("tests/core/test_example.py::test_new")
    items = [known, unknown]
    config = FakeConfig(
        {
            "node_tiers": {known.nodeid: "smoke"},
            "path_tiers": {"tests/core/test_example.py": "smoke"},
        },
        unknown_only=True,
    )

    timing_plugin.pytest_collection_modifyitems(config, items)

    assert items == [unknown]
    assert config.hook.deselected == [known]


def test_unknown_only_keeps_path_tiered_tests_missing_from_node_tiers():
    item = FakeItem("tests/core/test_example.py::test_new")
    items = [item]
    config = FakeConfig(
        {
            "node_tiers": {},
            "path_tiers": {"tests/core/test_example.py": "smoke"},
        },
        unknown_only=True,
    )

    timing_plugin.pytest_collection_modifyitems(config, items)

    assert items == [item]
    assert item.get_closest_marker("speed_smoke") is not None
    assert config.hook.deselected == []


def test_unknown_only_off_keeps_known_nodeids():
    item = FakeItem("tests/core/test_example.py::test_known")
    items = [item]
    config = FakeConfig(
        {"node_tiers": {item.nodeid: "smoke"}}, unknown_only=False
    )

    timing_plugin.pytest_collection_modifyitems(config, items)

    assert items == [item]
    assert config.hook.deselected == []


def test_unknown_only_no_tests_collected_exit_is_success():
    session = SimpleNamespace(
        config=FakeConfig({"node_tiers": {}}, unknown_only=True),
        exitstatus=pytest.ExitCode.NO_TESTS_COLLECTED,
    )

    timing_plugin.pytest_sessionfinish(
        session, pytest.ExitCode.NO_TESTS_COLLECTED
    )

    assert session.exitstatus == pytest.ExitCode.OK


def _matrix_item(nodeid, **params):
    return FakeItem(f"tests/example.py::{nodeid}", params=params)


def test_representative_selection_covers_real_values_on_every_axis_stably():
    items = [
        _matrix_item(
            f"test_matrix[{left}-{right}]", left=left, right=right
        )
        for left in ("left0", "left1", "left2")
        for right in ("right0", "right1", "right2")
    ]
    profile = {
        "name": "toy",
        "sample_multi_axis_at": 4,
        "representative_functions": {
            "tests/example.py": {
                "rationale": "Retain the matrix boundary.",
                "functions": ["test_matrix"],
            }
        },
        "sampled_functions": {
            "tests/example.py::test_matrix": "genuine product",
        },
    }

    first, deselected, summary = timing_plugin.select_representatives(
        items, profile
    )
    second, _, _ = timing_plugin.select_representatives(items, profile)

    assert [item.nodeid for item in first] == [item.nodeid for item in second]
    assert len(first) == 5
    assert len(deselected) == 4
    assert {item.callspec.params["left"] for item in first} == {
        "left0", "left1", "left2",
    }
    assert {item.callspec.params["right"] for item in first} == {
        "right0", "right1", "right2",
    }
    assert summary["reasons"] == {
        "marginal matrix representatives": 5,
        "redundant matrix interactions": 4,
    }
    assert sum(summary["reasons"].values()) == summary["candidates"]


def test_tuple_rows_use_values_instead_of_shared_pytest_row_indices():
    rows = [
        ("alpha", (0, "x")),
        ("beta", (1, "y")),
        ("gamma", (2, "z")),
    ]
    items = [
        _matrix_item(
            f"test_tuple[{flavor}-{row[0]}]",
            flavor=flavor,
            number=row[0],
            label=row[1],
        )
        for flavor in ("plain", "spiced", "sweet")
        for _, row in rows
    ]

    selected, _, _ = timing_plugin.select_representatives(
        items,
        {
            "name": "toy",
            "sample_multi_axis_at": 4,
            "sampled_functions": {
                "tests/example.py::test_tuple": "genuine product",
            },
        },
    )

    assert {item.callspec.params["number"] for item in selected} == {0, 1, 2}
    assert {item.callspec.params["label"] for item in selected} == {
        "x", "y", "z",
    }


def test_shared_fixture_axis_and_explicit_interaction_pin_are_retained():
    items = [
        _matrix_item(
            f"test_shared[{fixture}-{value}]",
            shared_fixture=fixture,
            value=value,
        )
        for fixture in ("first", "second", "third")
        for value in (0, 1, 2)
    ]
    pinned = items[4].nodeid

    selected, _, _ = timing_plugin.select_representatives(
        items,
        {
            "name": "toy",
            "sample_multi_axis_at": 4,
            "sampled_functions": {
                "tests/example.py::test_shared": "genuine product",
            },
            "must_run": {pinned: "known interaction"},
        },
    )

    assert pinned in {item.nodeid for item in selected}
    assert {item.callspec.params["shared_fixture"] for item in selected} == {
        "first", "second", "third",
    }


def test_single_axis_and_safety_pinned_matrices_remain_complete():
    single = [
        _matrix_item(f"test_validation[{value}]", invalid=value)
        for value in range(12)
    ]
    safety = [
        FakeItem(
            f"tests/safety.py::test_race[{left}-{right}]",
            params={"left": left, "right": right},
        )
        for left in range(3)
        for right in range(3)
    ]

    selected, deselected, _ = timing_plugin.select_representatives(
        single + safety,
        {
            "name": "toy",
            "sample_multi_axis_at": 4,
            "exhaustive_paths": {"tests/safety.py": "concurrency"},
            "representative_functions": {
                "tests/safety.py": {
                    "rationale": "Safety pin overrides ordinary curation.",
                    "functions": ["test_race"],
                }
            },
        },
    )

    assert selected == single + safety
    assert deselected == []


def test_file_representatives_omit_only_named_file_repetitions():
    representative = FakeItem("tests/example.py::test_boundary")
    repetition = FakeItem("tests/example.py::test_repetition")
    future = FakeItem("tests/new_file.py::test_new")

    selected, deselected, _ = timing_plugin.select_representatives(
        [representative, repetition, future],
        {
            "name": "toy",
            "representative_functions": {
                "tests/example.py": {
                    "rationale": "Retain one boundary proof.",
                    "functions": ["test_boundary"],
                }
            },
        },
    )

    assert selected == [representative, future]
    assert deselected == [repetition]


def test_mandatory_node_survives_a_curated_function_allowlist():
    ordinary = FakeItem("tests/example.py::test_boundary")
    pinned = FakeItem("tests/example.py::test_matrix[required]")
    omitted = FakeItem("tests/example.py::test_matrix[repetition]")
    selected, deselected, summary = timing_plugin.select_representatives(
        [ordinary, pinned, omitted],
        {
            "representative_functions": {
                "tests/example.py": {"functions": ["test_boundary"]},
            },
            "must_run": {pinned.nodeid: "required interaction"},
        },
    )
    assert selected == [ordinary, pinned]
    assert deselected == [omitted]
    assert summary["reasons"]["mandatory node pin"] == 1
    assert sum(summary["reasons"].values()) == summary["candidates"]


def test_exhaustive_only_is_omitted_only_from_representative_profiles():
    ordinary = FakeItem("tests/example.py::test_boundary")
    matrix = FakeItem(
        "tests/example.py::test_matrix[one]", markers=("exhaustive_only",),
    )
    items = [ordinary, matrix]
    selected, deselected, summary = timing_plugin.select_representatives(
        items, {"exhaustive_paths": {"tests/example.py": "whole-file pin"}},
    )
    assert selected == [ordinary]
    assert deselected == [matrix]
    assert summary["reasons"]["explicit exhaustive-only cases"] == 1

    timing_plugin.pytest_collection_modifyitems(FakeConfig({}), items)
    assert items == [ordinary, matrix]


def test_exhaustive_only_cannot_silently_remove_a_mandatory_profile_node():
    item = FakeItem("tests/example.py::test_required", markers=("exhaustive_only",))
    with pytest.raises(pytest.UsageError, match="mandatory.*exhaustive_only"):
        timing_plugin.select_representatives(
            [item], {"must_run": {item.nodeid: "required boundary"}},
        )


def test_representative_stale_checks_are_scoped_to_collected_files():
    profile = {
        "representative_functions": {
            "tests/other.py": {
                "rationale": "Different runner phase.",
                "functions": ["test_missing_here"],
            }
        }
    }

    selected, deselected, _ = timing_plugin.select_representatives(
        [FakeItem("tests/example.py::test_present")], profile
    )

    assert [item.nodeid for item in selected] == [
        "tests/example.py::test_present"
    ]
    assert deselected == []


def test_representative_stale_checks_see_tier_filtered_functions():
    selected_item = FakeItem("tests/example.py::test_selected")
    heavy_item = FakeItem("tests/example.py::test_heavy")
    profile = {
        "representative_functions": {
            "tests/example.py": {
                "rationale": "The other function belongs to another tier.",
                "functions": ["test_selected", "test_heavy"],
            }
        }
    }

    selected, deselected, _ = timing_plugin.select_representatives(
        [selected_item], profile, collected_items=[selected_item, heavy_item]
    )

    assert selected == [selected_item]
    assert deselected == []


def test_representative_stale_checks_reject_missing_function_in_collected_file():
    profile = {
        "representative_functions": {
            "tests/example.py": {
                "rationale": "Typo must fail collection.",
                "functions": ["test_missing"],
            }
        }
    }

    with pytest.raises(pytest.UsageError, match="test_missing"):
        timing_plugin.select_representatives(
            [FakeItem("tests/example.py::test_present")], profile
        )


def test_good_enough_policy_retains_required_boundary_proofs():
    policy = timing_plugin._load_baseline(timing_plugin.DEFAULT_PROFILES)
    profile = policy["profiles"]["good-enough"]
    representatives = profile["representative_functions"]

    required = {
        "tests/dispatch/test_backend_execution.py": {
            "test_dispatch_submit_and_run_use_the_existing_core_one_off_owner",
        },
        "tests/execute/test_subprocess_backend.py": {
            "test_subprocess_setup_entry_failure_withholds_payload_deserialization",
            "test_subprocess_setup_teardown_failure_preserves_result_and_cleanup_evidence",
        },
        "tests/core/test_execute_integration.py": {
            "test_real_subprocess_core_pre_and_post_go_cancellation_are_not_reported_as_results",
        },
        "tests/core/test_repo_forks.py": {
            "test_interruption_after_state_fork_boundary_leaves_complete_discoverable_authority",
        },
        "tests/core/test_state_ref_save.py": {
            "test_zip_store_publishes_the_same_state_ref_authority",
        },
        "tests/managed/test_repo_routing.py": {
            "test_resume_uses_retained_checkpoint_authority_after_routes_change",
            "test_interrupted_completed_zip_commit_preserves_prior_checkpoint",
        },
    }
    for path, functions in required.items():
        assert functions <= set(representatives[path]["functions"])

    assert (
        "tests/core/test_execute_managed_integration.py::"
        "test_managed_decorator_orders_materialize_once_and_publish_one_final_state"
        "[subprocess-operation_afm-written_order0]"
    ) in profile["must_run"]

    for complete_path in (
        "tests/environments/test_environment_selection.py",
        "tests/execute/test_admission.py",
        "tests/execute/test_ray_backend_unit.py",
        "tests/dispatch/test_explain.py",
    ):
        assert complete_path not in representatives


def test_good_enough_does_not_collect_backend_training_modules():
    """Training-module imports must not preempt managed visibility setup."""
    baseline = test_buckets.load_baseline(test_buckets.DEFAULT_BASELINE)
    policy = test_buckets.load_baseline(test_buckets.DEFAULT_PROFILES)
    selected = test_buckets.select_profile_files(
        baseline, {"smoke", "medium"}, policy["profiles"]["good-enough"]
    )
    assert not {
        "./tests/models/test_tf_training.py",
        "./tests/models/test_torch_training.py",
    }.intersection(selected)
