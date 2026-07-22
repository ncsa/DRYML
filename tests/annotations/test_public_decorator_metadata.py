from __future__ import annotations

import dryml
import dryml.annotations as ann


def test_env_req_passes_priority_and_merge_policy_as_metadata():
    @dryml.env.req(requirements=("torch>=2",), priority=10, merge_policy="override")
    def train():
        return None

    fragment = ann.own_fragments(train)[0]

    assert fragment.priority == 10
    assert fragment.merge_policy == "override"
    assert "priority" not in fragment.fragment
    assert "merge_policy" not in fragment.fragment


def test_world_req_and_default_pass_metadata_without_payload_pollution():
    @dryml.world.req(cpus={"min": 1}, priority=4, merge_policy="merge")
    @dryml.world.default(cpus=2, priority=5, merge_policy="replace")
    def train():
        return None

    default_fragment, requirement_fragment = ann.own_fragments(train)

    assert requirement_fragment.priority == 4
    assert requirement_fragment.merge_policy == "merge"
    assert default_fragment.priority == 5
    assert default_fragment.merge_policy == "replace"
    assert "priority" not in requirement_fragment.fragment
    assert "merge_policy" not in requirement_fragment.fragment
    assert "priority" not in default_fragment.fragment
    assert "merge_policy" not in default_fragment.fragment


def test_runtime_default_passes_metadata_without_payload_pollution():
    @dryml.runtime.default(torch={"num_threads": 2}, priority=3, merge_policy="merge")
    def train():
        return None

    fragment = ann.own_fragments(train)[0]

    assert fragment.priority == 3
    assert fragment.merge_policy == "merge"
    assert "priority" not in fragment.fragment
    assert "merge_policy" not in fragment.fragment
