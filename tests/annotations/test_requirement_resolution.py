from __future__ import annotations

import json

import dryml
import dryml.annotations as ann


def _env_fragment(requirement: str, *, priority: int, merge_policy: str | None = None) -> ann.AnnotationFragment:
    return ann.AnnotationFragment(
        "environment",
        "requirement",
        dryml.env.normalize_environment_requirement_fragment(requirements=(requirement,)),
        ann.SourceTrace("synthetic", label=requirement),
        priority=priority,
        merge_policy=merge_policy,
    )


def _world_req_fragment(cpus_min: int, *, priority: int, merge_policy: str | None = None) -> ann.AnnotationFragment:
    return ann.AnnotationFragment(
        "world",
        "requirement",
        dryml.world.requirement_fragment(cpus={"min": cpus_min}),
        ann.SourceTrace("synthetic", label=f"cpus>={cpus_min}"),
        priority=priority,
        merge_policy=merge_policy,
    )


def _world_default_fragment(cpus: int, *, priority: int, merge_policy: str | None = None) -> ann.AnnotationFragment:
    return ann.AnnotationFragment(
        "world",
        "default",
        dryml.world.default_fragment_data(cpus=cpus),
        ann.SourceTrace("synthetic", label=f"cpus={cpus}"),
        priority=priority,
        merge_policy=merge_policy,
    )


def test_resolve_fragments_returns_requirement_resolution_with_traces_and_data():
    provider = ann.AnnotationFragment(
        "environment",
        "requirement",
        {"requirements": ["provider>=1"]},
        ann.SourceTrace("provider", label="provider fragment"),
    )

    @dryml.env.req(requirements=("class>=1",))
    class Model:
        @dryml.world.req(cpus={"min": 2})
        @dryml.runtime.default(torch={"num_threads": 2})
        def train(self):
            return None

    fragments = ann.fragments_for_method(Model, "train")
    resolution = ann.resolve_fragments(fragments, provider_fragments=(provider,), source="unit-test")

    assert isinstance(resolution, ann.RequirementResolution)
    assert tuple(resolution.environment_requirement.requirements) == ("class>=1", "provider>=1")
    assert resolution.world_requirement.to_data()["roles"]["main"]["resources"]["cpus"]["min"] == 2
    assert resolution.runtime_default.frameworks["torch"]["num_threads"] == 2
    assert resolution.fragments[-1] is provider
    assert any(trace.label == "provider fragment" for trace in resolution.source_traces)
    json.dumps(resolution.to_data(), sort_keys=True)


def test_environment_requirement_merge_uses_stable_priority_order_for_override():
    low = _env_fragment("low>=1", priority=0, merge_policy="override")
    high = _env_fragment("high>=1", priority=10, merge_policy="override")

    resolution = ann.resolve_fragments((high, low))

    assert tuple(resolution.environment_requirement.requirements) == ("high>=1",)
    assert resolution.fragments == (high, low)


def test_world_requirement_merge_uses_stable_priority_order_for_override():
    low = _world_req_fragment(1, priority=0, merge_policy="override")
    high = _world_req_fragment(4, priority=10, merge_policy="override")

    resolution = ann.resolve_fragments((high, low))

    assert resolution.world_requirement.roles["main"].resources.cpus.to_data() == {"min": 4}
    assert resolution.fragments == (high, low)


def test_world_default_merge_uses_stable_priority_order_for_replace():
    low = _world_default_fragment(1, priority=0)
    high = _world_default_fragment(4, priority=10, merge_policy="replace")

    resolution = ann.resolve_fragments((high, low))

    assert resolution.world_default.roles["main"].process.resources.cpus == 4
    assert resolution.fragments == (high, low)


def test_resolve_target_requirements_supports_live_lambda():
    target = dryml.env.req(requirements=("lambda>=1",))(lambda: None)

    resolution = ann.resolve_target_requirements(target)

    assert tuple(resolution.environment_requirement.requirements) == ("lambda>=1",)


def test_requirement_resolution_captures_merge_diagnostics():
    first = ann.AnnotationFragment("environment", "requirement", {"requirements": ["pkg==1"]}, ann.SourceTrace("synthetic", label="first"))
    second = ann.AnnotationFragment("environment", "requirement", {"requirements": ["pkg==2"]}, ann.SourceTrace("synthetic", label="second"))

    resolution = ann.resolve_fragments((first, second))

    assert resolution.diagnostics
    assert resolution.diagnostics[0].level == "error"
    assert resolution.to_data()["diagnostics"][0]["code"] == "dryml.annotations.merge_issue"
