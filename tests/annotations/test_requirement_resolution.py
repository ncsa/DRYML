from __future__ import annotations

import json

import dryml
import dryml.annotations as ann


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
