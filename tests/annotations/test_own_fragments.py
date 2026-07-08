from __future__ import annotations

import dryml
import dryml.annotations as ann


def _requirements(fragments):
    values: list[str] = []
    for fragment in fragments:
        values.extend(fragment.fragment.get("requirements", ()))
    return tuple(values)


def test_own_fragments_returns_only_direct_fragments_for_function():
    @dryml.env.req(requirements=("fn>=1",))
    @dryml.world.req(cpus={"min": 1})
    def train():
        return None

    assert _requirements(ann.own_fragments(train, namespace="environment")) == ("fn>=1",)
    assert [fragment.kind for fragment in ann.own_fragments(train, kind="requirement")] == ["requirement", "requirement"]


def test_own_fragments_does_not_infer_owner_or_mro():
    @dryml.env.req(requirements=("class>=1",))
    class Model:
        @dryml.env.req(requirements=("method>=1",))
        def train(self):
            return None

    assert _requirements(ann.own_fragments(Model.train)) == ("method>=1",)
    assert _requirements(ann.own_fragments(Model.train, namespace="world")) == ()


def test_collect_fragments_accepts_single_target_and_appends_provider():
    provider = ann.AnnotationFragment(
        "environment",
        "requirement",
        {"requirements": ["provider>=1"]},
        ann.SourceTrace("provider", label="provider fragment"),
    )

    @dryml.env.req(requirements=("target>=1",))
    def train():
        return None

    fragments = ann.collect_fragments(train, provider_fragments=(provider,), namespace="environment")

    assert _requirements(fragments) == ("target>=1", "provider>=1")
    assert fragments[-1] is provider
