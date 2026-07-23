from __future__ import annotations

from collections.abc import Mapping

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


def test_collect_fragments_treats_mapping_like_target_as_single_target():
    class MappingTarget(Mapping):
        def __getitem__(self, key):
            return {"key": "value"}[key]

        def __iter__(self):
            return iter(("key",))

        def __len__(self):
            return 1

    target = dryml.env.req(requirements=("mapping-target>=1",))(MappingTarget())

    assert _requirements(ann.collect_fragments(target, namespace="environment")) == ("mapping-target>=1",)
