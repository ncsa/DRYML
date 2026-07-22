from __future__ import annotations

import inspect

import dryml
import dryml.annotations as ann
import pytest
from dryml.core2 import Object
from dryml.managed import ManagedOutput, managed


class DecoratorTargets(Object):
    @managed(outputs=(ManagedOutput("result", primary=True),))
    @dryml.env.req(requirements=("managed-test>=1",))
    def managed_outer_env(self):
        return "called"

    @dryml.env.req(requirements=("managed-test>=1",))
    @managed(outputs=(ManagedOutput("result", primary=True),))
    def managed_inner_env(self):
        return "called"

    @managed(outputs=(ManagedOutput("result", primary=True),))
    @dryml.world.req(cpus={"exact": 1})
    def managed_outer_world_req(self):
        return "called"

    @dryml.world.req(cpus={"exact": 1})
    @managed(outputs=(ManagedOutput("result", primary=True),))
    def managed_inner_world_req(self):
        return "called"

    @managed(outputs=(ManagedOutput("result", primary=True),))
    @dryml.world.default(cpus=1)
    def managed_outer_world_default(self):
        return "called"

    @dryml.world.default(cpus=1)
    @managed(outputs=(ManagedOutput("result", primary=True),))
    def managed_inner_world_default(self):
        return "called"


@pytest.mark.parametrize(
    ("method_name", "namespace", "kind"),
    [
        ("managed_outer_env", "environment", "requirement"),
        ("managed_inner_env", "environment", "requirement"),
        ("managed_outer_world_req", "world", "requirement"),
        ("managed_inner_world_req", "world", "requirement"),
        ("managed_outer_world_default", "world", "default"),
        ("managed_inner_world_default", "world", "default"),
    ],
)
def test_managed_composes_with_requirement_decorators_in_both_orders(
    method_name,
    namespace,
    kind,
):
    raw = inspect.getattr_static(DecoratorTargets, method_name)
    fragments = ann.fragments_for_method(DecoratorTargets, method_name)
    bound = getattr(DecoratorTargets(), method_name)

    assert bound() == "called"
    assert bound.__func__ is raw.__func__
    assert bound.result.slot == "result"
    assert any(fragment.namespace == namespace and fragment.kind == kind for fragment in fragments)
    assert ann.fragments_for(bound) == fragments
    assert ann.fragments_for(raw) == fragments
