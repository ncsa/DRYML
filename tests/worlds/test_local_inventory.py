import sys

from dryml.worlds import LocalResourceInventory, local_inventory


def test_injected_inventory_is_authoritative_and_visibility_excludes_available_memory():
    first = LocalResourceInventory((2, 1), {"gpu": ("a",)}, memory=8, accelerator_memory={"gpu": {"a": 4}}, metadata={"available_memory": 1})
    second = LocalResourceInventory((1, 2), {"gpu": ("a",)}, memory=8, accelerator_memory={"gpu": {"a": 4}}, metadata={"available_memory": 2})
    assert first.visibility_identity == second.visibility_identity


def test_lightweight_inventory_never_imports_frameworks():
    frameworks = {"tensorflow", "torch", "jax", "jaxlib"}
    before = frameworks & set(sys.modules)
    local_inventory(environ={}, device_root=None)
    assert frameworks & set(sys.modules) == before
