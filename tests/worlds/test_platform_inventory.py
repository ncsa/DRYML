from dryml.worlds import local_inventory


def test_platform_inventory_uses_injected_posix_and_windows_seams(monkeypatch):
    import dryml.worlds.inventory as module

    monkeypatch.setattr(module, "_platform_memory", lambda *_args: (None, "unknown"))
    monkeypatch.setattr(module.sys, "platform", "win32")
    inventory = local_inventory(environ={}, device_root=None)
    assert inventory.memory is None
