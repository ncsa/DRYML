import importlib

import pytest


LEGACY_MODULES = ("dryml." + "context", "dryml." + "execute")


@pytest.mark.parametrize("module", LEGACY_MODULES)
def test_removed_legacy_modules_are_not_importable(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_removed_legacy_modules_are_not_top_level_exports():
    import dryml

    assert "context" not in dryml.__all__
    assert "execute" not in dryml.__all__
    assert not hasattr(dryml, "context")
    assert not hasattr(dryml, "execute")
