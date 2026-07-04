import importlib
import sys


def test_annotation_decorator_imports_do_not_import_heavy_frameworks(monkeypatch):
    for name in ("torch", "tensorflow", "jax"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    import dryml.annotations
    import dryml.env
    import dryml.world
    import dryml.runtime

    importlib.reload(dryml.annotations)
    importlib.reload(dryml.env)
    importlib.reload(dryml.world)
    importlib.reload(dryml.runtime)

    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules
    assert "jax" not in sys.modules
