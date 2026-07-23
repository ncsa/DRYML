import sys
import os


def test_importing_providers_and_registering_ref_do_not_import_heavy_module():
    sys.modules.pop("providers.fake_heavy_provider", None)
    sys.modules.pop("providers.fake_heavy_module", None)
    import dryml.providers as providers

    registry = providers.ProviderRegistry()
    registry.register_ref(providers.ProviderRef("heavy", "providers.fake_heavy_provider"))

    assert "providers.fake_heavy_module" not in sys.modules


def test_subprocess_probe_imports_heavy_module_only_in_child(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.path.abspath("tests"))
    sys.modules.pop("providers.fake_heavy_provider", None)
    sys.modules.pop("providers.fake_heavy_module", None)
    import dryml.environments as envs
    import dryml.operations as ops
    import dryml.providers as providers

    registry = providers.ProviderRegistry()
    registry.register_ref(providers.ProviderRef("heavy", "providers.fake_heavy_provider"))

    report = providers.probe_operation(ops.make_function_call_spec("providers.fake_provider:target_fn"), environment=envs.CurrentEnvironmentSpec(), providers=("heavy",), registry=registry, timeout=30)

    assert report.status == "ok"
    assert report.reports[0].metadata["heavy_imported"] is True
    assert "providers.fake_heavy_module" not in sys.modules
