import sys

import pytest

import dryml.providers as providers


def test_identity_round_trip_and_stable_provider_id():
    identity = providers.ProviderIdentity("fake", "1", "providers.fake_provider", "Provider", ("b", "a", "a"), {"z": 1})
    round_trip = providers.ProviderIdentity.from_data(identity.to_data())

    assert round_trip == identity
    assert identity.capabilities == ("a", "b")
    assert identity.id.startswith("provider-v1-")
    assert identity.id == providers.ProviderIdentity.from_data(identity.to_data()).id


@pytest.mark.parametrize(
    "factory",
    [
        lambda: providers.ProviderIdentity("bad name"),
        lambda: providers.ProviderIdentity("ok", metadata={1: "bad"}),
        lambda: providers.ProviderRef("fake", "bad-module"),
        lambda: providers.ProviderRef("fake", "providers.fake_provider", "bad qual"),
    ],
)
def test_malformed_identity_and_ref_validation(factory):
    with pytest.raises(providers.ProviderError):
        factory()


def test_registry_order_duplicate_rejection_and_import_safety():
    sys.modules.pop("providers.fake_heavy_provider", None)
    sys.modules.pop("providers.fake_heavy_module", None)
    registry = providers.ProviderRegistry()
    registry.register_ref(providers.ProviderRef("z", "providers.fake_provider"))
    registry.register_ref(providers.ProviderRef("heavy", "providers.fake_heavy_provider"))

    assert [ref.name for ref in registry.list_refs()] == ["heavy", "z"]
    assert "providers.fake_heavy_module" not in sys.modules
    with pytest.raises(providers.ProviderRegistryError):
        registry.register_ref(providers.ProviderRef("z", "providers.fake_provider"))


def test_register_instance_and_unsupported_method():
    from providers.fake_provider import Provider

    provider = Provider()
    registry = providers.ProviderRegistry()
    identity = registry.register_instance(provider)

    assert identity.name == "fake"
    assert registry.get_instance("fake") is provider
    request = providers.RepresentationInspectionRequest()
    report = provider.inspect_representations(request)
    assert report.status == "unsupported"
    assert report.issues[0].code == "unsupported"
