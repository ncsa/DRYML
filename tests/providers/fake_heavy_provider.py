"""Fake provider variant that imports a sentinel heavy module at module import."""

from providers import fake_heavy_module  # noqa: F401
from dryml.providers import ProviderIdentity
from providers.fake_provider import Provider as _BaseProvider


class Provider(_BaseProvider):
    identity = ProviderIdentity("heavy", "1", __name__, "Provider", ("operation_inspection",), {"fixture": True})
