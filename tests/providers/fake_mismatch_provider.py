"""Fake provider whose identity intentionally mismatches its ref."""

from dryml.providers import ProviderIdentity
from providers.fake_provider import Provider as _BaseProvider


class Provider(_BaseProvider):
    identity = ProviderIdentity("actual", "2", __name__, "Provider")
