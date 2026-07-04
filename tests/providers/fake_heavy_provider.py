"""Fake provider variant that imports a sentinel heavy module at module import."""

from providers import fake_heavy_module  # noqa: F401
from providers.fake_provider import Provider as _BaseProvider


class Provider(_BaseProvider):
    pass
