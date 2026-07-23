"""Import-safe namespace for the retired Ray Tune adapters.

The Tune 1 and Tune 2 adapters depended on the removed ``Trainable`` and
``TrainSpec`` APIs and have no supported equivalent. Use ``dryml.SearchSpace``
with ``dryml.models.Experiment`` for bounded local search.
"""


_UNSUPPORTED_MESSAGE = (
    "the legacy dryml.ray.tune adapters are no longer supported; use "
    "dryml.SearchSpace with dryml.models.Experiment for bounded local search"
)


class _UnsupportedTuneAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_UNSUPPORTED_MESSAGE)


class Tune1ObjectSaver(_UnsupportedTuneAdapter):
    """Retired Ray Tune 1 checkpoint adapter."""


class Tune1Trainer(_UnsupportedTuneAdapter):
    """Retired Ray Tune 1 trainer adapter."""


class Tune2ObjectSaver(_UnsupportedTuneAdapter):
    """Retired Ray Tune 2 checkpoint adapter."""


class Tune2Trainer(_UnsupportedTuneAdapter):
    """Retired Ray Tune 2 trainer adapter."""


__all__ = [
    "Tune1ObjectSaver",
    "Tune1Trainer",
    "Tune2ObjectSaver",
    "Tune2Trainer",
]
