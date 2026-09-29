"""Exception types raised by :mod:`dryml.environments`."""


class DrymlEnvironmentError(Exception):
    """Base class for environment metadata, probing, and compatibility errors.

    Parameters
    ----------
    message:
        Human-readable error message.
    context:
        Optional structured details useful to tests, logs, or user interfaces.
    """

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context = dict(context or {})


class EnvironmentRequirementError(DrymlEnvironmentError):
    """Raised when an environment requirement is invalid or cannot be composed."""


class EnvironmentProbeError(DrymlEnvironmentError):
    """Raised when a failed probe result is required as a successful record."""


class EnvironmentSpecError(DrymlEnvironmentError):
    """Raised when an environment spec is malformed or unsupported."""


class EnvironmentCompatibilityError(DrymlEnvironmentError):
    """Raised when a compatibility report is incompatible."""


class EnvironmentSerializationError(DrymlEnvironmentError):
    """Raised when environment metadata cannot be serialized or decoded."""


class EnvironmentRegistryError(DrymlEnvironmentError):
    """Raised for duplicate or missing environment registry entries."""


class EnvironmentFeatureUnavailable(DrymlEnvironmentError):
    """Raised when an optional feature needed by the environment layer is absent."""


__all__ = [
    "DrymlEnvironmentError",
    "EnvironmentRequirementError",
    "EnvironmentProbeError",
    "EnvironmentSpecError",
    "EnvironmentCompatibilityError",
    "EnvironmentSerializationError",
    "EnvironmentRegistryError",
    "EnvironmentFeatureUnavailable",
]
