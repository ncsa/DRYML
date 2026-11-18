"""Local Dataset traversal failures; producer and cleanup errors remain causal."""


class TraversalError(RuntimeError):
    """Base for local traversal lifecycle failures; contains no saved state."""


class TraversalOwnershipError(TraversalError):
    """Reject competing views, recursive/foreign-thread pulls or stale tokens.

    The rejected operation does not advance accepted progress. Independent
    traversals over the same Dataset or prepared plan remain supported.
    """


class UnsupportedContinuationError(TraversalError):
    """Requested exact capture uses an unsupported source, value, or protocol.

    Live accepted progress and previously captured records remain valid.
    """


class ContinuationMismatchError(ValueError):
    """Well-formed continuation evidence disagrees with the supplied target."""


class ContinuationFormatError(ValueError):
    """Continuation schema, inventory, integrity, or protocol is invalid."""
