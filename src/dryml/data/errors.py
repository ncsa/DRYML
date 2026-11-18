"""Local Dataset traversal failures; producer and cleanup errors remain causal."""


class TraversalError(RuntimeError):
    """Base for local traversal lifecycle failures; contains no saved state."""


class TraversalOwnershipError(TraversalError):
    """Reject competing views, recursive/foreign-thread pulls or stale tokens.

    The rejected operation does not advance accepted progress. Independent
    traversals over the same Dataset or prepared plan remain supported.
    """
