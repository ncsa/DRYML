"""Runtime-default annotation sugar with no runtime activation behavior."""

from __future__ import annotations

from typing import Any

from .decorators import default as declare_default
from .namespaces import RUNTIME


def default(**kwargs: Any):
    """Declare a partial runtime default without mutating runtime state.

    Args:
        **kwargs: Runtime mapping fields plus optional annotation ``source``,
            ``priority``, and ``merge_policy`` metadata.

    Returns:
        An identity-preserving declaration decorator.
    """

    source = kwargs.pop("source", None)
    priority = kwargs.pop("priority", 0)
    merge_policy = kwargs.pop("merge_policy", None)
    return declare_default(namespace=RUNTIME, fragment=kwargs, source=source, priority=priority, merge_policy=merge_policy)


__all__ = ["default"]
