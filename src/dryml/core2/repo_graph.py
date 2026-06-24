from __future__ import annotations

from typing import Any

from .definition import ConcreteDefinition
from .object import Object


RevisionMapType = dict[ConcreteDefinition, str]


def manage_revision(obj: Any, revision: RevisionMapType | str | None):
    if revision is None:
        return {}
    if isinstance(revision, str):
        if isinstance(obj, Object):
            return {obj.definition: revision}
        if isinstance(obj, ConcreteDefinition):
            return {obj: revision}
        raise ValueError(
            "When revision is a string, manage_revision must get a clear object or definition to create the revision dictionary."
        )
    return revision
