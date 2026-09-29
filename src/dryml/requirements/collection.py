"""Private adapters from passive annotations to domain requirement declarations."""

from __future__ import annotations

from typing import Any, TypeVar

from dryml.annotations import Annotation, annotations_for_method, attach_annotation, collect_annotations

from .errors import RequirementError
from .model import RequirementDeclaration

R = TypeVar("R")


def attach_declaration(target: Any, *, key: str, declaration: RequirementDeclaration[R]) -> Any:
    """Attach one exact domain declaration through the public annotation seam.

    Args:
        target: A live target supported by passive annotation attachment.
        key: The domain's exact owner-qualified annotation key.
        declaration: The exact shared declaration to attach.

    Returns:
        The exact supplied target.

    Raises:
        RequirementError: If input is malformed or static annotation attachment
            rejects the target. Underlying diagnostic details are not retained.

    Side Effects:
        Adds one process-local annotation to a supported target; no target call,
        descriptor binding, runtime, or session behavior occurs.
    """

    if type(declaration) is not RequirementDeclaration:
        raise RequirementError("invalid requirement declaration")
    try:
        return attach_annotation(target, Annotation(key, declaration))
    except Exception:
        raise RequirementError("requirement annotation attachment failed") from None


def collect_declarations(
    owner: Any,
    *,
    key: str,
    value_type: type[R],
    method_name: str | None = None,
) -> tuple[RequirementDeclaration[R], ...]:
    """Collect one domain key's exact declarations through static annotations.

    Args:
        owner: A direct target or class; instance owners are normalized to their
            exact type only when selecting ``method_name``.
        key: The exact owner-qualified annotation key to collect.
        value_type: The exact domain value type expected under ``key``.
        method_name: Optional exact method name for static selected-method
            collection.

    Returns:
        Ordered identity-deduplicated exact declarations selected by the passive
        annotation kernel.

    Raises:
        RequirementError: If key, type, annotations, selected entries, or method
            arguments are malformed. No partial tuple is returned.

    Side Effects:
        None. Collection does not bind descriptors, invoke instance lookup, or
        inspect instance state.
    """

    if type(value_type) is not type:
        raise RequirementError("invalid requirement value type")
    try:
        if method_name is None:
            annotations = collect_annotations(owner, key=key)
        else:
            cls = owner if issubclass(type(owner), type) else type(owner)
            annotations = annotations_for_method(cls, method_name, key=key)
    except Exception:
        raise RequirementError("requirement annotation collection failed") from None
    declarations: list[RequirementDeclaration[R]] = []
    for annotation in annotations:
        value = annotation.value
        if type(value) is not RequirementDeclaration or type(value.value) is not value_type:
            raise RequirementError("invalid requirement annotation declaration")
        declarations.append(value)
    return tuple(declarations)


__all__: list[str] = []
