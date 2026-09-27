from typing import Any
from dataclasses import dataclass


class TemplateError(ValueError):
    """Report an invalid template expression, binding, or domain operation.

    Args:
        reason: Stable explanation of the rejected operation.
        path: Optional graph path at which the failure occurred.
        root: Optional fully-qualified template binding root involved in the
            failure.

    Attributes:
        reason: The supplied failure explanation.
        path: The optional structural failure location.
        root: The optional binding root.
    """

    def __init__(self, reason: str, *, path: object | None = None, root: str | None = None) -> None:
        self.reason = reason
        self.path = path
        self.root = root
        super().__init__(reason)


class UnresolvedTemplateError(TemplateError):
    """Report an operation that requires a template with no active expressions."""


class TemplateLimitError(TemplateError):
    """Report a template operation that exceeds a declared resource limit."""


class UnsupportedTemplateVerificationError(TemplateError):
    """Report an unsupported exact verification operation for a template."""

# -----------------------------
# Errors
# -----------------------------

@dataclass(frozen=True)
class ConcretizeError(TypeError):
    path: tuple[str|int, ...]
    value: Any
    msg: str = "Unsupported value for concretization"

    def __str__(self) -> str:
        p = "/".join(self.path) if self.path else "<root>"
        return f"{self.msg} at {p}: {type(self.value).__name__} -> {self.value!r}"


@dataclass(frozen=True)
class CycleError(ValueError):
    msg: str = ""
    def __str__(self) -> str:
        if self.msg != "":
            return f"Cycle detected: {self.msg}"
        else:
            return "Cycle detected"


@dataclass(frozen=True)
class PathAccessError(KeyError):
    path: tuple[str|int,...]
    def __str__(self) -> str:
        p = "/".join(self.path) if self.path else "<root>"
        return f"Path Access error at {p}"


class CannotConcretizeParameterizedDefinition(TypeError):
    def __init__(self, path, value, msg: str = "Cannot concretize unresolved Par"):
        self.path = tuple(path)
        self.value = value
        self.msg = msg
        super().__init__(str(self))

    def __str__(self) -> str:
        p = "/".join(map(str, self.path)) if self.path else "<root>"
        return f"{self.msg} at {p}: {type(self.value).__name__} -> {self.value!r}"


class CannotConcretizeSelectorReference(TypeError):
    def __init__(self, path, value, msg: str = "Cannot concretize Ref(Selector)"):
        self.path = tuple(path)
        self.value = value
        self.msg = msg
        super().__init__(str(self))

    def __str__(self) -> str:
        p = "/".join(map(str, self.path)) if self.path else "<root>"
        return f"{self.msg} at {p}: selector references are query-only; store SelectorSpec/QuotedDef for data values"
