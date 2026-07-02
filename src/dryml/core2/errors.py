from typing import Any
from dataclasses import dataclass

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
