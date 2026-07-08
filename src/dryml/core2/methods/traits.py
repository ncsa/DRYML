from dataclasses import dataclass

from dryml.core2.backend import Backend
from dryml.core2.tensor_spec import BatchMode


@dataclass(frozen=True, slots=True)
class Traits:
    """Backend and batch-mode selector for DRYML Method implementations.

    Args:
        backend: Backend value, backend name, or ``None`` as a wildcard.
        batch_mode: Batch mode value, batch mode name, or ``None`` as a wildcard.
    """

    backend: Backend | str | None = None
    batch_mode: BatchMode | str | None = None

    def __post_init__(self):
        if self.backend is not None and not isinstance(self.backend, Backend):
            object.__setattr__(self, "backend", Backend(self.backend))
        if self.batch_mode is not None and not isinstance(self.batch_mode, BatchMode):
            object.__setattr__(self, "batch_mode", BatchMode(self.batch_mode))

    def __hash__(self):
        return hash((self.backend, self.batch_mode))

    def match(self, rhs, strict=False):
        """Return whether this trait selector matches *rhs*.

        Args:
            rhs: Other :class:`Traits` object to compare against.
            strict: If true, require exact equality. Otherwise ``None`` fields
                act as wildcards.
        """

        if not isinstance(rhs, Traits):
            raise NotImplementedError(f"Trait matching not implemented against type {type(rhs)}")

        if strict:
            return (
                self.backend is rhs.backend and
                self.batch_mode is rhs.batch_mode)

        if self.backend is not None and rhs.backend is not None and self.backend != rhs.backend:
            return False
        if self.batch_mode is not None and rhs.batch_mode is not None and self.batch_mode != rhs.batch_mode:
            return False
        return True

    @property
    def specificity(self) -> int:
        """Return how many selector fields are explicitly constrained."""

        return int(self.backend is not None) + int(self.batch_mode is not None)


__all__ = ["BatchMode", "Traits"]
