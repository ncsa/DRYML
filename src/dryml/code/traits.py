from dryml.core.backend import Backend
from dryml.core.tensor_spec import BatchMode
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Traits:
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
        if not isinstance(rhs, Traits):
            raise NotImplementedError(f"Trait matching not implemented against type {type(rhs)}")

        if strict:
            return (
                self.backend == rhs.backend and
                self.batch_mode == rhs.batch_mode )

        if self.backend is not None and rhs.backend is not None and self.backend != rhs.backend:
            return False
        if self.batch_mode is not None and rhs.batch_mode is not None and self.batch_mode != rhs.batch_mode:
            return False
        return True

    @property
    def specificity(self) -> int:
        return int(self.backend is not None) + int(self.batch_mode is not None)
