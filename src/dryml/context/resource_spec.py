from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
import re

_RESOURCE_KEY_RE = re.compile(r"^(cpu|gpu)/(\d+)$")


class InvalidResourceSpecError(ValueError):
    pass


def _as_nonneg_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidResourceSpecError(f"{name} must be an int >= 0")
    if value < 0:
        raise InvalidResourceSpecError(f"{name} must be >= 0")
    return value


def _as_fraction(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidResourceSpecError(f"{name} must be a number in (0, 1]")
    value = float(value)
    if not (0.0 < value <= 1.0):
        raise InvalidResourceSpecError(f"{name} must be in (0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    num_cpus: int = 0
    num_gpus: int = 0
    memory_bytes: int | None = None
    specific: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_user(cls, value: Mapping[str, Any] | ResourceSpec | None) -> ResourceSpec:
        if value is None:
            return cls()
        if isinstance(value, ResourceSpec):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "Per-context resource spec must be a mapping, ResourceSpec, or None"
            )

        raw = dict(value)

        num_cpus = _as_nonneg_int("num_cpus", raw.pop("num_cpus", 0))
        num_gpus = _as_nonneg_int("num_gpus", raw.pop("num_gpus", 0))

        memory_bytes = raw.pop("memory_bytes", None)
        if memory_bytes is not None:
            memory_bytes = _as_nonneg_int("memory_bytes", int(memory_bytes))

        specific = {}
        for key, req in raw.items():
            if not _RESOURCE_KEY_RE.match(key):
                raise InvalidResourceSpecError(f"Invalid resource key {key!r}")
            specific[key] = _as_fraction(key, req)

        return cls(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory_bytes=memory_bytes,
            specific=specific,
        )


@dataclass(slots=True)
class ResourceAllocation:
    assigned: dict[str, float] = field(default_factory=dict)
    memory_bytes: int | None = None

    def add(self, key: str, value: float) -> None:
        if not _RESOURCE_KEY_RE.match(key):
            raise InvalidResourceSpecError(f"Invalid resource key {key!r}")
        self.assigned[key] = _as_fraction(key, value)

    def __iter__(self):
        return iter(self.assigned)

    def __getitem__(self, key: str) -> float:
        return self.assigned[key]

    @property
    def cpus(self) -> list[str]:
        return sorted(k for k in self.assigned if k.startswith("cpu/"))

    @property
    def gpus(self) -> list[str]:
        return sorted(k for k in self.assigned if k.startswith("gpu/"))

    @property
    def cpu_ids(self) -> list[int]:
        return [int(k.split("/")[1]) for k in self.cpus]

    @property
    def gpu_ids(self) -> list[int]:
        return [int(k.split("/")[1]) for k in self.gpus]

    @property
    def num_cpus(self) -> int:
        return len(self.cpus)

    @property
    def num_gpus(self) -> int:
        return len(self.gpus)

    def satisfies(self, spec: Mapping[str, Any] | ResourceSpec | None) -> bool:
        spec = ResourceSpec.from_user(spec)

        if spec.memory_bytes is not None:
            if self.memory_bytes is None or self.memory_bytes < spec.memory_bytes:
                return False

        for key, need in spec.specific.items():
            if self.assigned.get(key, 0.0) < need:
                return False

        available_full_cpus = [
            key for key in self.cpus
            if key not in spec.specific and self.assigned[key] >= 1.0
        ]
        available_full_gpus = [
            key for key in self.gpus
            if key not in spec.specific and self.assigned[key] >= 1.0
        ]

        if len(available_full_cpus) < spec.num_cpus:
            return False
        if len(available_full_gpus) < spec.num_gpus:
            return False

        return True

    def __repr__(self) -> str:
        return f"assigned: {self.assigned} memory_bytes: {self.memory_bytes}"


def combine_resource_specs(
    specs: Iterable[Mapping[str, Any] | ResourceSpec | None],
) -> ResourceSpec:
    specs = [ResourceSpec.from_user(spec) for spec in specs]
    if not specs:
        return ResourceSpec()

    specific_keys = set()
    for spec in specs:
        specific_keys.update(spec.specific.keys())

    return ResourceSpec(
        num_cpus=max(spec.num_cpus for spec in specs),
        num_gpus=max(spec.num_gpus for spec in specs),
        memory_bytes=max(
            (spec.memory_bytes for spec in specs if spec.memory_bytes is not None),
            default=None,
        ),
        specific={
            key: max(spec.specific.get(key, 0.0) for spec in specs)
            for key in specific_keys
        },
    )


def normalize_compute_reqs(reqs) -> dict[str, ResourceSpec]:
    """
    Normalize top-level compute requirements.

    Supported forms:
        None
        'tf'
        ['tf', 'plain']
        {'tf': None}
        {'tf': {}}
        {'tf': {'num_gpus': 1}}
        {'plain': ResourceSpec(num_cpus=4)}
    """
    if reqs is None:
        return {}

    # shorthand: __compute_reqs__ = 'tf'
    if isinstance(reqs, str):
        return {reqs: ResourceSpec()}

    # full mapping form
    if isinstance(reqs, Mapping):
        out = {}
        for ctx_name, spec in reqs.items():
            if not isinstance(ctx_name, str):
                raise TypeError(f"Context name must be str, got {type(ctx_name)}")
            out[ctx_name] = ResourceSpec.from_user(spec)
        return out

    # shorthand: __compute_reqs__ = ['tf', 'plain']
    if isinstance(reqs, Iterable):
        out = {}
        for ctx_name in reqs:
            if not isinstance(ctx_name, str):
                raise TypeError(
                    "Iterable compute requirements must contain only context names"
                )
            out[ctx_name] = ResourceSpec()
        return out

    raise TypeError(
        "Compute requirements must be None, a context-name string, "
        "an iterable of context-name strings, or a mapping from context name to spec"
    )


def combine_compute_reqs(*req_groups) -> dict[str, ResourceSpec]:
    gathered: dict[str, list[ResourceSpec]] = {}
    for req_group in req_groups:
        for ctx_name, spec in normalize_compute_reqs(req_group).items():
            gathered.setdefault(ctx_name, []).append(spec)

    return {
        ctx_name: combine_resource_specs(specs)
        for ctx_name, specs in gathered.items()
    }
