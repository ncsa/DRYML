from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ..definition import ConcreteDefinition
from .model import QueryCardinalityError
from .path import DefinitionPath


QueryTerminal = Literal["collect", "count", "exists", "one", "one_or_none", "page", "owners", "occurrences", "explain"]
ScanPolicyMode = Literal["allow", "warn", "forbid"]


@dataclass(frozen=True, slots=True)
class ScanPolicy:
    """Execution policy for query plans that require an unindexed scan.

    Args:
        mode: Whether to allow, warn about, or reject scan fallback.
        max_verify: Optional maximum number of Python CDef verifications.
    """

    mode: ScanPolicyMode = "allow"
    max_verify: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"allow", "warn", "forbid"}:
            raise ValueError("scan policy must be 'allow', 'warn', or 'forbid'.")
        if self.max_verify is not None and self.max_verify < 0:
            raise ValueError("max_verify must be non-negative or None.")


@dataclass(frozen=True, slots=True)
class PagedResultCursor:
    """Opaque keyset cursor for a Store-local lowered candidate relation."""

    source_key: str
    generation: int
    stable_hash: str
    collision_ordinal: int
    definition_id: int
    direction: str = "forward"


@dataclass(frozen=True, slots=True)
class CandidateRelation:
    """Backend-owned candidate relation contract exposed to federation.

    The relation is identified by source and generation, and pages in stable
    `(stable_hash, collision_ordinal, definition_id)` order. It carries no live
    backend connection or cursor state; concrete backends decide whether the
    relation is represented by SQL text, a CTE, or a temporary table.
    """

    source_key: str
    generation: int
    relation_id: str
    ordering: tuple[str, ...] = ("stable_hash", "collision_ordinal", "definition_id")
    supports_keyset: bool = True


@dataclass(frozen=True, slots=True)
class LoweredEdgeStep:
    """One SQL-side graph propagation step between selector graph nodes."""

    from_node: int
    to_node: int
    path: DefinitionPath
    direction: Literal["parent", "child"]
    unordered: bool = False


@dataclass(frozen=True, slots=True)
class LoweredGraphPlan:
    """Anchor-oriented graph relation plan emitted by lowering compilers."""

    anchor_node: int
    anchor_reason: str
    anchor_estimate: int | None
    propagation_steps: tuple[LoweredEdgeStep, ...]
    root_node: int


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    """CDef batch fetched from a lowered relation after a short read view."""

    ids: tuple[int, ...]
    cdefs: tuple[ConcreteDefinition, ...]
    next_cursor: PagedResultCursor | None = None


@dataclass(slots=True)
class LoweringDiagnostics:
    """Counters and plan facts emitted by lowered execution."""

    strategy: str = "fallback"
    anchor: str | None = None
    anchor_node: int | None = None
    anchor_reason: str | None = None
    anchor_relation_kind: str | None = None
    anchor_estimate: int | None = None
    anchor_fallback_reason: str | None = None
    propagation_steps: tuple[str, ...] = ()
    relation_strategy: str = "cte"
    estimated_rows: int | None = None
    sql_statements_executed: int = 0
    candidate_rows_read: int = 0
    cdef_blobs_decoded: int = 0
    python_verifications: int = 0
    relations_created: int = 0
    relations_dropped: int = 0
    temp_rows_inserted: int = 0
    terminal_stop_reason: str | None = None
    scan_required: bool = False
    scan_reason: str | None = None
    sqlite_plan: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "anchor": self.anchor,
            "anchor_node": self.anchor_node,
            "anchor_reason": self.anchor_reason,
            "anchor_relation_kind": self.anchor_relation_kind,
            "anchor_estimate": self.anchor_estimate,
            "anchor_fallback_reason": self.anchor_fallback_reason,
            "propagation_steps": self.propagation_steps,
            "relation_strategy": self.relation_strategy,
            "estimated_rows": self.estimated_rows,
            "sql_statements_executed": self.sql_statements_executed,
            "candidate_rows_read": self.candidate_rows_read,
            "cdef_blobs_decoded": self.cdef_blobs_decoded,
            "python_verifications": self.python_verifications,
            "relations_created": self.relations_created,
            "relations_dropped": self.relations_dropped,
            "temp_rows_inserted": self.temp_rows_inserted,
            "terminal_stop_reason": self.terminal_stop_reason,
            "scan_required": self.scan_required,
            "scan_reason": self.scan_reason,
            "sqlite_plan": self.sqlite_plan,
        }


@dataclass(frozen=True, slots=True)
class LoweredQueryPlan:
    """Backend-owned relation plan for candidate IDs.

    The SQL text is intentionally kept backend-local in practice; this dataclass
    carries it only between the SQLite compiler and SQLite read view.
    """

    source_key: str
    generation: int
    domain: str
    terminal: QueryTerminal
    candidate_sql: str
    params: tuple[Any, ...] = ()
    strategy: str = "sqlite-lowered"
    relation_id: str = "candidate_relation"
    ordering: tuple[str, ...] = ("stable_hash", "collision_ordinal", "definition_id")
    ordered: bool = True
    supports_keyset: bool = True
    estimated_size: int | None = None
    scan_required: bool = False
    scan_reason: str | None = None
    diagnostics: LoweringDiagnostics = field(default_factory=LoweringDiagnostics)

    def relation(self) -> CandidateRelation:
        """Return the backend-neutral candidate relation contract for this plan."""

        return CandidateRelation(
            source_key=self.source_key,
            generation=self.generation,
            relation_id=self.relation_id,
            ordering=self.ordering,
            supports_keyset=self.supports_keyset,
        )


class TerminalSink(Protocol):
    """Consumer for verified CDefs that controls terminal short-circuiting."""

    def accept(self, cdef: ConcreteDefinition, metadata: Any = None) -> bool:
        ...

    @property
    def done(self) -> bool:
        ...

    @property
    def stop_reason(self) -> str | None:
        ...

    def result(self):
        ...


class ExistsSink:
    """Terminal sink for `exists()`: stop at the first verified match."""

    def __init__(self) -> None:
        self._found = False

    def accept(self, cdef: ConcreteDefinition, metadata: Any = None) -> bool:
        self._found = True
        return False

    @property
    def done(self) -> bool:
        return self._found

    @property
    def stop_reason(self) -> str | None:
        return "first-match" if self._found else None

    def result(self) -> bool:
        return self._found


class CountSink:
    """Terminal sink for `count()`: retain only an integer count."""

    def __init__(self, *, stop_after: int | None = None) -> None:
        self._count = 0
        self._stop_after = stop_after

    def accept(self, cdef: ConcreteDefinition, metadata: Any = None) -> bool:
        self._count += 1
        return not self.done

    @property
    def done(self) -> bool:
        return self._stop_after is not None and self._count >= self._stop_after

    @property
    def stop_reason(self) -> str | None:
        return "count-limit" if self.done else None

    def result(self) -> int:
        return self._count


class CollectSink:
    """Terminal sink that collects verified CDefs in verification order."""

    def __init__(self, *, stop_after: int | None = None) -> None:
        self._items: list[ConcreteDefinition] = []
        self._stop_after = stop_after

    def accept(self, cdef: ConcreteDefinition, metadata: Any = None) -> bool:
        self._items.append(cdef)
        return not self.done

    @property
    def done(self) -> bool:
        return self._stop_after is not None and len(self._items) >= self._stop_after

    @property
    def stop_reason(self) -> str | None:
        return "collect-limit" if self.done else None

    def result(self) -> tuple[ConcreteDefinition, ...]:
        return tuple(self._items)


class OneSink:
    """Terminal sink for `one()`: stop after the second verified match."""

    def __init__(self, *, allow_none: bool = False) -> None:
        self._items: list[ConcreteDefinition] = []
        self._allow_none = allow_none

    def accept(self, cdef: ConcreteDefinition, metadata: Any = None) -> bool:
        self._items.append(cdef)
        return not self.done

    @property
    def done(self) -> bool:
        return len(self._items) >= 2

    @property
    def stop_reason(self) -> str | None:
        return "second-match" if self.done else None

    def result(self) -> ConcreteDefinition | None:
        if len(self._items) > 1:
            expected = "zero or one" if self._allow_none else "exactly one"
            raise QueryCardinalityError(f"Expected {expected} result, found {len(self._items)}.")
        if not self._items:
            if self._allow_none:
                return None
            raise QueryCardinalityError("Expected exactly one result, found 0.")
        return self._items[0]


class OneOrNoneSink(OneSink):
    """Terminal sink for `one_or_none()`: zero verified matches returns None."""

    def __init__(self) -> None:
        super().__init__(allow_none=True)


class PageSink:
    """Terminal sink for one verified page plus a next keyset cursor."""

    def __init__(self, page_size: int) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive.")
        self.page_size = page_size
        self._items: list[ConcreteDefinition] = []
        self._cursor: PagedResultCursor | None = None

    def accept(self, cdef: ConcreteDefinition, metadata: Any = None) -> bool:
        self._items.append(cdef)
        if isinstance(metadata, PagedResultCursor):
            self._cursor = metadata
        return not self.done

    @property
    def done(self) -> bool:
        return len(self._items) >= self.page_size

    @property
    def stop_reason(self) -> str | None:
        return "page-full" if self.done else None

    def result(self) -> tuple[tuple[ConcreteDefinition, ...], PagedResultCursor | None]:
        return tuple(self._items), self._cursor
