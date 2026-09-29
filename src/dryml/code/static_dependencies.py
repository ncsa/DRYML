"""Bounded provider-neutral static dependency resolution kernel."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .inspection import InspectionTarget, capture_inspection
from .kernels import AnalysisKernel, KernelContext


@dataclass(frozen=True, slots=True)
class StaticDependencies:
    """Immutable bounded static resolution result for one analysis invocation.

    Args:
        targets: Resolver-order canonical live or snapshot-backed targets.
        complete: Whether each target and edge was proven within bounds.
        diagnostics: Bounded framework-authored unresolved-coverage categories.

    Raises:
        ValueError: If result fields are malformed or diagnostics are not
            stable framework categories.

    Side Effects:
        None. Live target handles remain request-local when this result is used
        inline and the result is not a transport format.
    """

    targets: tuple[object, ...]
    complete: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate immutable result shape without calling target protocols."""

        from .targets import CodeTarget

        if type(self.targets) is not tuple or any(
                type(target) not in (CodeTarget, InspectionTarget)
                for target in self.targets):
            raise ValueError("static dependencies targets are invalid")
        if (type(self.complete) is not bool
                or type(self.diagnostics) is not tuple
                or any(type(item) is not str for item in self.diagnostics)):
            raise ValueError("static dependencies are invalid")


class StaticDependenciesKernel(AnalysisKernel[None, StaticDependencies]):
    """Resolve bounded static call targets from the canonical scheduler target.

    Args:
        max_targets: Maximum targets, including the root, returned by one run.
        max_depth: Maximum root-relative static call depth.

    Raises:
        ValueError: If either bound is not a positive exact integer.

    Side Effects:
        For a live target, capture reads passive source and binding facts. It
        never invokes target code or imports product/domain packages.
    """

    input_type = type(None)
    output_type = StaticDependencies

    def __init__(self, *, max_targets: int = 256, max_depth: int = 32) -> None:
        """Create a bounded static resolver kernel."""

        if (type(max_targets) is not int or max_targets < 1
                or type(max_depth) is not int or max_depth < 0):
            raise ValueError("static dependency limits are invalid")
        self.max_targets = max_targets
        self.max_depth = max_depth

    def run(self, graph: object, value: None,
            context: KernelContext) -> StaticDependencies:
        """Resolve call facts from a canonical live or detached context target.

        Args:
            graph: Immutable graph supplied by the scheduler, not independently
                used as a target source.
            value: Required ``None`` input.
            context: Canonical invocation context with the admitted target.

        Returns:
            Deterministic root-first targets and honest coverage state.

        Side Effects:
            Live targets are captured once for this request. Snapshot targets
            consume only their validated detached facts.
        """

        target = context.target
        capture = (None if type(target) is InspectionTarget else
                   capture_inspection(target))
        snapshot_target = (target if type(target) is InspectionTarget
                           else capture.target)
        snapshot = snapshot_target.snapshot
        records = {
            record.target_id: record for record in snapshot.records
        }
        local_targets = (
            capture._target_index
            if capture is not None else {}
        )
        selected: list[str] = []
        pending = deque([(snapshot_target.target_id, 0)])
        seen: set[str] = set()
        diagnostics: list[str] = []
        complete = True
        while pending:
            target_id, depth = pending.popleft()
            if target_id in seen:
                continue
            if len(selected) >= self.max_targets:
                complete = False
                diagnostics.append("static.target_limit")
                break
            seen.add(target_id)
            selected.append(target_id)
            record = records[target_id]
            if record.incomplete:
                complete = False
                diagnostics.append("static.unresolved")
            if depth >= self.max_depth:
                if record.calls:
                    complete = False
                    diagnostics.append("static.depth_limit")
                continue
            for call in record.calls:
                if call.target_id is None:
                    complete = False
                    diagnostics.append("static.unresolved")
                elif call.target_id not in seen:
                    pending.append((call.target_id, depth + 1))
        targets = tuple(
            target if target_id == snapshot_target.target_id else
            local_targets.get(target_id, InspectionTarget(snapshot, target_id))
            for target_id in selected
        )
        return StaticDependencies(targets, complete,
                                  tuple(dict.fromkeys(diagnostics)))


__all__ = ["StaticDependencies", "StaticDependenciesKernel"]
