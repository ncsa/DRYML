"""Use non-invoking analysis and explicit trusted current-process tracing.

``trace`` executes trusted code once. It is neither a sandbox nor a hard-timeout
boundary, so do not trace untrusted code.
"""

from __future__ import annotations

import dryml.code as code
from dryml.core import Definition


class TraceableModel:
    """A proxy-only receiver whose real method must never run during tracing."""

    def observe(self) -> None:
        """Represent one traceable method observation."""

        raise AssertionError("trace records this method without invoking it")


analysis_counter = 0


def analyzed_target() -> None:
    """Increment only if a caller actually invokes this target."""

    global analysis_counter
    analysis_counter += 1


def traced_target(model: TraceableModel) -> None:
    """Make one supported Definition-proxy method call."""

    model.observe()


def main() -> None:
    """Prove analysis is non-invoking and trace is explicit and bounded."""

    analysis = code.analyze(analyzed_target, algorithms=("source", "callables"))
    assert analysis.ok
    assert analysis_counter == 0

    traced = code.trace(
        traced_target,
        args=(Definition(TraceableModel),),
        context=code.CodeAnalysisContext(allow_dynamic_execution=True),
    )
    calls = traced.facts_of_kind("dynamic_call")
    assert traced.ok
    assert len(calls) == 1
    assert calls[0].data["method_name"] == "observe"


if __name__ == "__main__":
    main()
