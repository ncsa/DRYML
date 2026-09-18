"""Public U6 Dispatch facade contracts."""

from __future__ import annotations

import inspect

import dryml.dispatch as dispatch


def test_public_dispatch_surface_and_workload_only_signatures() -> None:
    """Expose the U6 facade while leaving all invocation controls on views."""

    assert set(dispatch.__all__) == {
        "BackendChoice",
        "DispatchCoverageWarning",
        "DispatchError",
        "DispatchReport",
        "DispatchView",
        "InProcess",
        "ProbeOptions",
        "backends",
        "explain",
        "register_backend",
        "run",
        "set_execute_backend_default",
        "set_probe_default",
        "set_worker_environment_default",
        "set_worker_python_default",
        "set_worker_world_default",
        "submit",
        "unregister_backend",
        "with_options",
    }
    for entry in (dispatch.explain, dispatch.run, dispatch.submit):
        signature = inspect.signature(entry)
        assert tuple(signature.parameters) == ("fn", "args", "kwargs")
        assert (
            signature.parameters["fn"].kind
            is inspect.Parameter.POSITIONAL_ONLY
        )


def test_in_process_marker_is_fieldless_and_inert() -> None:
    """Keep explicit local selection separate from Execute configurations."""

    assert str(inspect.signature(dispatch.InProcess)) == "() -> None"
    assert dispatch.InProcess() == dispatch.InProcess()
