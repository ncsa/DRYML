from __future__ import annotations

from typing import Any, Iterable, Mapping

from .backend import BackendBase, InlineBackend, LocalProcessBackend
from .transfer import prepare_call


class ExecutionOrchestrator:
    def __init__(self, backends: Iterable[BackendBase] | None = None):
        if backends is None:
            backends = (InlineBackend(), LocalProcessBackend())
        self.backends = {backend.name: backend for backend in backends}

    def backend(self, name: str | BackendBase | None) -> BackendBase:
        if isinstance(name, BackendBase):
            return name
        if name is None:
            name = "process"
        try:
            return self.backends[name]
        except KeyError as exc:
            raise ValueError(f"Unknown execution backend {name!r}.") from exc

    def submit(
            self,
            fn,
            *args,
            backend: str | BackendBase | None = None,
            repo=None,
            transfer_store=None,
            result_store=None,
            requirements: Mapping[str, Any] | None = None,
            update=None,
            env: Mapping[str, str] | None = None,
            **kwargs):
        from .future import OrchestratedFuture
        from .protocol import UnsupportedReferenceTransportError

        if update not in (None, False):
            raise UnsupportedReferenceTransportError(
                "Execution update transport is retired because it mutates current state; "
                "publish and load an exact StateRef instead."
            )

        prepared = prepare_call(
            args,
            kwargs,
            repo=repo,
            transfer_store=transfer_store,
            result_store=result_store,
        )
        from .protocol import ExecutionRequest
        request = ExecutionRequest.build(
            fn,
            prepared.args_canonical,
            prepared.kwargs_canonical,
            transfer_store=prepared.transfer_store,
            result_store=prepared.result_store,
            context_reqs=requirements,
            update_cdefs=(),
        )

        backend_obj = self.backend(backend)
        backend_future = backend_obj.submit(request, env=env)
        return OrchestratedFuture(
            backend_future=backend_future,
            prepared=prepared,
            update_targets=update,
            repo=repo,
        )

    def run(self, fn, *args, **kwargs):
        return self.submit(fn, *args, **kwargs).result()


_DEFAULT_ORCHESTRATOR = ExecutionOrchestrator()


def submit(fn, *args, **kwargs):
    return _DEFAULT_ORCHESTRATOR.submit(fn, *args, **kwargs)


def run(fn, *args, **kwargs):
    return _DEFAULT_ORCHESTRATOR.run(fn, *args, **kwargs)
