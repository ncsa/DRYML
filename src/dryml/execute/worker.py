from __future__ import annotations

import argparse
import traceback

from dryml.core2 import Repo
from dryml.core2.canonical import to_canonical
from dryml.runtime import RuntimeAllocationView, RuntimeMode, enter_runtime

from .protocol import (
    ExecutionResponse,
    load_request,
    save_response,
)


def execute_request(request) -> ExecutionResponse:
    transfer_store = request.transfer_store.open()
    result_store = request.result_store.open()
    repo = Repo(stores=[transfer_store, result_store])

    def run_call():
        fn = request.load_fn()
        args = repo.load_object(
            request.args_canonical,
            restore_state=True,
            build_missing=True,
        )
        kwargs = repo.load_object(
            request.kwargs_canonical,
            restore_state=True,
            build_missing=True,
        )

        result = fn(*args, **kwargs)
        result_canonical = to_canonical(result, repo=repo)
        if request.save_result_objects:
            repo.save_object(result_canonical, store=result_store)

        updated = []
        for cdef in request.update_cdefs:
            obj = repo.get_cached(cdef)
            if obj is None:
                obj = repo.load_object(cdef, restore_state=True, build_missing=False)
            repo.save_object(obj, store=result_store)
            updated.append(cdef)

        repo.flush()
        return ExecutionResponse.success(
            result_canonical=result_canonical,
            updated_cdefs=updated,
        )

    try:
        with enter_runtime(RuntimeMode.WORKER, _allocation_from_legacy_context_reqs(request.context_reqs)):
            return run_call()
    except BaseException as exc:
        return ExecutionResponse.failure(exc, traceback.format_exc())


def _allocation_from_legacy_context_reqs(context_reqs) -> RuntimeAllocationView:
    """Translate retained execute context requirements to a runtime allocation.

    This is a compatibility bridge only; dispatch v2 should pass a real
    WorldAllocation-derived RuntimeAllocationView.
    """

    cpus = 0
    gpus = 0
    for spec in (context_reqs or {}).values():
        if isinstance(spec, dict):
            cpus = max(cpus, int(spec.get("num_cpus", 0) or 0))
            gpus = max(gpus, int(spec.get("num_gpus", 0) or 0))
    return RuntimeAllocationView(
        role="legacy_execute_worker",
        replica=0,
        rank=0,
        local_rank=0,
        cpus=tuple(range(cpus)),
        accelerators={"gpu": tuple(range(gpus))} if gpus else {},
        metadata={"source": "dryml.execute legacy context_reqs"},
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a DRYML execution request.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)

    request = load_request(args.request)
    response = execute_request(request)
    save_response(response, args.response)
    return 0 if response.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
