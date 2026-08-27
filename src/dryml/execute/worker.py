from __future__ import annotations

import argparse
import traceback

from dryml.core import Repo
from dryml.core.canonical import to_canonical
from dryml.context import use_context

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
        if request.context_reqs:
            with use_context(request.context_reqs):
                return run_call()
        return run_call()
    except BaseException as exc:
        return ExecutionResponse.failure(exc, traceback.format_exc())


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
