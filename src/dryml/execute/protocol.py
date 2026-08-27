from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dryml.core.utils.general import pickle_load, pickle_save, pickler, unpickler


@dataclass(frozen=True, slots=True)
class StoreRef:
    kind: str
    uri: str

    @classmethod
    def directory(cls, uri) -> "StoreRef":
        return cls(kind="directory", uri=str(uri))

    def open(self):
        if self.kind != "directory":
            raise ValueError(f"Unsupported StoreRef kind {self.kind!r}.")
        from dryml.core.store.dir import DirStore
        return DirStore(self.uri)


@dataclass(slots=True)
class ExecutionRequest:
    fn_payload: bytes
    args_canonical: Any
    kwargs_canonical: Any
    transfer_store: StoreRef
    result_store: StoreRef
    context_reqs: dict[str, Any] = field(default_factory=dict)
    update_cdefs: tuple[Any, ...] = ()
    save_result_objects: bool = True

    @classmethod
    def build(
            cls,
            fn,
            args_canonical,
            kwargs_canonical,
            *,
            transfer_store: StoreRef,
            result_store: StoreRef,
            context_reqs=None,
            update_cdefs=(),
            save_result_objects: bool = True):
        return cls(
            fn_payload=pickler(fn),
            args_canonical=args_canonical,
            kwargs_canonical=kwargs_canonical,
            transfer_store=transfer_store,
            result_store=result_store,
            context_reqs=dict(context_reqs or {}),
            update_cdefs=tuple(update_cdefs or ()),
            save_result_objects=save_result_objects,
        )

    def load_fn(self):
        return unpickler(self.fn_payload)


@dataclass(slots=True)
class ExecutionResponse:
    ok: bool
    result_canonical: Any = None
    updated_cdefs: tuple[Any, ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None

    @classmethod
    def success(cls, result_canonical=None, updated_cdefs=()):
        return cls(
            ok=True,
            result_canonical=result_canonical,
            updated_cdefs=tuple(updated_cdefs or ()),
        )

    @classmethod
    def failure(cls, exc: BaseException, tb: str):
        return cls(
            ok=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=tb,
        )


class ExecutionError(RuntimeError):
    pass


class RemoteExecutionError(ExecutionError):
    def __init__(self, response: ExecutionResponse):
        msg = f"Remote execution failed with {response.error_type}: {response.error_message}"
        if response.traceback:
            msg = f"{msg}\n{response.traceback}"
        super().__init__(msg)
        self.response = response


def save_request(request: ExecutionRequest, path: str) -> None:
    pickle_save(request, path)


def load_request(path: str) -> ExecutionRequest:
    return pickle_load(path)


def save_response(response: ExecutionResponse, path: str) -> None:
    pickle_save(response, path)


def load_response(path: str) -> ExecutionResponse:
    return pickle_load(path)
