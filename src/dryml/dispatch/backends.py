"""Local subprocess backend for ``dryml.dispatch``."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dryml.environments import CondaEnvironmentSpec, CurrentEnvironmentSpec, PythonExecutableSpec, spec_from_data
from dryml.records import ExecutionCancellationInfo, ExecutionErrorInfo, ExecutionLogRef, ExecutionRecord, ProductWriteSession, StorageRef, write_execution_record

from .errors import DispatchCancelled, DispatchLaunchError, DispatchTimeout, WorkerProtocolError
from .protocol import WorkerHandshakeResponse, WorkerResponse, read_json_file, save_envelope, write_json_file


BACKEND_IDENTITY = {"name": "dryml.local_subprocess", "kind": "local_subprocess", "version": "1"}


@dataclass(slots=True)
class LocalSubprocessFuture:
    """Future representing one local subprocess dispatch worker."""

    process: subprocess.Popen
    plan: Any
    work_dir: str
    request_path: str
    handshake_path: str
    response_path: str
    stdout_path: str
    stderr_path: str
    preserve_work_dir: bool = False
    cancel_grace: float = 0.5
    handshake_timeout: float = 10.0

    _response: WorkerResponse | None = None
    _exception: BaseException | None = None
    _cancelled: bool = False
    _handshake: WorkerHandshakeResponse | None = None

    def done(self) -> bool:
        """Return whether the worker process has exited."""

        return self.process.poll() is not None

    def result(self, timeout: float | None = None) -> WorkerResponse:
        """Wait for the worker and return its compact response."""

        try:
            self.wait_for_handshake(timeout=self.handshake_timeout)
            self.process.wait(timeout=timeout)
            self._read_response()
            if self._exception is not None:
                raise self._exception
            assert self._response is not None
            self._persist_logs(self._response.execution_record_id)
            return self._response
        except subprocess.TimeoutExpired as exc:
            self.cancel(reason="timeout", record=False)
            self._response = self._parent_failure_response("timeout", error={"type": "TimeoutError", "message": "dispatch timed out"})
            raise DispatchTimeout("dispatch timed out") from exc
        except KeyboardInterrupt:
            self.cancel(reason="KeyboardInterrupt")
            raise
        except BaseException:
            self.cancel(reason="worker_protocol_error")
            raise
        finally:
            self._cleanup()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        """Return the exception raised by ``result()``, if any."""

        try:
            self.result(timeout=timeout)
        except BaseException as exc:
            return exc
        return None

    def wait_for_handshake(self, *, timeout: float | None = None) -> WorkerHandshakeResponse | None:
        """Wait for and validate the worker handshake phase."""

        if self._handshake is not None:
            return self._handshake
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if os.path.exists(self.handshake_path):
                self._handshake = WorkerHandshakeResponse.from_json(read_json_file(self.handshake_path))
                _report("dryml.dispatch.worker.handshake", "Checking worker handshake", operation_id=self.plan.envelope.operation_id, data={"status": self._handshake.status, "pid": self._handshake.pid})
                return self._handshake
            if self.done():
                return None
            if deadline is not None and time.monotonic() >= deadline:
                self.cancel(reason="handshake_timeout", record=False)
                self._response = self._parent_failure_response("failed", error={"type": "WorkerHandshakeError", "message": "worker handshake timed out"})
                return None
            time.sleep(0.01)

    def cancel(self, *, grace: float | None = None, reason: str = "user", record: bool = True) -> bool:
        """Cancel the process using SIGINT, SIGTERM, then SIGKILL where needed."""

        if self.done():
            # The leader may have exited while descendants still retain its
            # dedicated process group. Kill the group before reporting done.
            self.kill()
            return False
        self._cancelled = True
        _report("dryml.dispatch.worker.cancel", "Cancelling local subprocess worker", operation_id=self.plan.envelope.operation_id, data={"pid": self.process.pid, "reason": reason})
        wait = self.cancel_grace if grace is None else grace
        methods: list[str] = []
        for sig_name, sig_value in (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM)):
            methods.append(sig_name)
            self._signal(sig_value)
            if self._wait(wait):
                if record:
                    self._response = self._parent_failure_response("cancelled", cancellation={"requested": True, "method": sig_name, "escalated": len(methods) > 1, "reason": reason})
                return True
        methods.append("SIGKILL")
        self.kill()
        self._wait(wait)
        if record:
            self._response = self._parent_failure_response("cancelled", cancellation={"requested": True, "method": "SIGKILL", "escalated": True, "reason": reason})
        return True

    def terminate(self, *, grace: float | None = None) -> None:
        """Terminate the worker process group."""

        self._signal(signal.SIGTERM)
        if not self._wait(self.cancel_grace if grace is None else grace):
            self.kill()

    def kill(self) -> None:
        """Kill the worker process group."""

        if os.name == "posix":
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        else:
            self.process.kill()

    def _signal(self, sig: signal.Signals) -> None:
        try:
            if os.name == "posix":
                os.killpg(self.process.pid, sig)
            else:
                self.process.send_signal(sig)
        except ProcessLookupError:
            pass

    def _wait(self, timeout: float) -> bool:
        try:
            self.process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _read_response(self) -> None:
        if self._response is not None or self._exception is not None:
            return
        try:
            self.wait_for_handshake(timeout=0)
            if not os.path.exists(self.response_path):
                self._response = self._parent_failure_response("failed", error={"type": "WorkerProtocolError", "message": "worker exited without response", "exit_code": self.process.returncode})
                return
            response = WorkerResponse.from_json(read_json_file(self.response_path))
            if response.status == "ok" and (self._handshake is None or self._handshake.status != "ok"):
                self._response = self._parent_failure_response("failed", error={"type": "WorkerHandshakeError", "message": "worker returned ok without an ok handshake"})
                return
            self._response = response
        except Exception as exc:
            self._response = self._parent_failure_response("failed", error={"type": type(exc).__name__, "message": str(exc), "exit_code": self.process.returncode})

    def _parent_failure_response(self, status: str, *, error: Mapping[str, Any] | None = None, cancellation: Mapping[str, Any] | None = None) -> WorkerResponse:
        diagnostics = [{"message": "parent-side dispatch failure", "returncode": self.process.returncode}]
        try:
            record_id = _write_execution_record(
                self.plan.store,
                self.plan.envelope,
                status=status,
                error=error,
                cancellation=cancellation,
                diagnostics=tuple(diagnostics),
                stdout_path=self.stdout_path,
                stderr_path=self.stderr_path,
            )
        except Exception as exc:
            record_id = None
            diagnostics.append({"message": "parent-side failure provenance could not be written", "error_type": type(exc).__name__})
        response = WorkerResponse(
            status=status,
            operation_id=self.plan.envelope.operation_id,
            dispatch_id=self.plan.dispatch_spec.get("id"),
            recipe_id=self.plan.execution_recipe.get("id"),
            execution_record_id=record_id,
            error=error,
            cancellation=cancellation,
            diagnostics=tuple(diagnostics),
        )
        try:
            write_json_file(self.response_path, response.to_json())
        except Exception:
            pass
        self._persist_logs(record_id)
        return response

    def _persist_logs(self, record_id: str | None) -> None:
        if not record_id:
            return
        try:
            records = self.plan.store.records
            product_dir = records.resolve_storage_ref(StorageRef.self_product(path=".", role="logs"), record_id=record_id, create=True)
            for source, name in ((self.stdout_path, "stdout.txt"), (self.stderr_path, "stderr.txt")):
                if os.path.exists(source):
                    shutil.copyfile(source, product_dir / name)
        except Exception:
            pass

    def _cleanup(self) -> None:
        for path in self.plan.envelope.launch.get("cleanup_paths", ()):  # type: ignore[union-attr]
            if isinstance(path, str):
                shutil.rmtree(path, ignore_errors=True)
        if not self.preserve_work_dir:
            shutil.rmtree(self.work_dir, ignore_errors=True)


class LocalSubprocessBackend:
    """Popen-based local backend for one dispatch worker process."""

    name = "local_subprocess"

    def __init__(self, *, preserve_work_dir: bool = False, handshake_timeout: float = 10.0):
        self.preserve_work_dir = preserve_work_dir
        self.handshake_timeout = handshake_timeout

    def submit(self, plan: Any) -> LocalSubprocessFuture:
        """Launch a worker subprocess for *plan*."""

        work_dir = tempfile.mkdtemp(prefix="dryml-dispatch-")
        request_path = os.path.join(work_dir, "request.json")
        handshake_path = os.path.join(work_dir, "handshake.json")
        response_path = os.path.join(work_dir, "response.json")
        stdout_path = os.path.join(work_dir, "stdout.txt")
        stderr_path = os.path.join(work_dir, "stderr.txt")
        try:
            envelope = plan.envelope
            save_envelope(request_path, envelope)
            cmd, child_env = build_worker_command(envelope.environment_spec)
            child_env.update({str(key): str(value) for key, value in (envelope.allocation_view.get("env") or {}).items()})
            cmd.extend(["-m", "dryml.dispatch.worker", "--request", request_path, "--handshake", handshake_path, "--response", response_path])
            _report("dryml.dispatch.worker.launch", "Launching local subprocess worker", operation_id=envelope.operation_id, data={"cmd": _command_summary(cmd), "work_dir": work_dir})
            stdout = open(stdout_path, "w", encoding="utf-8")
            stderr = open(stderr_path, "w", encoding="utf-8")
            try:
                process = subprocess.Popen(cmd, env=child_env, stdout=stdout, stderr=stderr, cwd=work_dir, start_new_session=(os.name == "posix"))
            finally:
                stdout.close()
                stderr.close()
        except BaseException as exc:
            _cleanup_launch_paths(plan)
            shutil.rmtree(work_dir, ignore_errors=True)
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise DispatchLaunchError("failed to launch local subprocess worker", context={"error": str(exc)}) from exc
        future = LocalSubprocessFuture(process, plan, work_dir, request_path, handshake_path, response_path, stdout_path, stderr_path, self.preserve_work_dir, handshake_timeout=self.handshake_timeout)
        try:
            future.wait_for_handshake(timeout=self.handshake_timeout)
        except BaseException:
            try:
                future.cancel(reason="worker_protocol_error", record=False)
            finally:
                future._cleanup()
            raise
        return future


def _cleanup_launch_paths(plan: Any) -> None:
    """Remove normalization artifacts when worker launch never returns a future."""

    for path in plan.envelope.launch.get("cleanup_paths", ()):
        if isinstance(path, str):
            shutil.rmtree(path, ignore_errors=True)


def build_worker_command(environment_spec: Mapping[str, Any] | None) -> tuple[list[str], dict[str, str]]:
    """Build the executable prefix and environment for a worker launch."""

    spec = spec_from_data(environment_spec or CurrentEnvironmentSpec().to_data())
    child_env = os.environ.copy()
    executable = sys.executable
    prefix: list[str] | None = None
    env_overrides: Mapping[str, str] = {}
    pythonpath_policy = "dryml-source"
    extra_pythonpath: tuple[str, ...] = ()
    if isinstance(spec, CurrentEnvironmentSpec):
        executable = sys.executable
    elif isinstance(spec, PythonExecutableSpec):
        executable = spec.executable
        env_overrides = spec.env
        pythonpath_policy = spec.pythonpath_policy
        extra_pythonpath = spec.extra_pythonpath
    elif isinstance(spec, CondaEnvironmentSpec):
        env_overrides = spec.env
        pythonpath_policy = spec.pythonpath_policy
        extra_pythonpath = spec.extra_pythonpath
        if spec.launch_mode == "direct":
            executable = spec.direct_python_executable()
        else:
            prefix = [spec.conda_executable, "run"]
            if spec.prefix:
                prefix.extend(["-p", spec.prefix])
            elif spec.name:
                prefix.extend(["-n", spec.name])
            else:
                raise DispatchLaunchError("conda-run launch requires prefix or name")
            prefix.extend(["--no-capture-output", "--", "python"])
    else:
        raise DispatchLaunchError("unsupported environment spec for local subprocess", context={"kind": getattr(spec, "kind", None)})
    child_env.update({str(key): str(value) for key, value in env_overrides.items()})
    _apply_pythonpath_policy(child_env, pythonpath_policy, extra_pythonpath)
    return (prefix or [executable]), child_env


def _apply_pythonpath_policy(env: dict[str, str], policy: str, extra_pythonpath: tuple[str, ...]) -> None:
    if policy not in {"none", "inherit", "explicit", "dryml-source"}:
        raise DispatchLaunchError("unsupported pythonpath_policy", context={"policy": policy})
    for path in extra_pythonpath:
        if not isinstance(path, str) or not path:
            raise DispatchLaunchError("extra_pythonpath entries must be non-empty strings")
    if policy == "none":
        return
    paths: list[str] = []
    if policy == "inherit":
        existing = os.environ.get("PYTHONPATH")
        if existing:
            paths.extend(existing.split(os.pathsep))
    elif policy == "dryml-source":
        paths.append(str(_dryml_source_root()))
    paths.extend(extra_pythonpath)
    if paths or policy == "explicit":
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(path for path in paths if path))


def _dryml_source_root() -> Path:
    """Return the import root for DRYML source in checkouts or installs."""

    package_dir = Path(__file__).resolve().parents[1]
    if package_dir.parent.name == "src" and (package_dir.parent / "dryml").is_dir():
        return package_dir.parent
    cwd = Path.cwd()
    checkout_src = cwd / "src"
    if (cwd / "pyproject.toml").is_file() and (checkout_src / "dryml").is_dir():
        return checkout_src.resolve()
    return package_dir.parent


def _write_execution_record(store: Any, envelope: Any, *, status: str, error: Mapping[str, Any] | None = None, cancellation: Mapping[str, Any] | None = None, diagnostics: tuple[Mapping[str, Any], ...] = (), stdout_path: str | None = None, stderr_path: str | None = None, consumed_cdef_ids: tuple[str, ...] = (), produced_cdef_ids: tuple[str, ...] = (), result_record_ids: tuple[str, ...] = ()) -> str | None:
    if envelope.record_policy == "none" or store is None:
        return None
    _persist_provenance_specs(store, envelope)
    logs = (
        ExecutionLogRef("stdout", StorageRef.self_product(path="stdout.txt", role="stdout"), "text/plain"),
        ExecutionLogRef("stderr", StorageRef.self_product(path="stderr.txt", role="stderr"), "text/plain"),
    )
    execution = ExecutionRecord(
        execution_kind="python",
        operation_id=envelope.operation_id,
        backend=_backend_identity(envelope),
        status=status,
        dispatch_id=envelope.dispatch_spec.get("id"),
        recipe_id=envelope.execution_recipe.get("id"),
        world_id=envelope.launch.get("world_id"),
        world_allocation_id=envelope.allocation_view.get("world_allocation_id") or envelope.launch.get("world_allocation_id"),
        consumed_cdef_ids=consumed_cdef_ids,
        produced_cdef_ids=produced_cdef_ids,
        produced_records=result_record_ids,
        logs=logs,
        error=ExecutionErrorInfo.from_json(error) if error else None,
        cancellation=ExecutionCancellationInfo.from_json(cancellation) if cancellation else None,
        diagnostics=diagnostics,
        metadata=_execution_metadata(envelope),
        extra=_execution_extra(envelope),
    )
    _report("dryml.dispatch.execution_record.write", "Writing execution record", operation_id=envelope.operation_id, data={"status": status})
    return write_execution_record(store.records, execution).record_id


def _persist_provenance_specs(store: Any, envelope: Any) -> None:
    record_io = store.records
    record_io.write_spec(envelope.operation_spec, family="operation")
    record_io.write_spec(envelope.dispatch_spec, family="dispatch")
    record_io.write_spec(envelope.execution_recipe, family="execution_recipe")
    if isinstance(envelope.launch.get("world_spec"), Mapping):
        record_io.write_spec(envelope.launch["world_spec"], family="world")
    if isinstance(envelope.launch.get("world_allocation_spec"), Mapping):
        record_io.write_spec(envelope.launch["world_allocation_spec"], family="world_allocation")


def _backend_identity(envelope: Any) -> Mapping[str, Any]:
    if envelope.execution_recipe.get("payload", {}).get("backend", {}).get("kind") == "local_world":
        return {"name": "dryml.local_world", "kind": "local_world", "version": "1"}
    return BACKEND_IDENTITY


def _execution_metadata(envelope: Any) -> dict[str, Any]:
    alloc = envelope.allocation_view or {}
    metadata = dict(alloc.get("metadata") or {})
    for field_name in ("role", "replica", "rank", "local_rank"):
        if field_name in alloc:
            metadata[field_name] = alloc.get(field_name)
    env = alloc.get("env") or {}
    for key, name in (("DRYML_WORLD_SIZE", "world_size"), ("DRYML_WORLD_ROLE_SIZE", "role_size")):
        if key in env:
            try:
                metadata[name] = int(env[key])
            except Exception:
                metadata[name] = env[key]
    coordination = envelope.launch.get("coordination") or {}
    if coordination.get("group_id"):
        metadata["coordination_group_id"] = coordination.get("group_id")
    return metadata


def _execution_extra(envelope: Any) -> dict[str, Any]:
    alloc = envelope.allocation_view or {}
    return {
        "worker_key": {
            "role": alloc.get("role"),
            "replica": alloc.get("replica"),
            "rank": alloc.get("rank"),
            "local_rank": alloc.get("local_rank"),
        }
    }


def _report(name: str, message: str, *, operation_id: str | None = None, data: Mapping[str, Any] | None = None) -> None:
    try:
        from dryml import reporting

        reporting.step(name, message, operation_id=operation_id, data=data or {})
    except Exception:
        pass


def _command_summary(cmd: list[str]) -> list[str]:
    return [cmd[0], *cmd[1:4], "..."] if len(cmd) > 4 else cmd


__all__ = ["BACKEND_IDENTITY", "LocalSubprocessBackend", "LocalSubprocessFuture", "build_worker_command"]
