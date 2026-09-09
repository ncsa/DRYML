from __future__ import annotations

import abc
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Mapping, TypeVar

from .protocol import (
    ExecutionError,
    RemoteExecutionError,
    load_response,
    save_request,
)
from .worker import execute_request

if TYPE_CHECKING:
    from dryml.environments import EnvironmentRequirement
    from dryml.worlds import WorldRequirement

    from .models import DiscoverySnapshot, ResourceSnapshot, SubmittedCall
    from .output import ExecutionOutput


T = TypeVar("T")


class Backend(abc.ABC):
    """Define the additive backend contract without launching a workload.

    Concrete U4+ backends own discovery, admission, transfer, native execution,
    and cleanup for already accepted calls. This interface intentionally coexists
    with ``BackendBase`` until the U7 public cutover preserves no legacy users.
    """

    @abc.abstractmethod
    def start(self) -> None:
        """Initialize the selected backend when an executor explicitly starts it."""

    @abc.abstractmethod
    def capabilities(self) -> frozenset[str]:
        """Return supported control capability names without claiming admission."""

    @abc.abstractmethod
    def create_future(self, submission_id: str, output: "ExecutionOutput") -> "ExecutionFuture[T]":
        """Create one inert concrete future without launching or binding output."""

    @abc.abstractmethod
    def submit(self, call: "SubmittedCall[T]", *, future: "ExecutionFuture[T]") -> None:
        """Schedule one accepted descriptor-only call on its matching future."""

    @abc.abstractmethod
    def discover(self, *, environment: "EnvironmentRequirement | None" = None, world: "WorldRequirement | None" = None, timeout: float) -> "DiscoverySnapshot":
        """Return a bounded discovery snapshot without invoking workload code."""

    @abc.abstractmethod
    def resources(self, *, timeout: float) -> "ResourceSnapshot":
        """Return the selected backend's bounded resource observation."""

    @abc.abstractmethod
    def reconcile_cleanup(self, submission_id: str, *, timeout: float) -> None:
        """Retry only the named submission's cleanup; unknown IDs are errors."""

    @abc.abstractmethod
    def close(self, *, cancel: bool, timeout: float | None) -> None:
        """Close this backend without cancelling unrelated caller-owned resources."""


class ExecutionFuture(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def done(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def result(self, timeout: float | None = None):
        raise NotImplementedError

    @abc.abstractmethod
    def exception(self, timeout: float | None = None):
        raise NotImplementedError

    @abc.abstractmethod
    def cancel(self) -> bool:
        raise NotImplementedError


class BackendBase(abc.ABC):
    name = "base"

    def can_run(self, requirements: Mapping[str, Any] | None = None) -> bool:
        return True

    @abc.abstractmethod
    def submit(self, request, *, env: Mapping[str, str] | None = None) -> ExecutionFuture:
        raise NotImplementedError

    def run(self, request, *, env: Mapping[str, str] | None = None):
        return self.submit(request, env=env).result()


class InlineFuture(ExecutionFuture):
    def __init__(self, request):
        self._response = None
        self._exception = None
        try:
            self._response = execute_request(request)
        except BaseException as exc:
            self._exception = exc

    def done(self) -> bool:
        return True

    def result(self, timeout: float | None = None):
        if self._exception is not None:
            raise self._exception
        if self._response is not None and not self._response.ok:
            raise RemoteExecutionError(self._response)
        return self._response

    def exception(self, timeout: float | None = None):
        try:
            self.result(timeout=timeout)
        except BaseException as exc:
            return exc
        return None

    def cancel(self) -> bool:
        return False


class InlineBackend(BackendBase):
    name = "inline"

    def submit(self, request, *, env: Mapping[str, str] | None = None) -> ExecutionFuture:
        if env:
            raise ValueError("InlineBackend cannot apply child-only environment overrides.")
        return InlineFuture(request)


@dataclass(slots=True)
class LocalProcessFuture(ExecutionFuture):
    process: subprocess.Popen
    response_path: str
    stdout_path: str
    stderr_path: str
    work_tmp: tempfile.TemporaryDirectory

    _response: object | None = None
    _exception: BaseException | None = None

    def done(self) -> bool:
        return self.process.poll() is not None

    def _read_response(self):
        if self._response is not None or self._exception is not None:
            return

        if not os.path.exists(self.response_path):
            stderr = ""
            if os.path.exists(self.stderr_path):
                with open(self.stderr_path, "r", encoding="utf-8", errors="replace") as f:
                    stderr = f.read()
            self._exception = ExecutionError(
                "Local process finished without writing a response. "
                f"returncode={self.process.returncode}\n{stderr}"
            )
            return

        response = load_response(self.response_path)
        if response.ok:
            self._response = response
        else:
            self._exception = RemoteExecutionError(response)

    def result(self, timeout: float | None = None):
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Execution did not finish before timeout.") from exc

        self._read_response()
        if self._exception is not None:
            raise self._exception
        return self._response

    def exception(self, timeout: float | None = None):
        try:
            self.result(timeout=timeout)
        except BaseException as exc:
            return exc
        return None

    def cancel(self) -> bool:
        if self.done():
            return False

        try:
            if os.name == "posix":
                os.killpg(self.process.pid, signal.SIGINT)
            else:
                self.process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return False
        return True

    def terminate(self, timeout: float = 5.0) -> None:
        if self.done():
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()


class LocalProcessBackend(BackendBase):
    name = "process"

    def __init__(self, python_executable: str | None = None):
        self.python_executable = python_executable or sys.executable

    def submit(self, request, *, env: Mapping[str, str] | None = None) -> LocalProcessFuture:
        work_tmp = tempfile.TemporaryDirectory(prefix="dryml-exec-")
        request_path = os.path.join(work_tmp.name, "request.pkl")
        response_path = os.path.join(work_tmp.name, "response.pkl")
        stdout_path = os.path.join(work_tmp.name, "stdout.txt")
        stderr_path = os.path.join(work_tmp.name, "stderr.txt")

        save_request(request, request_path)

        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        pythonpath = [p for p in sys.path if p]
        existing_pythonpath = child_env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath.extend(existing_pythonpath.split(os.pathsep))
        child_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))

        stdout = open(stdout_path, "w", encoding="utf-8")
        stderr = open(stderr_path, "w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [
                    self.python_executable,
                    "-m",
                    "dryml.execute.worker",
                    "--request",
                    request_path,
                    "--response",
                    response_path,
                ],
                env=child_env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=(os.name == "posix"),
            )
        finally:
            stdout.close()
            stderr.close()

        return LocalProcessFuture(
            process=process,
            response_path=response_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            work_tmp=work_tmp,
        )
