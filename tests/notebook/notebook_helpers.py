"""Shared contracts and isolated execution for maintained DRYML notebooks."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping


_REPOSITORY = Path(__file__).resolve().parents[2]
_MAX_CELLS = 200
_MAX_SOURCE_CHARS = 200_000
_MAX_METADATA_BYTES = 256
_OPTIONAL_IMPORTS = frozenset(
    {
        "cupy",
        "jax",
        "jaxlib",
        "keras",
        "nvidia",
        "pynvml",
        "py3nvml",
        "ray",
        "sklearn",
        "tensorflow",
        "torch",
        "xgboost",
    }
)
_EXTRA_OPTIONAL_IMPORTS = {
    "jax": frozenset({"jax", "jaxlib"}),
    "ray": frozenset({"ray"}),
    "sklearn": frozenset({"sklearn"}),
    "tf": frozenset({"keras", "tensorflow"}),
    "torch": frozenset({"torch"}),
    "xgboost": frozenset({"sklearn", "xgboost"}),
}
_NETWORK_IMPORTS = (
    "aiohttp",
    "ftplib",
    "http.client",
    "requests",
    "socket",
    "urllib",
    "wget",
)
_OBSOLETE_IMPORTS = (
    "dryml.context",
    "dryml.contexts",
    "dryml.tune",
)
_OBSOLETE_NAMES = frozenset({"ComputeContext", "ObjectDef", "Trainable"})
_MAGIC_LINE = re.compile(r"^\s*%")
_SHELL_LINE = re.compile(r"^\s*!")
_MAX_CAPTURE_BYTES = 8_000
_TRUNCATION_MARKER = "\n... output truncated ..."


class NotebookValidationError(ValueError):
    """A notebook does not satisfy the canonical source contract."""


class NotebookExecutionError(RuntimeError):
    """A notebook did not complete in the isolated child contract."""


@dataclass(frozen=True)
class NotebookSpec:
    """Execution requirements for one canonical notebook."""

    path: Path
    extras: tuple[str, ...] = ()
    python_max_exclusive: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Validate declared extras and the optional Python upper bound."""

        unknown = set(self.extras) - set(_EXTRA_OPTIONAL_IMPORTS)
        if unknown:
            raise ValueError(f"unknown notebook extras: {', '.join(sorted(unknown))}")
        bound = self.python_max_exclusive
        if bound is not None:
            if not isinstance(bound, tuple) or len(bound) != 2:
                raise ValueError("python_max_exclusive must be a major/minor tuple")
            if any(not isinstance(value, int) for value in bound):
                raise ValueError("python_max_exclusive must be a major/minor tuple")

    @property
    def allowed_optional_imports(self) -> frozenset[str]:
        """Return optional import roots implied by the declared DRYML extras."""

        return frozenset().union(*(_EXTRA_OPTIONAL_IMPORTS[extra] for extra in self.extras))

    @property
    def supports_current_python(self) -> bool:
        """Return whether the running Python satisfies this notebook's upper bound."""

        return self.python_max_exclusive is None or sys.version_info[:2] < self.python_max_exclusive


@dataclass(frozen=True)
class NotebookExecutionResult:
    """Bounded execution and cleanup evidence returned by the child runner."""

    returncode: int
    state_restored: bool
    working_directory_restored: bool
    module_table_restored: bool
    linecache_restored: bool
    optional_imports: frozenset[str]
    unexpected_writes: tuple[str, ...]
    repository_on_pythonpath: bool
    stdout: str
    stderr: str


CANONICAL_NOTEBOOKS = (
    NotebookSpec(Path("examples/notebooks/objects_definitions_and_repos.ipynb")),
    NotebookSpec(Path("examples/notebooks/datasets_and_transforms.ipynb")),
    NotebookSpec(Path("examples/notebooks/local_defaults_and_plain_mode.ipynb")),
    NotebookSpec(
        Path("examples/notebooks/models_experiments_and_metrics.ipynb"),
        extras=("sklearn",),
        python_max_exclusive=(3, 14),
    ),
    NotebookSpec(
        Path("examples/notebooks/definition_driven_experiments.ipynb"),
        extras=("sklearn",),
        python_max_exclusive=(3, 14),
    ),
    NotebookSpec(
        Path("examples/notebooks/local_hyperparameter_search.ipynb"),
        extras=("sklearn",),
        python_max_exclusive=(3, 14),
    ),
)


def repository_path(relative: Path) -> Path:
    """Return an absolute path for one repository-relative canonical path."""

    return _REPOSITORY / relative


def _error(path: Path, message: str, cell: int | None = None) -> NotebookValidationError:
    location = str(path)
    if cell is not None:
        location = f"{location}: cell {cell}"
    return NotebookValidationError(f"{location}: {message}")


def _cell_source(path: Path, cell: Mapping[str, Any], index: int) -> str:
    if "source" not in cell:
        raise _error(path, "source is required", index)
    source = cell["source"]
    if isinstance(source, str):
        result = source
    elif isinstance(source, list) and all(isinstance(line, str) for line in source):
        result = "".join(source)
    else:
        raise _error(path, "source must be a string or a list of strings", index)
    if len(result) > _MAX_SOURCE_CHARS:
        raise _error(path, "source is too large", index)
    return result


def _attribute_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_absolute_path(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


class _ExecutablePolicy(ast.NodeVisitor):
    """Reject executable notebook syntax outside the canonical policy."""

    def __init__(self, path: Path, cell: int):
        """Record the notebook path and one-based cell number."""

        self.path = path
        self.cell = cell

    def fail(self, message: str) -> None:
        """Raise a cell-aware validation error with *message*."""

        raise _error(self.path, message, self.cell)

    def visit_Import(self, node: ast.Import) -> None:
        """Reject forbidden modules in an import statement."""

        for alias in node.names:
            self._check_import(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Reject forbidden modules and obsolete top-level DRYML names."""

        module = node.module or ""
        self._check_import(module)
        if module == "dryml":
            for alias in node.names:
                if alias.name in _OBSOLETE_NAMES:
                    self.fail(f"obsolete API 'dryml.{alias.name}' is forbidden")
        self.generic_visit(node)

    def _check_import(self, module: str) -> None:
        """Reject one forbidden network or obsolete module path."""

        for forbidden in _NETWORK_IMPORTS:
            if module == forbidden or module.startswith(f"{forbidden}."):
                self.fail(f"network module '{module}' is forbidden")
        for forbidden in _OBSOLETE_IMPORTS:
            if module == forbidden or module.startswith(f"{forbidden}."):
                self.fail(f"obsolete import '{forbidden}' is forbidden")

    def visit_Constant(self, node: ast.Constant) -> None:
        """Reject absolute paths and literal network URLs."""

        if isinstance(node.value, str):
            if _is_absolute_path(node.value):
                self.fail("absolute path is forbidden")
            if node.value.startswith(("http://", "https://", "ftp://")):
                self.fail("network URL is forbidden")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Reject direct use of obsolete API names."""

        if node.id in _OBSOLETE_NAMES:
            self.fail(f"obsolete API '{node.id}' is forbidden")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Reject obsolete qualified names and generated identity access."""

        name = _attribute_name(node)
        if node.attr == "dry_id":
            self.fail("obsolete API 'dry_id' is forbidden")
        if name and name.startswith("dryml.") and node.attr in _OBSOLETE_NAMES:
            self.fail(f"obsolete API '{name}' is forbidden")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Reject shell, IPython, and common network calls."""

        name = _attribute_name(node.func)
        if name == "get_ipython" or (name and name.startswith("get_ipython.")):
            self.fail("IPython execution is forbidden")
        if name in {"os.system", "os.popen"} or (name and name.startswith("subprocess.")):
            self.fail("shell execution is forbidden")
        if name and name.rsplit(".", 1)[-1] in {
            "create_connection",
            "download",
            "getaddrinfo",
            "urlopen",
            "urlretrieve",
        }:
            self.fail(f"network API '{name}' is forbidden")
        self.generic_visit(node)


def _validate_metadata(path: Path, document: Mapping[str, Any]) -> None:
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise _error(path, "metadata must be an object")
    unknown = set(metadata) - {"kernelspec", "language_info"}
    if unknown:
        key = sorted(unknown)[0]
        raise _error(path, f"notebook metadata key '{key}' is forbidden")

    kernelspec = metadata.get("kernelspec")
    if kernelspec is not None:
        expected = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        if kernelspec != expected:
            raise _error(path, "kernelspec metadata must be the generic Python 3 kernel")

    language_info = metadata.get("language_info")
    if language_info is not None:
        if not isinstance(language_info, dict) or set(language_info) - {"name", "version"}:
            raise _error(path, "language_info metadata is not bounded")
        if language_info.get("name") != "python" or not isinstance(language_info.get("version", ""), str):
            raise _error(path, "language_info metadata must describe Python")

    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise _error(path, "metadata is too large")


def validate_notebook(path: Path) -> dict[str, Any]:
    """Parse and validate one notebook without importing notebook tooling."""

    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        if isinstance(exc, json.JSONDecodeError):
            raise _error(path, "invalid JSON") from exc
        raise _error(path, f"cannot read notebook: {exc}") from exc
    if not isinstance(document, dict):
        raise _error(path, "notebook must be an object")
    if document.get("nbformat") != 4:
        raise _error(path, "nbformat must be 4")
    if not isinstance(document.get("nbformat_minor"), int) or isinstance(document.get("nbformat_minor"), bool):
        raise _error(path, "nbformat_minor must be an integer")
    _validate_metadata(path, document)

    cells = document.get("cells")
    if not isinstance(cells, list):
        raise _error(path, "cells must be a list")
    if len(cells) > _MAX_CELLS:
        raise _error(path, "notebook has too many cells")

    for index, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            raise _error(path, "must be an object", index)
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            raise _error(path, "cell_type must be code, markdown, or raw", index)
        if cell.get("metadata") != {}:
            raise _error(path, "metadata must be empty", index)
        source = _cell_source(path, cell, index)
        if cell_type != "code":
            continue
        if "execution_count" not in cell or cell["execution_count"] is not None:
            raise _error(path, "execution_count must be null", index)
        if cell.get("outputs") != []:
            raise _error(path, "outputs must be empty", index)
        for line in source.splitlines():
            if _MAGIC_LINE.match(line):
                raise _error(path, "magic syntax is forbidden", index)
            if _SHELL_LINE.match(line):
                raise _error(path, "shell syntax is forbidden", index)
        filename = f"{path}::cell-{index}"
        try:
            tree = ast.parse(source, filename=filename, mode="exec")
            compile(tree, filename, "exec")
        except SyntaxError as exc:
            detail = f"invalid Python syntax at line {exc.lineno}"
            raise _error(path, detail, index) from exc
        _ExecutablePolicy(path, index).visit(tree)
    return document


_WORKER_SITECUSTOMIZE = r'''
import atexit
import json
import os
from pathlib import Path
import socket
import sys

audit_value = os.environ.get('DRYML_NOTEBOOK_AUDIT_DIR')
optional_value = os.environ.get('DRYML_NOTEBOOK_OPTIONAL_IMPORTS')
if audit_value and optional_value:
    audit_directory = Path(audit_value)
    audit_directory.mkdir(parents=True, exist_ok=True)
    optional_roots = frozenset(json.loads(optional_value))
    record_path = audit_directory / f'worker-{os.getpid()}.json'

    def _write_audit_record():
        """Atomically record this worker's identity and optional imports."""
        record = {
            'optional_imports': sorted(
                {name.split('.', 1)[0] for name in sys.modules} & optional_roots
            ),
            'pid': os.getpid(),
            'process_group': os.getpgrp() if os.name == 'posix' else None,
        }
        temporary = record_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, sort_keys=True), encoding='utf-8')
        os.replace(temporary, record_path)

    def _network_disabled(*args, **kwargs):
        """Reject network access in an audited dispatch worker."""
        raise RuntimeError('network access is disabled')

    class _OfflineSocket(socket.socket):
        """Socket replacement that rejects construction before I/O."""

        def __init__(self, *args, **kwargs):
            """Reject socket construction regardless of its arguments."""
            _network_disabled()

    _write_audit_record()
    atexit.register(_write_audit_record)
    socket.socket = _OfflineSocket
    socket.create_connection = _network_disabled
    socket.getaddrinfo = _network_disabled
'''


_CHILD_RUNNER = r'''
import json
import linecache
import os
from pathlib import Path
import socket
import sys
import types

notebook_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
optional_roots = frozenset(json.loads(sys.argv[3]))
repository = Path(sys.argv[4]).resolve()
audit_bootstrap = Path(sys.argv[5])
audit_directory = Path(sys.argv[6])
document = json.loads(notebook_path.read_text(encoding='utf-8'))
first_code_cell = next(
    (index for index, cell in enumerate(document['cells'], start=1) if cell['cell_type'] == 'code'),
    None,
)
if first_code_cell is not None:
    report_path.write_text(
        json.dumps({'error': None, 'error_cell': first_code_cell, 'status': 'starting'}, sort_keys=True),
        encoding='utf-8',
    )
before_optional = {name.split('.', 1)[0] for name in sys.modules} & optional_roots

import dryml
from dryml.runtime import active_runtime

before_environment = dryml.environments.current()
before_world = dryml.worlds.current()
before_runtime = active_runtime()
before_cwd = Path.cwd()
before_modules = dict(sys.modules)
before_linecache = dict(linecache.cache)
pythonpath_entries = [entry for entry in os.environ.get('PYTHONPATH', '').split(os.pathsep) if entry]
repository_on_pythonpath = any(Path(entry).resolve() == repository for entry in pythonpath_entries)

def _network_disabled(*args, **kwargs):
    """Reject network access in the notebook interpreter."""
    raise RuntimeError('network access is disabled')

class _OfflineSocket(socket.socket):
    """Socket replacement that rejects construction before I/O."""

    def __init__(self, *args, **kwargs):
        """Reject socket construction regardless of its arguments."""
        _network_disabled()

socket.socket = _OfflineSocket
socket.create_connection = _network_disabled
socket.getaddrinfo = _network_disabled

def _audited_worker_builder(builder):
    """Wrap one dispatch command builder with worker audit bootstrap state."""
    def build(environment_spec):
        """Return the original command with an injected sitecustomize path."""
        command, child_environment = builder(environment_spec)
        pythonpath = child_environment.get('PYTHONPATH')
        paths = [str(audit_bootstrap)]
        if pythonpath:
            paths.extend(pythonpath.split(os.pathsep))
        child_environment['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(paths))
        child_environment['DRYML_NOTEBOOK_AUDIT_DIR'] = str(audit_directory)
        child_environment['DRYML_NOTEBOOK_OPTIONAL_IMPORTS'] = json.dumps(sorted(optional_roots))
        return command, child_environment
    return build

import dryml.dispatch.backends as dispatch_backends
import dryml.dispatch.local_world as dispatch_local_world

dispatch_backends.build_worker_command = _audited_worker_builder(
    dispatch_backends.build_worker_command
)
dispatch_local_world.build_worker_command = _audited_worker_builder(
    dispatch_local_world.build_worker_command
)

module_name = '_dryml_tutorial_notebook'
module = types.ModuleType(module_name)
module.__file__ = str(notebook_path)
sys.modules[module_name] = module
execution_error = None
error_cell = None

try:
    for index, cell in enumerate(document['cells'], start=1):
        if cell['cell_type'] != 'code':
            continue
        report_path.write_text(
            json.dumps({'error': None, 'error_cell': index, 'status': 'running'}, sort_keys=True),
            encoding='utf-8',
        )
        source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        filename = f'{notebook_path}::cell-{index}'
        module.__file__ = filename
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        try:
            exec(compile(source, filename, 'exec'), module.__dict__)
        except BaseException as exc:
            execution_error = {
                'type': type(exc).__name__,
                'message': str(exc)[:1000],
            }
            error_cell = index
            break
finally:
    optional_imports = ({name.split('.', 1)[0] for name in sys.modules} & optional_roots) - before_optional
    try:
        state_restored = (
            dryml.environments.current() is before_environment
            and dryml.worlds.current() is before_world
            and active_runtime() is before_runtime
        )
    except BaseException:
        state_restored = False

    try:
        os.chdir(before_cwd)
        working_directory_restored = Path.cwd() == before_cwd
    except BaseException:
        working_directory_restored = False

    linecache.cache.clear()
    linecache.cache.update(before_linecache)
    linecache_restored = linecache.cache == before_linecache

    for name in tuple(sys.modules):
        if name not in before_modules:
            sys.modules.pop(name, None)
    for name, value in before_modules.items():
        sys.modules[name] = value
    module_table_restored = (
        set(sys.modules) == set(before_modules)
        and all(sys.modules[name] is value for name, value in before_modules.items())
    )

report = {
    'error': execution_error,
    'error_cell': error_cell,
    'state_restored': state_restored,
    'working_directory_restored': working_directory_restored,
    'module_table_restored': module_table_restored,
    'linecache_restored': linecache_restored,
    'optional_imports': sorted(optional_imports),
    'repository_on_pythonpath': repository_on_pythonpath,
    'status': 'finished',
}
report_path.write_text(json.dumps(report, sort_keys=True), encoding='utf-8')
clean = all((state_restored, working_directory_restored, module_table_restored, linecache_restored))
raise SystemExit(0 if execution_error is None and clean else 1)
'''


def _tree_snapshot(root: Path, ignored: frozenset[Path]) -> dict[str, tuple[Any, ...]]:
    snapshot: dict[str, tuple[Any, ...]] = {}
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        names[:] = [
            name
            for name in names
            if not any((current / name).relative_to(root).is_relative_to(item) for item in ignored)
        ]
        names.sort()
        filenames.sort()
        for name in names + filenames:
            path = current / name
            relative = path.relative_to(root)
            if any(relative.is_relative_to(item) for item in ignored):
                continue
            key = relative.as_posix()
            if path.is_symlink():
                snapshot[key] = ("symlink", os.readlink(path))
            elif path.is_dir():
                snapshot[key] = ("directory",)
            else:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    size = os.fstat(stream.fileno()).st_size
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                snapshot[key] = ("file", size, digest.hexdigest())
    return snapshot


def _changed_paths(before: Mapping[str, tuple[Any, ...]], after: Mapping[str, tuple[Any, ...]]) -> tuple[str, ...]:
    return tuple(sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key)))


def _drain_output(stream: Any, captured: bytearray, truncated: threading.Event) -> None:
    """Drain one child stream while retaining only the configured byte prefix."""

    for chunk in iter(lambda: stream.read(64 * 1024), b""):
        remaining = _MAX_CAPTURE_BYTES - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated.set()


def _captured_output(captured: bytearray, truncated: threading.Event) -> str:
    """Decode retained child output and append a stable truncation marker."""

    value = bytes(captured).decode("utf-8", errors="replace")
    return f"{value}{_TRUNCATION_MARKER}" if truncated.is_set() else value


def _read_report(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _worker_records(audit_directory: Path) -> tuple[dict[str, Any], ...]:
    """Return complete worker audit records, ignoring partial writes."""

    records = []
    for path in sorted(audit_directory.glob("worker-*.json")):
        record = _read_report(path)
        if record is not None and isinstance(record.get("pid"), int):
            records.append(record)
    return tuple(records)


def _process_exists(pid: int) -> bool:
    """Return whether *pid* still names a process visible to this user."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _taskkill_process_tree(pid: int) -> None:
    """Ask Windows to terminate *pid* and its descendant process tree."""

    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _kill_registered_worker(record: Mapping[str, Any]) -> None:
    """Kill one audited dispatch worker and its ordinary descendants."""

    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        return
    if os.name == "posix":
        process_group = record.get("process_group")
        try:
            if isinstance(process_group, int) and process_group > 0 and process_group != os.getpgrp():
                os.killpg(process_group, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif os.name == "nt":  # pragma: no cover - exercised by native Windows CI.
        _taskkill_process_tree(pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _kill_notebook_process(process: subprocess.Popen[bytes]) -> None:
    """Kill the notebook leader and its platform-owned process tree."""

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif os.name == "nt":  # pragma: no cover - exercised by native Windows CI.
        _taskkill_process_tree(process.pid)
        try:
            process.kill()
        except ProcessLookupError:
            pass


def _terminate_audited_workers(audit_directory: Path, *, settle: float) -> tuple[int, ...]:
    """Kill registered worker groups and return PIDs still alive at deadline."""

    deadline = time.monotonic() + settle
    while True:
        records = _worker_records(audit_directory)
        for record in records:
            if _process_exists(record["pid"]):
                _kill_registered_worker(record)
        if time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    return tuple(
        sorted(record["pid"] for record in _worker_records(audit_directory) if _process_exists(record["pid"]))
    )


def execute_notebook(
    path: Path,
    spec: NotebookSpec | None = None,
    work_root: Path | None = None,
    *,
    timeout: float = 60.0,
    validate: bool = True,
) -> NotebookExecutionResult:
    """Copy and execute one notebook in an offline, audited child process."""

    path = Path(path).resolve()
    if validate:
        validate_notebook(path)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if spec is None:
        spec = NotebookSpec(Path(path.name))
    if work_root is None:
        raise ValueError("work_root is required")
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    notebook_root = root / "notebook-tree"
    execution_cwd = root / "work"
    home = root / "home"
    temporary_root = root / "tmp"
    harness_root = root / ".harness"
    audit_bootstrap = harness_root / "bootstrap"
    audit_directory = harness_root / "workers"
    report_path = root / "child-report.json"
    for directory in (
        notebook_root,
        execution_cwd,
        home,
        temporary_root,
        harness_root,
        audit_bootstrap,
        audit_directory,
    ):
        directory.mkdir()
    (audit_bootstrap / "sitecustomize.py").write_text(_WORKER_SITECUSTOMIZE, encoding="utf-8")
    copied_notebook = notebook_root / path.name
    shutil.copy2(path, copied_notebook)

    ignored = frozenset({report_path.relative_to(root), harness_root.relative_to(root)})
    before = _tree_snapshot(root, ignored)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    for name in tuple(environment):
        if name.startswith("COVERAGE_") or name.startswith("PYTEST_"):
            environment.pop(name, None)
    environment.update(
        {
            "HOME": str(home),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TEMP": str(temporary_root),
            "TMP": str(temporary_root),
            "TMPDIR": str(temporary_root),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local" / "share"),
        }
    )
    command = [
        sys.executable,
        "-c",
        _CHILD_RUNNER,
        str(copied_notebook),
        str(report_path),
        json.dumps(sorted(_OPTIONAL_IMPORTS)),
        str(_REPOSITORY),
        str(audit_bootstrap),
        str(audit_directory),
    ]
    process = subprocess.Popen(
        command,
        cwd=execution_cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name == "posix"),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    assert process.stdout is not None and process.stderr is not None
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    stdout_truncated = threading.Event()
    stderr_truncated = threading.Event()
    drainers = (
        threading.Thread(target=_drain_output, args=(process.stdout, stdout_buffer, stdout_truncated)),
        threading.Thread(target=_drain_output, args=(process.stderr, stderr_buffer, stderr_truncated)),
    )
    for drainer in drainers:
        drainer.start()
    timeout_error = None
    leaked_workers: tuple[int, ...] = ()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_audited_workers(audit_directory, settle=0)
        _kill_notebook_process(process)
        process.wait()
        leaked_workers = _terminate_audited_workers(audit_directory, settle=1.0)
        timeout_error = exc
    finally:
        for drainer in drainers:
            drainer.join()
    stdout = _captured_output(stdout_buffer, stdout_truncated)
    stderr = _captured_output(stderr_buffer, stderr_truncated)
    if timeout_error is not None:
        report = _read_report(report_path)
        cell = report.get("error_cell") if report else None
        location = f"{path}: cell {cell}" if isinstance(cell, int) else str(path)
        leak_detail = f"; worker cleanup failed for PIDs {', '.join(map(str, leaked_workers))}" if leaked_workers else ""
        raise NotebookExecutionError(
            f"{location}: execution timed out after {timeout:g}s{leak_detail}"
        ) from timeout_error

    after = _tree_snapshot(root, ignored)
    unexpected_writes = _changed_paths(before, after)
    report = _read_report(report_path)
    if report and report.get("error"):
        error = report["error"]
        message = str(error.get("message", "child execution failed"))
        if message == "network access is disabled":
            detail = message
        else:
            detail = f"{error.get('type', 'Exception')}: {message}"
        raise NotebookExecutionError(
            f"{path}: cell {report.get('error_cell')}: {detail}; child exited with status {process.returncode}"
        )
    cleanup_fields = (
        "state_restored",
        "working_directory_restored",
        "module_table_restored",
        "linecache_restored",
    )
    failed_cleanup = (
        tuple(name for name in cleanup_fields if report.get(name) is not True)
        if report and report.get("status") == "finished"
        else ()
    )
    if failed_cleanup:
        raise NotebookExecutionError(f"{path}: child cleanup failed: {', '.join(failed_cleanup)}")
    if process.returncode != 0 or report is None:
        cell = report.get("error_cell") if report else None
        location = f"{path}: cell {cell}" if isinstance(cell, int) else str(path)
        raise NotebookExecutionError(
            f"{location}: child exited with status {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )

    worker_optional_imports = {
        name
        for record in _worker_records(audit_directory)
        for name in record.get("optional_imports", ())
        if isinstance(name, str)
    }
    optional_imports = frozenset(report.get("optional_imports", ())) | worker_optional_imports
    undeclared = optional_imports - spec.allowed_optional_imports
    if undeclared:
        raise NotebookExecutionError(f"{path}: undeclared optional imports: {', '.join(sorted(undeclared))}")
    if unexpected_writes:
        raise NotebookExecutionError(f"{path}: unexpected writes: {', '.join(unexpected_writes)}")

    return NotebookExecutionResult(
        returncode=process.returncode,
        state_restored=True,
        working_directory_restored=True,
        module_table_restored=True,
        linecache_restored=True,
        optional_imports=optional_imports,
        unexpected_writes=unexpected_writes,
        repository_on_pythonpath=bool(report.get("repository_on_pythonpath")),
        stdout=stdout,
        stderr=stderr,
    )
