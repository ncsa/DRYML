# Environments

`dryml.environments` describes Python/software environments without changing DRYML object identity, Repo behavior, Store layout, or materialization semantics.

The module is intentionally lightweight. Importing `dryml.environments` does not import Repo internals, TensorFlow, PyTorch, JAX, Ray, Conda Python APIs, or dispatch code.

## Inspect Current Environment

```python
import dryml.environments as envs

info = envs.inspect_current()
print(info.python.version)
print(info.platform.platform)
print(info.id)
```

Inspection uses `importlib.metadata` for installed distributions. It does not import package runtime modules to learn versions.

## Requirements And Reports

Use Python packaging requirement strings for package constraints.

```python
req = envs.EnvironmentRequirement(
    python=">=3.10,<3.13",
    requirements=("dryml>=0.3", "torch>=2.4,<2.7"),
    excludes=("tensorflow",),
    capabilities=("dryml.environments.v1",),
    tags=("torch",),
)

report = req.check(info, policy="compatible")
if not report.ok:
    print(report.explain())
```

Policies are `ignore`, `warn`, `compatible`, and `strict`. Reports keep structured `CompatibilityIssue` entries with stable issue codes and readable `explain()` output.

PEP 508 environment markers are evaluated from the `EnvironmentRecord` being checked, not from the coordinator process. If a marker references platform metadata that the record cannot provide, the check reports an `unknown` compatibility issue instead of silently using local platform facts.

Environment checks are software-focused. CUDA, GPU allocation, process topology, and framework runtime configuration are future provider/context/world work, not ordinary `EnvironmentRequirement` fields.

## Content IDs

`EnvironmentRecord`, `EnvironmentRequirement`, environment specs, and lock refs have stable content IDs.

```python
record_id = info.id
requirement_id = req.id
```

`EnvironmentRecord.id` includes observed provenance such as interpreter path, prefixes, platform, packages, tags, and details. It is an exact observed-environment key, not a package-solver equivalence class.

Record, requirement, and spec metadata fields are deeply frozen at construction. Mutating an input dictionary or list after construction cannot change the object payload or invalidate its content ID. Arbitrary JSON metadata must use string mapping keys and finite numeric values; non-string keys, key collisions caused by coercion, non-finite floats, and non-JSON values are rejected at construction/freezing time.

## Probing

Probe the current interpreter in process:

```python
result = envs.probe(envs.CurrentEnvironmentSpec())
info = result.require_ok()
```

Probe another Python executable through the worker protocol:

```python
result = envs.probe_python("/opt/envs/torch/bin/python")
if result.ok:
    print(result.record.id)
else:
    print(result.report.explain())
```

Probe specs enforce `pythonpath_policy` before launching a worker:

```python
isolated = envs.PythonExecutableSpec(
    "/opt/envs/torch/bin/python",
    pythonpath_policy="none",       # remove inherited PYTHONPATH
)

explicit = envs.PythonExecutableSpec(
    "/opt/envs/torch/bin/python",
    pythonpath_policy="explicit",   # use only these paths
    extra_pythonpath=("/project/src",),
)
```

The supported policies are `none`, `explicit`, `inherit`, and `dryml-source`. `dryml-source` injects this DRYML checkout's source root without inheriting unrelated coordinator paths. `PYTHONPATH` is controlled only by `pythonpath_policy` and `extra_pythonpath`; `env={"PYTHONPATH": ...}` is ignored so callers cannot accidentally bypass the selected isolation policy.

Represent Conda probes without depending on Conda Python libraries:

```python
direct = envs.CondaEnvironmentSpec(prefix="/opt/envs/torch", launch_mode="direct")
print(direct.probe_command())

conda_run = envs.CondaEnvironmentSpec(name="torch", launch_mode="conda-run")
print(conda_run.probe_command())
```

If Conda is absent or the command fails, probing returns a structured failure report.

## Registry

Use `EnvironmentRegistry` as an in-memory named environment catalog.

```python
registry = envs.EnvironmentRegistry()
registry.register(
    "torch-dev",
    envs.CondaEnvironmentSpec(prefix="/opt/envs/torch"),
    provides=("dryml.environments.v1",),
    tags=("torch", "dev"),
)

match = registry.find(req)
```

Duplicate names are rejected so selection is deterministic.

## Requirement Fragments

Decorators are sugar over requirement fragments. They are not the future provider system.

```python
@envs.req(requirements=("torch>=2.4,<2.7",), tags=("torch",))
class TorchObject:
    pass

@envs.add_req(requirements=("transformers>=4.45",), tags=("nlp",))
class TextTorchObject(TorchObject):
    pass

final = envs.requirements_for_class(TextTorchObject)
print(final.requirements)
print(final.explain_sources())
```

Use `override_req(...)` when a subclass intentionally replaces a field.

## What This Does Not Change

This module does not attach environment metadata to `ConcreteDefinition`, `Definition`, or `Object`. It does not add Store `records/` persistence, SQLite record tables, object-load enforcement, dispatch, provider probes, worker handshakes, or class-update tooling.

Existing object-only Stores remain valid and loadable without environment records.
