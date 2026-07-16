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

## Context-Local Planning Defaults

`current()`, `set_current(...)`, `reset_current()`, and `use(...)` manage a
context-local default candidate for later dispatch or explain calls. They do not
activate a runtime, allocate resources, mutate a registry, or import ML
frameworks.

```python
import dryml.environments as environments

with environments.use(environments.CurrentEnvironmentSpec()):
    assert environments.current() is not None
```

An explicit dispatch candidate takes precedence over this default. A current
environment remains subject to declared hard requirements and bounded candidate
checks.

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

Probe the current interpreter through the bounded worker protocol (the default
finite timeout):

```python
result = envs.probe(envs.CurrentEnvironmentSpec())
info = result.require_ok()
```

Pass `timeout=None` only when explicitly requesting in-process inspection.

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
`get()`, `list()`, and `unregister(name)` are probe-free lifecycle operations;
`list()` is name-sorted and `unregister()` returns the removed entry.

The registry is an explicit catalog, while `envs.resolve(...)` performs bounded
candidate selection for dispatch: caller candidates precede name-sorted registry
entries, then the current environment. Registry labels are only probe prefilters;
the observed environment record remains the compatibility authority. Resolver
search never replaces an explicit, annotation-default, or context-current
candidate.

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
# Environment Resolution

`EnvironmentRegistry` is an explicit in-memory notebook or application object;
DRYML does not provide a global or persistent registry. Use
`environments.resolve(requirement, candidates=..., registry=...)` to search
caller candidates, name-sorted registry entries, then the current environment.
Resolution is bounded, deduplicates canonical specs, records attempts, and
selects the first strictly compatible candidate when a requirement is supplied.
Without a requirement it performs no probe fan-out and selects the first
structurally launchable candidate. Unsupported container specs and Conda specs
that cannot produce a local worker command are recorded and skipped without a
probe. Registry labels are only probe prefilters, never proof of compatibility.
`max_candidates`, `probe_timeout`,
and `total_timeout` bound DRYML's own search work; a finite total timeout also
bounds a built-in probe when no per-probe timeout is supplied. Injected probe
runners and arbitrary candidate iterators are cooperative callbacks, so callers
requiring a hard deadline must provide a timeout-enforcing subprocess runner and
bounded candidates. Resolver reports redact environment overrides and bound
diagnostic metadata before serialization. Current-environment resolver probes use
the bounded probe worker path rather than synchronous local introspection.
Probe cleanup terminates the probe process group, but cannot reliably terminate
an untrusted descendant that deliberately escapes that group; use an external
sandbox or cgroup when probing untrusted executables.

Notebook users retain the registry object themselves; re-running a setup cell is
deterministic as long as registration names are not duplicated. Use
`environments.use(...)` for temporary context-local overrides and
`set_current(...)` only for a deliberate session default. Resolution has no
cross-plan cache: each dispatch/explain request performs only its bounded work.
