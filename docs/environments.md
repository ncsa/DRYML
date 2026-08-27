# Environments

`dryml.environments` describes Python/software environments without changing DRYML object identity, Repo behavior, Store layout, records, sidecars, or materialization semantics.

The module is intentionally lightweight. Importing `dryml.environments` does not inspect the host, import optional frameworks, or activate session/runtime controls. Introspection is explicit. Existing probe helpers are separate opt-in tools and are not consumed by `dryml.session`, annotations, world planning, or runtime publication.

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
    capabilities=("dryml.environments.v1.1",),
    tags=("torch",),
)

report = req.check(info, policy="compatible")
if not report.ok:
    print(report.explain())
```

Policies are `ignore`, `warn`, `compatible`, and `strict`. Reports keep structured `CompatibilityIssue` entries with stable issue codes and readable `explain()` output.

PEP 508 environment markers are evaluated from the `EnvironmentRecord` being checked, not from the coordinator process. If a marker references platform metadata that the record cannot provide, the check reports an `unknown` compatibility issue instead of silently using local platform facts.

Environment checks are software-focused. Accelerator allocation, process topology, runtime activation, and framework outcomes belong to the world/runtime layers described in [World And Runtime](world_runtime.md), not to `EnvironmentRequirement`.

## Content IDs

`EnvironmentRecord`, `EnvironmentRequirement`, environment specs, and lock refs have stable content IDs.

```python
record_id = info.id
requirement_id = req.id
```

These values emit closed `contract_version: "1.1"` envelopes. The environment families are `dryml.environment_record.v1.1` / `envrec`, `dryml.environment_requirement.v1.1` / `envreq`, `dryml.environment_spec.v1.1` / `envspec`, and `dryml.environment_lock.v1.1` / `envlock`. IDs use `<prefix>-v1.1-<sha256>` over the schema, kind, and identifying projection. Source-v1 and future contract versions are rejected; there is no metadata migration.

Record identity includes interpreter version/implementation, platform facts, normalized distribution names/versions, DRYML protocol/schema/features, kind, and tags. Interpreter paths/prefixes, distribution location/installer/editability, DRYML git revision, `details`, and envelope metadata are inspectable but non-identifying. Requirement `details` and all envelope metadata are likewise non-identifying. Specs and locks identify every payload field.

Record, requirement, and spec metadata fields are deeply frozen at construction. Mutating an input dictionary or list after construction cannot change the object payload or invalidate its content ID. Canonical JSON sorts string mapping keys, uses compact UTF-8 encoding, rejects duplicate textual keys, non-string keys, non-finite floats, and non-JSON values. Shared values are bounded to depth 8, 1,024 nodes, 64 entries/container, 4,096-codepoint strings/keys, 4,096-bit integers, and 4 MiB envelopes; environment records allow 4,096 distributions, 65,536 nodes, and 16 MiB envelopes.

## Explicit Inspection Tools

`inspect_current()` is the supported lightweight in-process observation used by
environment compatibility. Existing `probe(...)` and `probe_python(...)`
helpers remain explicit tools, but they are not provider selection, automatic
inference, session activation, or a dispatch protocol. No declaration example
in this release launches a probe or another process.

## Registry

Use `EnvironmentRegistry` as an in-memory named environment catalog.

```python
registry = envs.EnvironmentRegistry()
registry.register(
    "torch-dev",
    envs.CondaEnvironmentSpec(prefix="/opt/envs/torch"),
    provides=("dryml.environments.v1.1",),
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
