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

Policies are `ignore`, `warn`, `compatible`, and `strict`. Reports keep structured `CompatibilityIssue` entries with stable issue codes and readable `explain()` output. `ok` follows the selected compatibility policy, while `admission_ok` is the fail-closed hard-admission decision: only an internally consistent compatible report returned by `EnvironmentRequirement.check` under `compatible` or `strict` is admissible. Manually constructed or deserialized reports, and `warn`, `ignore`, unknown, malformed, and unavailable reports, are not.

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

## Passive Hard Declarations

`req(...)` attaches one process-local hard requirement through passive annotations. It returns the exact class, function, method definition, `staticmethod`, `classmethod`, or supported custom descriptor that it receives; it never wraps a call, probes a host, selects an environment, or activates a runtime.

```python
@envs.req(requirements=("torch>=2.4,<2.7",), tags=("torch",))
class TorchObject:
    pass

@envs.req(requirements=("transformers>=4.45",), tags=("nlp",))
class TextTorchObject(TorchObject):
    pass

result = envs.requirements_for(TextTorchObject)
if result.has_value:
    print(result.value.requirements)
else:
    print(result.report.issues)
```

`requirements_for(...)` combines declarations attached directly to a target or inherited by a class. `requirements_for_method(owner, method_name)` combines inherited class declarations with one method selected statically, including when `owner` is an instance; it does not bind a descriptor or read instance state. A `RequirementResult` is either empty when no environment declarations are present, valued with one compatible `EnvironmentRequirement`, or valueless with a complete bounded conflict report. Environment declarations ignore annotations owned by worlds and do not import the world package.

Repeat `@envs.req(...)` declarations to express several hard constraints; the
resolver combines every compatible declaration rather than providing additive or
override modes. `dryml.env` is a lazy alias for this plural owner. Environment
resolution is independent of world resolution and does not choose an environment
candidate.

Package requirements accept normalized names, version specifiers, and record-evaluable markers. Extras, direct URLs, and markers that mention `extra` are rejected because an `EnvironmentRecord` cannot prove those constraints. Each iterable or mapping field accepts at most 64 entries; combination accepts at most 64 environment declarations and preserves up to 64 ordinalized source explanations in `EnvironmentRequirement.details["sources"]`.

Hard declarations are not defaults, candidate selection, automatic enforcement, runtime/session state, or dispatch behavior. Consumers explicitly call `check(...)` with environment evidence and may pass its resulting report to the shared admission barrier when they need fail-closed admission. They do not infer constraints from code, run probes, or install packages.

The retired fragment, additive, and override declaration forms are not supported and have no compatibility decoder or migration path. Fresh inspected records no longer advertise the former `environment_fragment` schema capability; this changes fresh record IDs while leaving the environment-record schema itself at v1.1.

## What This Does Not Change

This module does not attach environment metadata to `ConcreteDefinition`, `Definition`, or `Object`. It does not add Store `records/` persistence, SQLite record tables, object-load enforcement, dispatch, provider probes, worker handshakes, or class-update tooling.

Existing object-only Stores remain valid and loadable without environment records.
