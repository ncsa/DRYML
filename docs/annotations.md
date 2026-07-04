# DRYML Requirement And Default Annotations

`dryml.annotations` is the generic sidecar metadata layer for planning. It lets code declare environment, world, and runtime intent without creating DRYML Objects, activating runtime, importing heavy frameworks, or changing `ConcreteDefinition` identity.

## Public Decorators

Use the Pythonic facades for normal code:

```python
import dryml

@dryml.env.req(packages={"torch": ">=2.4"}, tags=("training",))
@dryml.world.req(accelerators={"gpu": {"min": 1}})
@dryml.world.default(cpus=8, memory="32GiB", accelerators={"gpu": 1})
@dryml.runtime.default(torch={"num_threads": 8}, env={"OMP_NUM_THREADS": "8"})
def train(model, dataset):
    ...
```

`dryml.env.req(...)` declares hard software constraints such as Python versions, package requirements, capabilities, and tags. `packages={"torch": ">=2.4"}` becomes the PEP 508 requirement `torch>=2.4`; `packages={"torch": None}` becomes `torch`.

`dryml.world.req(...)` declares hard resource and topology constraints. The default role is `main`; pass `roles={...}` for a full multi-role requirement payload.

`dryml.world.default(...)` declares an overrideable requested world shape. Defaults are suggestions and can be replaced by user overrides, but the final world spec must still satisfy hard world requirements.

`dryml.runtime.default(...)` declares process-local runtime defaults such as framework settings, device visibility policy, limits, environment overrides, and metadata. It does not call `enter_runtime(...)`, `activate(...)`, mutate `os.environ`, or import torch, TensorFlow, or JAX.

## Generic API

The low-level decorators attach `AnnotationFragment` objects directly:

```python
import dryml.annotations as ann

@ann.require(namespace="environment", fragment={"requirements": ["numpy>=1.26"]})
@ann.default(namespace="runtime", fragment={"frameworks": {"plain": {"num_threads": 4}}})
def analyze():
    ...
```

Fragments are collected with `fragments_for(...)`, `fragments_for_class(...)`, `fragments_for_callable(...)`, or `collect_fragments(...)`. Provider/probe code can participate later by passing synthetic `AnnotationFragment` instances through `provider_fragments=`.

## Resolution

Use resolution helpers to merge metadata without executing it:

```python
result = ann.resolve(train, overrides={"world": {"roles": {"main": {"process": {"resources": {"cpus": 4}}}}}})
if not result.report.ok:
    print(result.report.explain())
```

`resolve_requirements(...)` merges hard requirements. `resolve_defaults(...)` merges defaults and applies user overrides. `resolve(...)` validates the final default world spec against hard world requirements and returns structured `AnnotationIssue` entries with source traces.

## Legacy Environment Decorators

The existing `dryml.environments.req`, `add_req`, `override_req`, `fragments_for_class`, and `requirements_for_class` APIs remain supported. Legacy environment decorators also attach equivalent generic environment requirement fragments so the annotation resolver can see them.

## Argument-Role Annotations

`RefCDef`, `RefCDefArg`, `SelectorArg`, `ValueArg`, and `ArgRole` remain constructor argument-role annotations. They affect argument canonicalization and can affect `ConcreteDefinition` identity. They are re-exported from `dryml.annotations` and `dryml.annotations.arg_roles` for discoverability, but they are not serialized as `dryml.annotation.v1` specs and are not collected by requirement/default resolution.

## Runtime Barrier

Annotation decorators only declare intent. Explicit runtime activation remains available through `dryml.runtime.activate(...)` for backend, worker, inline, and power-user setup:

```python
with dryml.runtime.activate(mode="worker", allocation=allocation, spec=runtime_spec):
    train(model, dataset)
```

Future dispatch/provider work can consume the same fragments, apply overrides, validate requirements, and then enter the runtime barrier in the correct worker process.
