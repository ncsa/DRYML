# DRYML Requirement And Default Annotations

`dryml.annotations` is the generic sidecar metadata layer for planning. It lets code declare environment, world, and runtime intent without creating DRYML Objects, activating runtime, importing heavy frameworks, or changing `ConcreteDefinition` identity.

## Public Decorators

Use the Pythonic facades for normal code:

```python
import dryml

@dryml.env.req(packages={"torch": ">=2.4"}, tags=("training",))
@dryml.env.default(dryml.environments.CurrentEnvironmentSpec())
@dryml.world.req(accelerators={"gpu": {"min": 1}})
@dryml.world.default(cpus=8, memory="32GiB", accelerators={"gpu": 1})
@dryml.runtime.default(torch={"num_threads": 8}, env={"OMP_NUM_THREADS": "8"})
def train(model, dataset):
    ...
```

`dryml.env.req(...)` declares hard software constraints such as Python versions, package requirements, capabilities, and tags. `packages={"torch": ">=2.4"}` becomes the PEP 508 requirement `torch>=2.4`; `packages={"torch": None}` becomes `torch`.

`dryml.env.default(...)` declares an overrideable environment candidate. Pass an
environment spec object or its JSON-ready mapping, for example
`dryml.environments.CurrentEnvironmentSpec()`. Dispatch selects an explicit
`environment=` candidate before this annotation default; the selected candidate
must still satisfy any hard environment requirements. The decorator attaches
metadata only and does not activate an environment or runtime.

`dryml.world.req(...)` declares hard resource and topology constraints. The default role is `main`; pass `roles={...}` for a full multi-role requirement payload.

`dryml.world.default(...)` declares an overrideable requested world shape. Defaults are suggestions and can be replaced by user overrides, but the final world spec must still satisfy hard world requirements.

`dryml.runtime.default(...)` declares process-local runtime defaults such as framework settings, device visibility policy, limits, environment overrides, and metadata. Runtime defaults are mode-neutral unless `mode=` is explicit; dispatch or an explicit activation chooses worker/probe/inline mode. The decorator does not call `enter_runtime(...)`, `activate(...)`, mutate `os.environ`, or import torch, TensorFlow, or JAX.

## Managed Direct Calls

Hard environment and world requirements on supported decorated callables are also
checked before direct user-code entry in a managed `dryml.session`. Python mode
intentionally bypasses session-derived checks, and unannotated direct code has
only the process-level controls DRYML established. Functions, methods,
static/class methods, managed descriptors, and explicitly decorated construction
methods are supported. Class-object/custom-metaclass calls, properties and other
custom descriptors, post-decoration assignment, and references captured before
decoration are not automatic boundaries. Dispatch and tracing use an internal
scoped bypass only while reconstructing their own target; worker code checks its
own allocation.

## Generic API

The low-level decorators attach `AnnotationFragment` objects directly:

```python
import dryml.annotations as ann

@ann.require(namespace="environment", fragment={"requirements": ["numpy>=1.26"]})
@ann.default(namespace="runtime", fragment={"frameworks": {"plain": {"num_threads": 4}}})
def analyze():
    ...
```

Fragments are collected with `own_fragments(...)`, `fragments_for_method(...)`,
`fragments_for_definition_method(...)`, and `collect_fragments(...)`. Class MRO
fragments are base-to-subclass; a method's fragments come from its concrete
implementation, so an override does not inherit base method fragments by default.
Both classmethod and staticmethod decorator orders are supported. Definition/CDef
method collection resolves the class without building the object. Provider/probe
code can participate by passing synthetic `AnnotationFragment` instances through
`provider_fragments=`.

Generic mapping fragments support `merge_policy="merge"` by default. The small policy set for early provider/default composition is `merge`, `replace`, `append`, and `error_on_conflict`. Environment requirements use namespace-specific `base`, `add`, and `override` semantics instead of the mapping policy set.

## Resolution

Use resolution helpers to merge metadata without executing it:

```python
result = ann.resolve(train, overrides={"world": {"roles": {"main": {"process": {"resources": {"cpus": 4}}}}}})
if not result.report.ok:
    print(result.report.explain())
```

`resolve_target_requirements(...)`, `resolve_method_requirements(...)`, and
`resolve_definition_method_requirements(...)` return the authoritative
`RequirementResolution`, including fragments and source traces.
`resolve_requirements(...)` merges hard requirements. `resolve_defaults(...)`
merges defaults and applies user overrides. `resolve(...)` validates the final
default world spec against hard world requirements and returns structured
`AnnotationIssue` entries with source traces. Dispatch consumes this resolution;
it owns candidate selection and launch policy.

Runtime requirement fragments can be collected through the low-level
`namespace="runtime"` API. Public `dryml.runtime.default(...)` declares a soft
runtime default; runtime compatibility remains checked by dispatch/runtime policy
rather than by the decorator itself.

Overrides are applied after defaults. Empty mappings in overrides replace the corresponding default mapping, so `{"frameworks": {}}` clears runtime framework defaults. For explicit nested control, use `{"$replace": value}` to replace a subtree or `{"$delete": True}` to delete a key.

## Legacy Environment Decorators

The existing `dryml.environments.req`, `add_req`, `override_req`, `fragments_for_class`, and `requirements_for_class` APIs remain supported. Legacy environment decorators also attach equivalent generic environment requirement fragments so the annotation resolver can see them. The generic resolver preserves legacy `override_req` replacement behavior while returning structured source traces.

## Argument-Role Annotations

`RefCDef`, `RefCDefArg`, `SelectorArg`, `ValueArg`, and `ArgRole` remain constructor argument-role annotations. They affect argument canonicalization and can affect `ConcreteDefinition` identity. They are re-exported from `dryml.annotations` and `dryml.annotations.arg_roles` for discoverability, but they are not serialized as `dryml.annotation.v1` specs and are not collected by requirement/default resolution.

## Runtime Barrier

Annotation decorators only declare intent. Explicit runtime activation remains available through `dryml.runtime.activate(...)` for backend, worker, inline, and power-user setup:

```python
with dryml.runtime.activate(mode="worker", allocation=allocation, spec=runtime_spec):
    train(model, dataset)
```

Dispatch can consume the same fragments, apply overrides, validate requirements,
and then enter the runtime barrier in the correct worker process.

## Dynamic Provider Fragments

`dryml.providers` supplies target-environment probe reports for dynamic facts that require importing provider, framework, or user code. Probe reports expose ordinary `AnnotationFragment` objects through `ProbeReport.annotation_fragments(...)`; pass those fragments via `provider_fragments=` to reuse the same merge and conflict reporting path as static decorators. See `docs/providers.md` for the subprocess protocol, record shape, and cache hooks.
