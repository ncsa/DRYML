# Hard Requirements

`dryml.requirements` supplies dependency-light mechanics for explicit hard
requirements. It owns declaration carriers, deterministic combination outcomes,
diagnostics, and an explicit admission barrier. `dryml.environments` and
`dryml.worlds` independently own their declaration values, semantic
combination, compatibility evidence, and admission meaning.

Declarations are passive and process-local. Applying a domain decorator attaches
metadata to a live target and returns that exact target unchanged. It never wraps
or calls the target, inspects a host, reserves resources, activates a runtime,
mutates a session, or selects a candidate.

## Declaration To Admission

The normal flow is:

1. Apply `dryml.environments.req(...)` or `dryml.worlds.req(...)` to a supported
   target.
2. Resolve one domain with `requirements_for(target)` or
   `requirements_for_method(owner, method_name)`.
3. Inspect the resulting `RequirementResult` and, when it has a value, ask the
   owning domain to check explicit evidence.
4. Inspect the domain report. Pass it to `require_admission(...)` only when an
   explicit fail-closed admission decision is required.

Environment and world resolution is independent. Each resolver collects only its
owner-qualified annotation key and returns only its domain value; using one does
not import, combine, or select the other.

```python
from dryml import env
from dryml.requirements import require_admission


@env.req(requirements=("numpy>=2",))
class Analysis:
    pass


result = env.requirements_for(Analysis)
if result.has_value:
    report = result.value.check(env.inspect_current())
    require_admission(report, operation="run analysis")
```

The concise `dryml.env` and `dryml.world` root attributes are lazy aliases for
the exact `dryml.environments` and `dryml.worlds` modules. They introduce no
singular implementation packages and preserve the plural owners' `current`,
`set_current`, `reset_current`, and `use` selector-state APIs. Selector state is
not a target-level requirement default.

## Shared API

`RequirementSource(label, *, module=None, qualname=None)` provides a bounded,
non-identifying explanation for a declaration. `RequirementDeclaration(value,
*, source=...)` carries one already-validated domain value. Sources are not
discovered from targets: omitted decorator sources use fixed labels, and explicit
source text is bounded, validated, and redacted before public retention.

`RequirementIssue(code, message, *, path=None, sources=())` is one structured
combination diagnostic. `RequirementReport(issues=())` is its immutable,
deterministic collection and exposes `ok`. `RequirementResult(value=None,
report=RequirementReport())` carries the outcome from combination and exposes
`ok` and `has_value`. `RequirementError`, `RequirementCombinationError`, and
`RequirementBarrierError` report invalid shared input, invalid combination
protocols, and denied explicit admission respectively.

`RequirementCombiner` is a domain protocol with
`combine(declarations) -> RequirementResult`. `combine_requirements(
declarations, *, combiner=...)` freezes declaration order, assigns safe
one-based source ordinals, validates the result, and delegates all semantics to
the supplied domain combiner. It does not interpret environment or world values.

There are exactly three legal result states: empty success, valued success, and
conflict failure.

- **Empty success:** no value and an OK empty report, meaning this domain found
  no declarations.
- **Valued success:** one complete domain value and an OK empty report.
- **Conflict failure:** no value and a non-OK report containing the discovered
  semantic conflicts.

Malformed declarations, corrupt annotations, unsupported values, oversized
input, and invalid combiner results raise before a partial result is exposed.
In particular, a value plus issues and a nonempty combination with no value and
an OK report are not legal outcomes.

Shared combination accepts at most 256 declarations. Reports accept at most
1,024 issues and 4,096 issue-source associations. Domain multi-declaration
combination preflights at most 1,024 canonical paths, 4,096 declaration-path
occurrences, and 1 MiB of caller-controlled text; projected diagnostics are
bounded to 4 MiB. Source labels are at most 256 characters, while source module,
qualified name, issue text, paths, and operation labels are at most 512
characters. These bounds make every accepted conflict report complete rather
than silently truncated.

## Explicit Admission

`AdmissionReport` is a structural protocol exposing `admission_ok`. It is
deliberately separate from a domain report's policy-dependent `ok` property.
`require_admission(report, *, operation=None)` returns only when
`admission_ok` is an exact `True`; malformed reports raise `RequirementError`,
and `False` raises `RequirementBarrierError` retaining the identical report and
optional operation label.

The barrier is effect-free. It does not run a protected operation, bind a target,
or mutate runtime, session, dispatch, or global state. Environment compatibility
can report policy-permissive `ok` under `warn` or `ignore`, but those reports do
not have fail-closed `admission_ok`; unevaluated, unknown, malformed, and
unavailable environment evidence cannot admit a hard requirement. World reports
retain their domain-owned all-constraints-satisfied admission meaning.

`EnvironmentRequirement.check(record, *, policy="compatible")` returns the
domain-owned `CompatibilityReport`; its `admission_ok` is separate from `ok` as
described above. `check_world_spec_satisfies_requirement(world, requirement)`
and `check_allocation_satisfies_requirement(allocation, requirement)` return
`WorldCompatibilityReport`, whose `admission_ok` preserves world ownership of
the all-constraints-satisfied decision. These checks evaluate supplied evidence;
they do not select a candidate, allocate resources, or activate a runtime.

## Domain Entry Points

Both domain decorators support compatible classes, standalone functions, method
definitions, `staticmethod` and `classmethod` descriptors, and supported custom
descriptors. They return their input unchanged. `requirements_for(...)` resolves
class inheritance or one direct target. `requirements_for_method(...)` combines
inherited class declarations with one method selected statically. With an
instance owner it uses the exact class, never reads instance state, binds a
descriptor, or invokes dynamic attribute hooks.

`dryml.environments.req(...)` constructs hard `EnvironmentRequirement` values.
Environment package declarations allow normalized names, version specifiers, and
record-evaluable markers. Direct URLs, extras, and markers using `extra` are
rejected because environment evidence cannot verify them. Every environment
iterable or mapping field accepts at most 64 entries; combination accepts at
most 64 declarations and retains bounded ordinalized source explanations in
`EnvironmentRequirement.details["sources"]`.

`dryml.worlds.req(...)` constructs hard `WorldRequirement` values. Omitted
flattened constraints are unconstrained: `@dryml.world.req(cpus=2)` does not add
a replica constraint. Supplying `roles=` chooses the complete multi-role grammar
and rejects simultaneous flattened fields rather than silently merging grammars.
A single valid world declaration passes through unchanged; multi-declaration
world combination uses the shared diagnostic bounds.

## Boundaries And Clean Break

Hard requirements are not defaults, preferred candidates, selection policy,
active runtime/session state, worker intent, dispatch coordination, or automatic enforcement.
Stage 4 does not perform code inference, run probes, acquire resources,
install packages, wrap direct calls, or persist declarations or reports. Runtime
declarations, all generic and domain defaults, candidate precedence, session
worker defaults, dispatch, and inference remain deferred to their owning work.

The retired environment fragment path is a clean break. `add_req`,
`override_req`, `RequirementFragment`, `requirements_for_class`, the fragment
module, and fragment schema constant have no compatibility alias, decoder,
migration, or import path. Fresh environment records omit only
`payload.dryml.schema_versions["environment_fragment"]`; their semantic IDs and
therefore record IDs change from the reduced payload, while the environment
record schema remains v1.1.
