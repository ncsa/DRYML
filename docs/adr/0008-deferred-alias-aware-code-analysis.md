# ADR 0008: Deferred Alias-Aware Code Analysis

## Status

Accepted as deferred on 2026-07-16. No implementation sprint is scheduled.

## Context

DRYML has two distinct code-analysis modes:

- `static_calls` inspects source without executing it; and
- `dynamic_trace` executes one trusted synchronous function with Definition
  proxies and records supported proxy-method calls.

The word *alias* can describe several different behaviors. A local receiver
alias already works during dynamic tracing because ordinary Python assignment
retains the same proxy object:

```python
def target(model):
    alias = model
    alias.train()
```

The call to `train()` is recorded. Direct `dryml.code.trace()` invocation also
preserves repeated Definition and exact-container identity within its bounded
invocation grammar. Dispatch has a narrower canonical input contract and rejects
aliased or cyclic input containers before tracing; local assignments made by the
target after invocation are not input-container aliases.

Static analysis deliberately resolves only direct global calls and direct
methods on direct parameters with supported concrete annotations. It reports
local names as unsupported instead of following assignments:

```python
alias = helper
alias()

receiver = model
receiver.train()
```

Dynamic tracing executes an aliased Python function normally, but it records only
supported calls made through Definition proxies. It does not emit a fact for an
arbitrary function call merely because the function was reached through an alias.
Neither analysis mode records the local alias name or assignment chain.

The unsupported untracked `dryml.graph` prototypes explored simple assignment
propagation such as `x = model` and `y = x`. They did not define safe semantics
for branches, loops, reassignment, closures, mutation, descriptors, unpacking,
or persistent/versioned provenance. Their legacy `__dry_compute_spec__` and
last-wins requirement aggregation are not part of the current architecture;
`dryml.annotations` owns requirement merge semantics.

## Decision

Keep the following features unimplemented until a future proposal supplies a
concrete use case and a bounded contract:

1. Static callable aliases, such as `run = helper; run()`.
2. Static receiver aliases, such as `item = model; item.train()`.
3. Alias chains, destructuring, branch/loop joins, closure aliases, and other
   control-flow-sensitive alias analysis.
4. Alias provenance that records local variable names, assignment locations, or
   an assignment chain in static or dynamic facts.
5. General dynamic Python call tracing, including facts for arbitrary direct or
   aliased function calls rather than only supported Definition-proxy methods.
6. A dispatch transport or canonical identity grammar for shared/aliased input
   container graphs.

Do not represent these features as missing support for ordinary dynamic receiver
aliases. A target-local alias to an admitted Definition proxy already behaves as
normal Python and records the same method call as the original local name.

Do not restore the prototype `ExecutionPlan`, `AstCallSpec`,
`__dry_compute_spec__`, or dispatch-specific last-wins requirement aggregation.
Future work must use the established `dryml.code` fact protocol and leave
requirement semantics in `dryml.annotations`.

## Rationale

Static alias resolution needs explicit soundness rules. A local name may be
reassigned on one branch, captured by a closure, mutated in a loop, shadow a
global, or refer to a descriptor or callable object. Guessing through those
cases would turn conservative source facts into misleading resolution evidence.

Dynamic alias provenance requires instrumentation beyond the current proxy
boundary. Python object identity reveals that two values are the same proxy, but
does not reliably reveal which local variable name the target used. Capturing
assignment names or every Python call would require AST rewriting, bytecode or
frame tracing, or a broader proxy model. Each option changes execution behavior,
failure modes, performance, and the trusted-code boundary.

General call facts or alias graphs would also require bounded, versioned schemas;
redaction and persistence rules; overlap and nested-trace isolation; and an
explicit decision about whether dispatch consumes the facts. Those contracts
should not be inferred from a small prototype.

## Future Entry Criteria

A future mini-sprint or ADR should begin only when a concrete consumer needs one
of the deferred features. It must specify:

- which alias categories are supported and which fail closed;
- assignment, reassignment, branch, loop, closure, and mutation semantics;
- whether results are possibilities, runtime observations, or both;
- exact fact schemas, identities, limits, redaction, and serialization rules;
- descriptor-safe and non-invoking behavior for static analysis;
- instrumentation and exception behavior for dynamic analysis;
- nested, overlapping, interrupted, and stale-proxy lifecycle behavior;
- whether dispatch consumes the facts or only reports them; and
- focused direct-call, alias-chain, ambiguity, limit, and regression tests.

A staged implementation should prefer reusable analysis facts first. Dispatch
policy or requirement integration should be a separate explicit decision after
the analysis contract is stable.

## Consequences

Current direct static resolution remains conservative and predictable. Dynamic
Definition-proxy aliases continue to work without new APIs or metadata. Users
cannot request alias names, assignment chains, general Python call facts, or
static resolution through local aliases.

The untracked `src/dryml/graph/` directory remains unsupported local work and is
not a compatibility source. This ADR preserves its remaining alias-analysis idea
as design context without adopting its code or legacy compute-spec semantics.

## Source Anchors

- `src/dryml/code/algorithms/static_calls.py`
- `src/dryml/code/algorithms/dynamic_trace.py`
- `src/dryml/code/facts.py`
- `src/dryml/dispatch/normalize.py`
- `tests/code/test_static_calls_algorithm.py`
- `tests/code/test_dynamic_trace.py`
- `docs/adr/0001-code-analysis-boundaries.md`
- `docs/architecture/code_analysis.md`
