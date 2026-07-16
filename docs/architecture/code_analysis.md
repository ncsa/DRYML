# Code Analysis Architecture

## Status

Current architecture for the shipped reusable `dryml.code` analysis API.

## Current State

`dryml.code` contains a fact-oriented analysis layer plus compatibility helper modules for callable inspection, source extraction, and AST access collection. `dryml.core2.symbol` owns import references and source-backed symbol references. `dryml.core2.methods` owns method semantic objects such as `Method`, `Traits`, `CompilerInfo`, and the `traits` decorator. The helper modules now delegate to analyzer implementations under `dryml.code.algorithms` or re-export core semantic names for compatibility.

## Problem Statement

Dispatch, probes, annotations, and later analyzers need shared answers about Python targets. Those answers should not be duplicated inside dispatch or hidden inside annotation merging.

## Guiding Principle

`dryml.code` should collect reusable algorithms that discover facts about code. It should not decide where operations run. Dispatch should consume code facts when selecting and checking candidates. Probes should be able to run code-analysis algorithms outside the orchestrator when useful.

## What Belongs in dryml.code

- Callable identity and signature inspection.
- Importability analysis.
- Source location and source text extraction.
- Source-backed fallback data for non-importable functions/classes.
- Symbol/import dependency discovery.
- Direct annotation-fragment discovery and serialized annotation requirement-resolution metadata as facts.
- Method contract facts.
- Shape hints where they are code-derived.
- AST access and method-call hints.
- Explicit opt-in dynamic trace call facts.
- Structured diagnostics.

## What Does Not Belong in dryml.code

`dryml.code` should not select environments, allocate worlds, enforce runtime policy, launch workers, or decide dispatch compatibility. Those responsibilities belong to dispatch, worlds, runtime, and provider/probe layers.

## Fact-Oriented API

`dryml.code.analyze(...)` returns a structured `CodeAnalysisResult`. The result
contains facts and diagnostics rather than a launch decision. Facts are
serializable through `to_data()` so dispatch and probes can persist or pass them
between processes.

Core public types:

- `CodeTargetSpec`: serializable target representation. It records descriptive target kind, optional import path, optional source spec, optional method metadata, and JSON-compatible metadata.
- `CodeTarget`: local analysis wrapper. It may hold live Python objects and is not used for serialized output.
- `CodeFact`: generic fact record with `kind`, `source`, and `data` fields.
- `DiagnosticFact`: structured diagnostic with severity, code, and message.
- `RequirementFact`: raw requirement/default annotation fragment fact. It preserves namespace, kind, priority, merge policy, fragment data, annotation source trace data, and serialized `RequirementResolution` data without selecting dispatch candidates.
- `CodeAnalysisContext`: analysis options such as selected algorithms, source/import permissions, annotation inclusion, method-contract inclusion, and diagnostics policy.
- `CodeAnalysisResult`: aggregate target, facts, diagnostics, `ok`, filtering helpers, and JSON-compatible serialization.

`CodeAnalysisContext.metadata` is analysis-run metadata supplied by the caller. During local analysis it is copied into normalized target metadata so serialized results can retain caller provenance, but analyzers should not interpret it as dispatch requirements or make execution decisions from it.

Example:

```python
import json
import dryml.code as code

result = code.analyze(run_training, algorithms=("callables", "source", "direct_annotations"))
callable_facts = result.facts_of_kind("callable")
json.dumps(result.to_data())
```

Built-in analyzers are registered by name:

- `callables`: callable identity, signature, and importability facts.
- `source`: source text and source-location facts.
- `ast_access`: static attribute-access and method-call-like hints.
- `symbol_capture`: `ImportRef`/`SourceSpec`-style symbol facts using `dryml.core2.symbol`.
- `direct_annotations`: raw annotation and requirement facts using the authoritative `dryml.annotations` collection/resolution APIs.
- `method_contracts`: minimal DRYML `Method` contract metadata from `dryml.core2.methods`.
- `static_calls`: opt-in conservative resolution for direct globals and direct
  methods on concretely annotated parameters. It is not in either default
  analyzer tuple.

Analyzer failures become `DiagnosticFact(error)` by default. Setting `CodeAnalysisContext(diagnostics_policy="raise")` raises a `CodeAnalysisError` instead.

Compatibility imports remain available:

- `dryml.code.callable_info.CallableInfo`
- `dryml.code.callable_info.analyze_callable`
- `dryml.code.source.SourceInfo`
- `dryml.code.source.get_source_info`
- `dryml.code.source.func_source_extract`
- `dryml.code.ast_tools.AccessCollector`
- `dryml.code.ast_tools.collect_accesses_from_source`

## Relationship to core2.symbol

`core2.symbol` already provides stable `ImportRef` and `SourceSpec` primitives. `core2.methods` provides stable method semantic primitives. `core2` must not depend on `dryml.code`; `dryml.code` may depend on `core2`. This keeps the core semantic model independent of higher-level analysis algorithms.

## Relationship to Method and Method Handles

`Method`, `Traits`, `CompilerInfo`, `BatchMode` re-export convenience, and the `traits` decorator now live under `dryml.core2.methods`. `dryml.code` re-exports these names for compatibility, but code-analysis algorithms only inspect method facts; they do not own the semantic model. Future method handle or signature semantic APIs should also live under `dryml.core2.methods` if they are introduced.

Preferred import:

```python
from dryml.core2.methods import Method, Traits, CompilerInfo, traits
```

Compatibility imports remain supported:

```python
from dryml.code import Method, Traits, CompilerInfo, traits
```

## Relationship to dispatch and probes

Dispatch should ask `dryml.code` for code facts and then apply requirement/candidate logic. Code probes should reuse the same algorithms in a lightweight `RuntimeMode.PROBE` process when orchestrator-local analysis is insufficient or risky.

`dryml.code` emits facts and diagnostics only. Dispatch consumes accepted facts
under its explicit policy, while `dryml.annotations` remains the merge authority.
Analysis itself does not select environments, allocate worlds, enforce runtime
policy, launch workers, or decide candidate compatibility.

## Sprint 9A Analysis Contract

Static analysis and dynamic tracing share the existing `CodeAnalyzer`
protocol and `CodeAnalysisResult` model. `analyze(...)` directly runs selected
analyzers and never intentionally invokes the submitted target body. Importing an
import-path target can still execute module-level code. After import, qualname
components are inspected through static module/class dictionaries so descriptor,
metaclass, and module dynamic-attribute hooks are not invoked.

`probe_target(...)` only selects execution location for the same analyzers. An
inline probe enters `RuntimeMode.PROBE` but does not imply a new OS process. A
worker process may use only analyzers installed and registered in that worker.

Sprint 9B ships this invocation-bearing facade:

```python
dryml.code.trace(
    target,
    *,
    args=(),
    kwargs=None,
    context=None,
    policy=None,
) -> CodeAnalysisResult
```

`trace(...)` is the invocation-bearing API and requires
`CodeAnalysisContext.allow_dynamic_execution=True`. Its invocation data is
explicit rather than hidden in metadata. Trace facts use a distinct fact
kind, inline live notebook targets are supported use cases, and subprocess or
cross-environment tracing is not part of this contract. The facade runs exactly
the `dynamic_trace` modality; selecting that analyzer through `analyze(...)` or
`probe_target(...)` remains non-invoking and returns
`dryml.code.dynamic_trace_requires_trace_facade`.

### Target and Location Support

The user-facing support matrix is intentionally narrower than a generic table of
implementation mechanisms:

| Target/path | Direct `analyze` | Isolated/timeout probe | Selected environment probe | `trace` | Opt-in dispatch trace |
|---|---|---|---|---|---|
| Live module function | Supported | Supported when importable/reconstructible | Supported when importable there | Supported for an exact synchronous function and supported arguments | Supported under the narrower dispatch grammar |
| Live notebook/local function | Supported inline | Unsupported without reconstructible target | Unsupported | Supported inline under trace policy | Supported only when normal dispatch transport is valid, including explicit same-Python pickle rules |
| Import-path target | Supported with import permission | Supported | Supported | Unsupported without exact live function | Unsupported |
| Bound method | Supported by ordinary direct analyzers | Not generally reconstructible | Not generally reconstructible | Unsupported | Unsupported |
| Source-spec-only target | Static/source facts where implemented | No source-backed reconstruction | Unsupported | Unsupported | Unsupported |
| Definition/CDef method dispatch | Direct class/method requirement collection | Existing static/probe behavior only | Existing static/probe behavior only | A live orchestration function may accept Definition proxies | Method-target tracing unsupported; ordinary method dispatch supported |

`analyze(...)` does not intentionally invoke a target body. `trace(...)` runs
trusted code once in the current process; it is not a sandbox and has no hard
timeout. A probe changes runtime role/location, not analyzer semantics.
Dispatch tracing is explicit and default-off; requested structural trace failures
block planning rather than being ignored by compatibility policy. Source-backed
subprocess reconstruction is not implemented.

| Input | Direct `analyze` | Inline `probe_target` | Subprocess probe | `static_calls` |
|---|---|---|---|---|
| Live module-level importable function | Supported | Supported | Stable import path only | Supported when source exists |
| Import path string/spec | Supported when imports are allowed | Supported | Supported | Supported after import/source retrieval |
| Live notebook or `__main__` function | Supported | Supported without timeout | Unsupported | Supported when source is available |
| Live local function/closure | Supported | Supported without timeout | Unsupported | Supported without inferring captured values |
| Live bound method | Supported inline | Supported without timeout | Unsupported because receiver state is not transported | Supported from the live method source |
| Importable class or unbound method | Supported | Supported | Stable import path only | Supported when source exists |
| Source-spec-only target | Descriptive only | Descriptive only | Unsupported | Unavailable until reconstruction exists |
| Unknown/non-callable target | Structured normalization diagnostic | Structured normalization diagnostic | Unsupported | Not applicable |

| Requested location | Import-path target | Live non-importable target | Source-spec-only target | Timeout guarantee |
|---|---|---|---|---|
| Direct `analyze(...)` | Inline import and analysis | Inline analysis | Descriptive only; live-object analyzers report unavailable | None |
| `probe_target(..., environment=None, timeout=None)` | Inline probe runtime | Inline probe runtime | Descriptive only; no reconstruction | None |
| Current Python with finite timeout | Worker subprocess | Structured rejection | Structured reconstruction-unavailable rejection | Managed process deadline |
| `PythonExecutableSpec` | Worker subprocess | Structured rejection | Structured reconstruction-unavailable rejection | Managed process deadline when configured |
| Supported `CondaEnvironmentSpec` | Worker subprocess | Structured rejection | Structured reconstruction-unavailable rejection | Managed process deadline when configured |
| Container or remote environment | Existing unsupported diagnostic | Unsupported | Unsupported | Not applicable |

Workers independently require a stable import path, including requests sent
directly to the worker protocol. A source spec is serializable provenance data;
it does not reconstruct closures, notebook frames, bound instances, or arbitrary
dependencies. Custom analyzers registered only in the orchestrator are not
assumed to be available in a worker.

### Static Facts and Bounds

`ast_access` emits syntactic `ASTAccessFact` and `CallSiteFact` records. Every
call-site fact says `semantic_resolution="not_attempted"`; nested receivers such
as `obj.child().train()` remain partial rather than flattened through the call
result. Locations include source-relative line, file-absolute line when known,
and column offset.

`static_calls` emits one `StaticCallFact` per inspected `ast.Call` and one
`static_call_summary` fact after traversal. A resolved fact has
`status="resolved"`, `confidence="exact_static"`, and a fixed target mapping
containing only `kind`, `import_path`, `method_name`, and `subject_ref`.
Unresolved, ambiguous, and unsupported facts use `confidence="conservative_hint"`.
They are static possibilities, not runtime observations or dispatch requirements.

Supported resolution forms are safely described plain Python or builtin
functions and ordinary classes from the analyzed function's real globals mapping
and direct methods on direct parameters with an ordinary concrete class
annotation, inspected without descriptor binding. Importable resolved targets
retain a verified import path. Inline-only globals and annotated methods instead
retain a bounded source-level `subject_ref`; their null import path does not make
them worker-runnable. Lambdas and bound builtin methods remain conservative
because their identity depends on live state that the fixed target mapping cannot
describe defensibly.
String annotations, unions, generics, protocols, aliases, reassignment, nested
scopes, attribute chains, call-result receivers, properties, dynamic `getattr`,
callable instances, non-standard metaclasses, and control-flow inference do not
resolve. This restriction avoids dynamic attribute hooks while constructing facts.

Both static analyzers enforce these limits before unbounded fact expansion:

| Dimension | Limit |
|---|---:|
| UTF-8 source bytes | 1,048,576 |
| AST nodes | 100,000 |
| Call sites | 10,000 |
| Attribute/call chain components | 64 |
| Semantic-resolution diagnostics | 1,000 |
| Serialized display/reference scalars | 4,096 characters |

Source, AST-node, call-site, and target-reference hard-bound exhaustion returns
an error diagnostic with `limit_name`, `limit`, and `observed_lower_bound`;
static-call summaries then set `complete` to false. Chain exhaustion produces a
per-site unsupported fact, while non-target display scalar overflow uses a
bounded replacement and can leave the summary complete.
`static_calls` represents per-site resolution outcomes as bounded facts rather
than one diagnostic per site, so it does not emit an unbounded stream of
semantic-resolution diagnostics.
Source unavailable, source disabled, parse failure, and no matching call remain
distinct outcomes.

### Static Fact Examples

Every serialized `StaticCallFact` has this fixed source and data schema. These
examples omit only ordinary line-number variation:

```json
{"kind":"static_call","source":{"analyzer":"static_calls","target_kind":"function","filename":"example.py"},"data":{"status":"resolved","confidence":"exact_static","syntax":"direct_name","display":"helper","receiver":null,"method_name":"helper","target":{"kind":"function","import_path":"example:helper","method_name":null,"subject_ref":null},"reason":null,"relative_line":2,"absolute_line":12,"col_offset":4}}
{"kind":"static_call","source":{"analyzer":"static_calls","target_kind":"function","filename":"example.py"},"data":{"status":"unresolved","confidence":"conservative_hint","syntax":"direct_name","display":"missing","receiver":null,"method_name":"missing","target":null,"reason":"global_name_unavailable","relative_line":2,"absolute_line":12,"col_offset":4}}
{"kind":"static_call","source":{"analyzer":"static_calls","target_kind":"function","filename":"example.py"},"data":{"status":"ambiguous","confidence":"conservative_hint","syntax":"annotated_receiver_method","display":"model.train","receiver":"model","method_name":"train","target":null,"reason":"non_concrete_annotation","relative_line":2,"absolute_line":12,"col_offset":4}}
{"kind":"static_call","source":{"analyzer":"static_calls","target_kind":"function","filename":"example.py"},"data":{"status":"unsupported","confidence":"conservative_hint","syntax":"attribute_chain","display":"helpers.train","receiver":"helpers","method_name":"train","target":null,"reason":"attribute_chain_unsupported","relative_line":2,"absolute_line":12,"col_offset":4}}
```

### Diagnostics and Safety Evidence

| Condition | Severity | Diagnostic code |
|---|---|---|
| Source disabled | info | `dryml.code.source_disabled` |
| Source unavailable | warning | `dryml.code.source_unavailable` |
| Parse failure | error | `dryml.code.ast_parse_failed` |
| Static source/AST/call bound | error | `dryml.code.static_<limit>_limit_exceeded` |
| Oversized static target reference | error | `dryml.code.static_target_reference_limit_exceeded` |
| Source-spec worker request | error | `code_probe.source_spec_reconstruction_unavailable` |
| Unstable worker target | error | `code_probe.non_serializable_target` |
| Live bound method in a worker | error | `code_probe.bound_method_receiver_unavailable` |
| Invalid worker timeout | error | `code_probe.invalid_timeout` |

`tests/code/test_static_calls_algorithm.py` proves global callables, methods,
properties, callable instances, annotation mappings, wrapper metadata, and
metaclass lookups are not invoked. `tests/code/test_callable_algorithm.py` proves
callable analysis avoids hostile target, wrapper, and metaclass metadata hooks.
`tests/code/test_targets.py` proves
import-path and bound-method normalization do not bind descriptors, invoke
module/metaclass hooks, or truth-test raw descriptors during analysis.
`tests/code/test_probe_target.py` proves probes do not execute submitted target
bodies or instantiate classes.

## Sprint 9B Dynamic Trace Contract

### Explicit facade and trust boundary

Tracing is deliberately separate from ordinary analysis:

```python
import dryml.code as code

result = code.trace(
    orchestration_function,
    args=(model_definition,),
    context=code.CodeAnalysisContext(allow_dynamic_execution=True),
    policy=code.DynamicTracePolicy(max_calls=100),
)
```

The facade validates a private typed invocation request and calls the dedicated
runner in `dryml.code.algorithms.dynamic_trace`; it does not place invocation
objects in context metadata, target metadata, analyzer state, or probe transport.
Tracing executes trusted target code once in the caller's current process. It is
not a sandbox, has no hard timeout, and does not prevent imports, I/O, mutation,
threads, subprocesses, or other target side effects. `analyze(...)` and every
probe location remain non-invoking, even when `dynamic_trace` is selected.

`DynamicTracePolicy` is frozen and slotted. `max_calls` accepts exact integers
from 1 through 10,000 (not `bool`); `require_proxy_only_args` and
`collect_requirements` accept exact booleans. Untyped policy mappings are not
coerced. By default all invocation leaves must be Definitions/CDefs. Setting
`require_proxy_only_args=False` additionally permits bounded `None`, exact bool,
integer, finite float, and string leaves.

Those scalar leaves are exact native built-in values: subclasses of `int`,
`float`, `str`, and `bytes` are unsupported by bounded Definition/CDef identity
generation and are rejected before a custom representation or conversion hook
can run.

### Target and execution-location support

| Target | Direct `trace(...)` | Notes |
|---|---|---|
| Live module synchronous Python function | Supported inline | Invoked once |
| Live notebook/`__main__`, local function, closure, lambda | Supported inline | Uses the live object; no reconstruction |
| Import-path synchronous Python function | Supported inline when imports are allowed | Import can execute trusted module code |
| `CodeTarget` wrapping a supported function | Supported inline | Preserves live target/spec |
| Unbound method function | Supported as an ordinary function | Caller supplies every explicit argument |
| Source-spec-only target | Unsupported | No reconstruction |
| Bound method, callable instance, class, builtin | Unsupported | Receiver/construction/call state is not invoked |
| Async function/generator or generator function | Unsupported | No coroutine/generator is created |
| Unknown/non-callable value | Unsupported | Structured target diagnostic |

| Requested path | Dynamic invocation | Isolation/timeout |
|---|---|---|
| Direct `trace(...)` | Supported in current process | Cooperative; no hard timeout |
| `analyze(..., algorithms=("dynamic_trace",))` | Never | Requires-trace-facade diagnostic |
| Inline `probe_target(..., algorithms=("dynamic_trace",))` | Never | Existing inline probe role only |
| Current-Python timeout worker | Never | Existing probe routing/diagnostics |
| Python executable or Conda probe | Never | Existing probe routing/diagnostics |
| Container or remote probe | Never | Existing unsupported probe behavior |

The locked trace signature has no location or timeout parameter. Subprocess and
selected-environment tracing are not implied.

### Invocation grammar and receiver observation

Exact built-in list, tuple, and `dict[str, value]` containers are recursively
copied. Cycles and non-string keys are rejected before execution; mutable
container aliases and repeated Definition/CDef proxy identity are preserved in
the invocation copy. Caller containers are not mutated. Custom mappings,
sequences, iterators, bytes, sets, dataclasses, DRYML Objects, NumPy values, and
arbitrary instances are rejected without generic repr serialization.

Receiver classes may be live classes, `ImportRef` values when imports are
allowed, or trusted `SourceSpec` values when dynamic execution, imports, and
source are all allowed. Resolution never builds the Definition/CDef. Proxy
attribute lookup walks static class dictionaries and supports exact Python
functions plus staticmethod/classmethod descriptors containing exact functions.
It does not bind or execute real methods, properties, custom descriptors,
metaclass hooks, or dynamic attribute hooks. Missing, non-method, and dunder
attributes abort with a structured diagnostic.

Definition observations have one of these forms:

```json
{"definition_kind":"definition","definition_ref":"<bounded-stable-hash>"}
{"definition_kind":"concrete_definition","definition_ref":"cdef-v4-<bounded-stable-hash>"}
```

These are bounded correlation/candidate keys, not exact equality proofs. They
must not be used alone for equality, requirement merging, authorization, or
dispatch. Exact equality needs a live structural comparison or a separately
owned registry that already verified equality. Importable receiver classes use a
statically verified `module:qualname`; inline classes use `null`.

### Facts, method facts, and outcomes

Each accepted call emits a typed `DynamicCallFact` with exactly this schema:

```json
{"kind":"dynamic_call","source":{"analyzer":"dynamic_trace","target_kind":"local_function"},"data":{"sequence":0,"receiver_kind":"concrete_definition","receiver_ref":"cdef-v4-0123456789abcdef","receiver_class":null,"method_name":"train","args":[],"kwargs":{},"method_facts":[]}}
```

The full source/data schema is validated before generic recursive conversion.
It contains no live callable, class, descriptor, Definition/CDef, proxy, source,
traceback, frame, local, arbitrary repr output, or exception message.

When both policy and context permit annotations, observed methods collect class
and concrete-method fragments through `dryml.annotations` and preserve current
`AnnotationFact`/`RequirementFact` resolution data. `collect_requirements=False`
or `include_annotations=False` omits those facts. Applicable core2 Method facts
are controlled independently by `include_method_contracts`. Their trait
selectors use a fixed `{"backend": string|null, "batch_mode": string|null}`
mapping rather than a Python representation; malformed selector metadata fails
method-fact collection closed. Nested requirement facts validate exact
`RequirementSourceTrace` and `RequirementResolution` wire forms and must match
their enclosing annotation. Nested shape facts, when present, use the
`method_contracts` source and fixed `input_handles`/`output_handles` array data
form. No facts are read
from legacy `__dry_compute_spec__`/`compute_reqs`, merged across calls, or used to
select an environment, world, runtime, or dispatch candidate.

Every run that starts execution adds exactly one `dynamic_trace_summary` with
`complete`, `outcome`, `calls_recorded`, and `max_calls`. A successful zero-call
run is therefore distinct from a pre-execution failure (no summary) and an
incomplete run. Outcomes are `complete`, `call_limit_exceeded`,
`unsupported_return_operation`, `unsupported_argument`,
`unsupported_receiver_attribute`, `stale_proxy`,
`method_fact_collection_failed`, `target_failed`, `result_limit_exceeded`,
`diagnostics_limit_exceeded`, and `algorithm_failed`. Expected failures retain
only prior complete call facts and use an incomplete summary.

Proxy methods return a private unsupported value rather than `None`. Ignoring,
assigning, passing through, or directly returning it is supported. Truth/length
testing, iteration, indexing/mutation, attribute chaining, calling, arithmetic,
comparison, awaiting, context management, numeric/bytes/index conversion,
hashing, and formatting abort with
`dryml.code.dynamic_trace_unsupported_return_operation`. Python identity tests
(`value is None`, `value is other`) cannot be intercepted and have no diagnostic
guarantee.

### Bounds and diagnostics

| Dimension | Hard limit |
|---|---:|
| Proxy calls | Policy maximum, hard ceiling 10,000 |
| Argument/container depth | 32 |
| Invocation or one-call container entries | 10,000 |
| String scalar | 4,096 characters |
| Integer magnitude | 4,096 bits |
| Method name | 512 characters |
| Receiver reference/class | 4,096 characters |
| Method facts per call | 256 |
| Serialized dynamic call | 1,048,576 bytes |
| Serialized calls plus summary | 16,777,216 bytes |
| Diagnostics | 256 |
| Diagnostic code/message | 1,024 characters |
| Definition hash depth (root 0) | 128 |
| Definition hash value occurrences | 100,000 |
| Definition hash edges | 200,000 |
| Definition hash digest-update bytes | 4,194,304 |

The bounded core2 hasher computes the existing stable digest in the same
budgeted traversal. It charges memo hits, nested identity values, mapping keys,
edges, and every digest update before exceeding a limit. Hard-limit diagnostics
include `limit_name`, `limit`, and `observed_lower_bound` when meaningful; no
truncated run reports complete.

| Condition | Diagnostic code |
|---|---|
| Permission disabled | `dryml.code.dynamic_trace_disabled` |
| Ordinary analyzer/probe path | `dryml.code.dynamic_trace_requires_trace_facade` |
| Conflicting context algorithms | `dryml.code.dynamic_trace_invalid_context` |
| Unsupported target | `dryml.code.dynamic_trace_unsupported_target` |
| Unsupported/bounded argument | `dryml.code.dynamic_trace_unsupported_argument`, `dryml.code.dynamic_trace_argument_limit_exceeded` |
| Receiver resolution/attribute failure | `dryml.code.dynamic_trace_receiver_resolution_failed`, `dryml.code.dynamic_trace_unsupported_receiver_attribute` |
| Foreign or escaped active proxy | `dryml.code.dynamic_trace_stale_proxy` |
| Call limit | `dryml.code.dynamic_trace_call_limit_exceeded` |
| Unsupported return operation | `dryml.code.dynamic_trace_unsupported_return_operation` |
| Method facts | `dryml.code.dynamic_trace_method_fact_collection_failed` |
| Result/diagnostic limit | `dryml.code.dynamic_trace_result_limit_exceeded`, `dryml.code.dynamic_trace_diagnostics_limit_exceeded` |
| Target exception | `dryml.code.dynamic_trace_target_failed` |
| Unexpected framework failure | `dryml.code.algorithm_failed` |

Target failures retain only a bounded exception type identity. Their message,
repr, traceback, frames, and locals are not serialized.

### State and cleanup

Each trace owns a unique planner, lock, proxy memo, accumulators, lifecycle, and
exact `ContextVar` token. The token is reset and the planner closed in `finally`
before failures or interruptions propagate. Nested traces restore the outer
planner; independent thread/task contexts do not share calls, limits, or
diagnostics. A foreign proxy aborts the currently active trace. A proxy used in a
thread with no copied current planner, after its owner closes, or from a copied
context retained after close raises fixed-message `DynamicTraceProxyError`
without mutating a returned result. The runner does not wait for target-spawned
work and does not promise deterministic order for unsupported concurrent child
work.

## Non-Goals

- This note does not implement source-spec subprocess reconstruction.
- This note does not add static-call dispatch policy or alter dispatch planning.
- This note does not add trace dispatch policy, subprocess/selected-environment
  tracing, hard-timeout isolation, arbitrary Python interpretation, or return
  proxies.

## Source Anchors

- `src/dryml/code/callable_info.py`
- `src/dryml/code/source.py`
- `src/dryml/code/ast_tools.py`
- `src/dryml/core2/methods/method.py`
- `src/dryml/core2/methods/traits.py`
- `src/dryml/core2/methods/compiler_info.py`
- `src/dryml/code/method.py` compatibility wrapper
- `src/dryml/code/traits.py` compatibility wrapper
- `src/dryml/core2/symbol.py`
- `src/dryml/core2/tensor_spec.py`

## Open Questions

Remaining Sprint 9B contract questions for Sprint 9A acceptance: none. The
following longer-term design questions remain outside the Sprint 9B invocation
contract locked here:

- Should facts be dataclasses, records specs, or both?
- Which diagnostics must be JSON-compatible in Sprint 1?
- How much source-backed fallback should be accepted before probes are required?
- Should later probe/dispatch metadata use a source-text policy such as `metadata_only`, `include_text`, or `hash_only` instead of always serializing full source text?

## Deferred Work

Alias-aware static resolution, alias provenance, and general Python call tracing
remain deferred under [ADR 0008](../adr/0008-deferred-alias-aware-code-analysis.md).
