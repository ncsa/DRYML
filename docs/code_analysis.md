# Code Analysis

`dryml.code` is the dependency-light local analysis foundation. It builds
immutable generic evidence without importing DRYML product packages or optional
frameworks. It does not define domain policy, persistence, transformation,
transport, or a registry of globally installed analyzers.

## Targets And Source

`analyze()` and `probe()` accept a Python function, bound Python method,
admitted callable instance, class, `DescriptorTarget`, `ImportTarget`,
`SourceTarget`, or an already normalized `CodeTarget`. Static analysis does not
invoke target bodies, class constructors, dynamic attribute hooks, arbitrary
descriptors, signature hooks, wrapper hooks, or implicit module imports.

`ImportTarget.path` has the exact grammar `module` or `module:qualname`.
`module` is one or more dot-separated Python identifiers. `qualname`, when
present, is one or more dot-separated identifiers; empty segments and
`<locals>` are rejected. Only the module segment is imported, so module import
has its ordinary trusted Python import side effects. Qualified segments are then
traversed through module dictionaries or class MRO dictionaries without dynamic
lookup or binding.

`SourceTarget` is static-only. Its source is parsed for exactly one selected
function, class, or lambda subject, but is never reconstructed, compiled, or
executed. It is valid for static `analyze()` and `probe()` and rejected by
`trace()`. File-backed source is request-local `SourceInfo`; returned graph,
fact, trace, and diagnostic locations use module-like names or basenames rather
than raw filesystem paths.

All imported modules, supplied target values, kernel classes, and invoked
callables are trusted inputs. This API is a correctness boundary for ordinary
callers, not a sandbox or safe-deserialization facility.

## Public Type Contracts

The public aliases describe closed framework-owned values rather than runtime
conversion APIs. `TargetKind` and `DescriptorKind` enumerate normalized target
and descriptor categories, while `CodeTargetInput` names the admitted target
wrappers and live Python forms. Invalid values fail during carrier or target
validation; evaluating the aliases has no side effects.

`ProgramNodeKind` and `ProgramEdgeKind` enumerate the only graph node and edge
categories accepted by construction and query methods. `KernelMode` separates
static outputs from trace-derived outputs. Unknown values are rejected before
kernel execution, and these aliases do not build or mutate a graph.

`FactScalar` and recursive `FactValue` define finite immutable payloads made of
exact scalars, tuples, or sorted unique string-keyed tuple mappings.
`AnalysisErrorCode` enumerates stable machine-readable failure categories.
Carrier validation raises `ValueError`; analysis operations expose the
documented `CodeAnalysisError` subclasses. These aliases retain no target,
source, exception, or runtime state.

## Graphs And Kernels

`build_program_graph()` produces a closed immutable `ProgramGraph`: a target
node plus static syntax, lexical-symbol, attribute-access, and static-call
evidence. `trace()` can derive a second graph by adding trace events. Graph
nodes and edges use fixed vocabularies, canonical ordering, sanitized locations,
and stable digests. The program graph is evidence, not an executable plan.

A kernel DAG is separate from the program graph. Each request submits explicit
`KernelCall` values; there is no process-global registry. A kernel declares its
input/output types, accepted target kinds, required producer kernels, mode, and
optional conservative fusion eligibility. The scheduler validates the complete
DAG before execution, preserves submission-order outcomes, and exposes only
declared successful dependency artifacts through `KernelContext.require()` and
`KernelContext.facts()`.

`AnalysisResult` reports succeeded, failed, and skipped kernel outcomes. A
failed or skipped output is unavailable to `require()` and raises
`MissingOutputError`; `output()` returns `None` for both absence and a valid
`None` output. Structural target, source, graph, declaration, dependency, and
trace-admission errors raise a typed `CodeAnalysisError`. Errors after graph
construction are represented as redacted diagnostics and partial results where
supported. Diagnostics omit arbitrary object representations, raw exception
text, absolute paths, and recognizable secrets.

Framework facts are `CodeFact` or `CodeFacts` values with a closed immutable
payload grammar. Other kernel outputs are opaque consumer artifacts. Analysis
results, facts, graphs, and diagnostics are ephemeral and may be nonserializable;
they define no Store, record, recovery, or transport authority. A consumer that
needs persistence owns a separately designed projection and its lifecycle.

Traversal fusion is conservative. The scheduler only fuses verified static
`TraversalKernel` instances that retain the inherited traversal template, have
no unverified instance state, are independent of one another, and have only
safe dependency evidence. Declaring `fusion_safe=True` is a cooperative claim,
not permission to mutate graph data, artifacts, inputs, or shared state.

The built-in lexical kernel provides generic free-name evidence only. The one
approved dependency direction is a one-way lazy call from `dryml.core.symbol`
to `dryml.code.algorithms.lexical_dependencies`. `dryml.code` does not import
core, product packages, or dispatch policy.

## Limits

Static analysis is intentionally branch-insensitive. It makes no
branch-aware control-flow, data-flow, alias, or whole-program analysis claim.
It does not resolve names or prove runtime behavior. Transformation APIs are
absent; consumer transformations remain deferred.

`probe()` is only the ordinary in-process static convenience entry point. It
does not create subprocesses, serialize requests or results, execute target
bodies, or apply environment policy. Future process isolation, transport, and
execution policy belong to `dryml.execute`; candidate coordination and
submission policy belong to `dryml.dispatch`.

## Trace

`trace()` supports only a live synchronous Python function or bound Python
method. It validates the request and builds the base graph before target
invocation, runs static kernels, invokes the target exactly once, and then runs
trace-mode kernels over derived event evidence. It rejects classes, callable
instances, `SourceTarget`, source reconstruction, coroutines, generators, async
generators, and native-only callables before invoking target or kernel code.

Trace capture is current-thread only and includes the root Python frame and its
descendant Python frames. `max_events` must be a non-boolean integer from 1
through 100,000, inclusive. On overflow capture stops immediately while the
ordinary target invocation continues; every trace-mode outcome fails with
`trace.limit`, has no output or facts, and cannot satisfy `require()`.

An existing current-thread trace hook is rejected. DRYML's temporary hook does
not compose with another hook and is restored in `finally` after every ordinary
success or failure. Ordinary target failures become a redacted invocation
outcome and preserve completed static artifacts. An interruption-style failure
cleans up and re-raises without returning a result. If hook cleanup itself
fails, DRYML makes a best-effort attempt to disable its hook and raises a typed
`trace.cleanup` error without a result; that cleanup error takes precedence.

There is no same-process concurrency guarantee for tracing: the API has no
lock, registry, or coordination mechanism. Callers that need parallel analysis
must use separate processes. Trace evidence retains only immutable event name,
opaque code identity, sanitized source location, and coverage metadata, never
frames, locals, globals, arguments, return values, exceptions, tracebacks, or
live target handles.
