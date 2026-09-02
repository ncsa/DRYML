# Methods

`dryml.methods` owns DRYML's logical callable IR. A `Method` is a CDef-backed
logical operation: its configured identity is stable, while its selected
implementation and preparation cache are local runtime details. This package
does not own code analysis, dispatch, managed lifecycle, persistence, records,
or backend warmup.

Import the public API from this package. `dryml.code.Method`,
`dryml.code.Traits`, and `dryml.code.traits` were removed without aliases.

```python
from dryml.core.backend import Backend
from dryml.core.tensor_spec import BatchMode, TensorSpec
from dryml.methods import Method, Traits, traits
```

`Backend` and `BatchMode` remain core vocabulary. The exact Methods public API
is `Method`, `MethodImplementation`, `Traits`, `traits`, `MethodCallNode`,
`MethodCallSignature`, `MethodCallMode`, `MethodCallNodeKind`, `MethodError`,
`ImplementationDeclarationError`, `ImplementationSelectionError`,
`PreparedCallMismatchError`, `SelectionFailureReason`, and
`SelectionTraitName`.

## Authoring And Direct Calls

A simple Method declares one ordinary `__call__` implementation. Its positional
and keyword arguments are logical arguments forwarded unchanged, and its return
value is the implementation result. `infer_output_spec(input_spec)` accepts one
normalized `SpecTree`, returns a normalized output `SpecTree` without executing
user or model code, and raises `NotImplementedError` if that pure contract is
not supplied.

```python
import numpy as np

from dryml.core import TensorSpec
from dryml.methods import Method


class AddOne(Method):
    def __call__(self, value):
        return value + 1

    def infer_output_spec(self, input_spec):
        return input_spec


method = AddOne()
assert method(np.array([1, 2])).tolist() == [2, 3]
assert method.infer_output_spec(TensorSpec("int64", shape=(2,))) == TensorSpec(
    "int64", shape=(2,)
)
```

Use either a direct `__call__` declaration or trait-decorated alternatives in a
class, never both. Direct declarations retain normal descriptor and cooperative
`super().__call__` behavior. Method identity excludes selected targets,
arguments, cache state, persistence state, and dispatch/lifecycle state.

## Closed Traits And Catalogs

`Traits(backend=None, batch_mode=None)` is an immutable closed declaration for
one implementation. `backend` accepts `Backend` or its string spelling;
`batch_mode` accepts `BatchMode` or `"element"`/`"batched"`. Omitted traits are
unspecified. Invalid values raise `ValueError` or `TypeError`; invalid decorator
declarations raise `ImplementationDeclarationError`.

The `traits(...)` decorator attaches passive metadata and returns its exact
target without wrapping, binding, backend imports, or extensible trait keys.
Repeated, malformed, shadowed, ambiguous, or unsupported descriptor evidence
raises `ImplementationDeclarationError` during catalog validation.

```python
from dryml.methods import Method, traits


class Double(Method):
    @traits(backend="numpy")
    def numpy(self, value):
        return value * 2
```

`implementations()` returns every authored `MethodImplementation` in stable
order. Each immutable carrier has `name`, the exact raw authored `target`, and
complete `traits`; inspection does not bind, select, invoke, warm, or import a
candidate. Catalog inspection is an ordinary local operation that may enter
Object/runtime machinery. It is unsupported inside an active orchestrator in
Stage 2. Stage 3's isolated probe does not exist in this release.

`compatible_implementations(input_spec=None, *, backend=None, batch_mode=None)`
accepts an optional first-input `SpecTree` and core trait constraints, returns
every compatible candidate in deterministic order, and never ranks or selects.
It returns an empty tuple for no compatible entries and raises
`ImplementationSelectionError(reason="conflict")` for malformed or
contradictory constraints.

`find_implementation(...)` uses the same inputs to return one uniquely
most-specific callable `MethodImplementation`. It raises
`ImplementationSelectionError` with `reason` `"no_candidate"`, `"ambiguous"`,
`"unknown_traits"`, or `"conflict"` before a target runs. For
`"unknown_traits"`, `unknown_traits` names missing `"backend"` and/or
`"batch_mode"` facts.

```python
import numpy as np

from dryml.core import TensorSpec

input_spec = TensorSpec("float32", shape=(2,), batch=4, backend="numpy")
implementation = Double().find_implementation(input_spec=input_spec)
assert implementation(np.ones((4, 2), dtype=np.float32)).shape == (4, 2)
```

The selected carrier validates the retained `input_spec` against exactly the
first positional logical argument. Known structure, mapping key/order, dtype,
shape, layout, backend, and batch facts must agree; unknown and `Dynamic` facts
accept concrete observations. Missing or conflicting first input raises
`ImplementationSelectionError` before the target. Later positional and keyword
arguments are forwarded unchanged. Calling a selected carrier never discovers
candidates or reads/mutates Method preparation state.

## Eager, Learning, And Cached Calls

Alternative-backed Methods begin with `call_mode == "eager"`. Each direct call
derives observable backend and batch facts, selects a unique safe candidate, and
invokes it locally. Direct calls do not imply dispatch, sessions, persistence,
managed execution, code transformation, or backend warmup.

Dense runtime arrays do not reveal element-versus-batched intent. While eager,
`default_batched` accepts exact `True`, exact `False`, or `None`. It fills only
an otherwise unknown batch fact: `True` requests batched, `False` element, and
`None` leaves selection unknown. Invalid values raise `TypeError`; mutation
while learning/cached raises `RuntimeError` without changing state. Observable
facts always win.

`learn()` returns `None`, clears an old cache, and enters `"learning"` mode
without selection, invocation, warmup, persistence, or optional-framework
imports. The next supported call normalizes complete positional and keyword
layout, selects under eager rules, and publishes an immutable
`cached_signature` plus target before invocation. Selection/normalization
failure stays learning without a partial cache; a selected target that fails
still leaves the cache available. Unsupported opaque learning values raise
`MethodError`; ordinary eager generic calls remain available.

Cached calls must exactly match the learned positional and keyword structure,
dtype, shape, layout, backend, and observable batch facts. Matching calls invoke
the retained implementation with no catalog discovery, ranking, output
inference, or default lookup. A mismatch raises
`PreparedCallMismatchError(expected, observed)` before user code and preserves
the cache. `MethodCallNode` and `MethodCallSignature` are immutable diagnostic
carriers that preserve tensor, tuple, list, mapping, and mapping-order facts.

`eager()` returns `None`, clears learning/cached state, and preserves an explicit
`default_batched`. `learn()` from cached state similarly clears the old cache and
preserves the default for exactly one new learning call.

## Local State, Composition, And Migration

`default_batched`, `call_mode`, and `cached_signature` are process-local state
of one live Method instance. They are excluded from CDef identity, Object state,
serialization, records, persistent compilation caches, and transport. Freshly
realized loads and forked children are eager with default `None`; loading that
reuses the identical live object preserves its local state. Concurrent cached
reads do not mutate state. Concurrent mode/default mutation of one Method is
unsupported and requires caller coordination.

`Map` selects a local callable once from a complete source spec. `Project`
selects each branch, `Pipe` threads pure intermediate specs, and `AutoEncoder`
selects encoder then decoder from threaded specs. This is local structural
selection, not global pipeline optimization, adapter insertion, or shared-node
planning.

```python
import numpy as np

from dryml.core import TensorSpec
from dryml.data import ArrayDataset, Map, Scale

dataset = ArrayDataset(
    np.array([[1.0, 2.0]], dtype=np.float32),
    spec=TensorSpec("float32", shape=(2,), backend="numpy"),
)
assert next(iter(Map(dataset, Scale(mean=0.0, std=0.5)))).tolist() == [2.0, 4.0]
```

Migrate `from dryml.code import Method, Traits, traits` to
`from dryml.methods import Method, Traits, traits`; import enums from core.
There are no `bind_first`, `resolve_impl`, `resolve_impl_for`, `get_impl`, or
`get_impl_func` compatibility APIs. Use direct calls,
`compatible_implementations()`, or `find_implementation()` instead.

## Errors

`MethodError` is the bounded contract base error.
`ImplementationDeclarationError` reports invalid authoring/catalog evidence.
`ImplementationSelectionError` reports pre-invocation selection or retained
input validation through `reason` and `unknown_traits`.
`PreparedCallMismatchError` exposes immutable `expected` and `observed`
signatures. These failures occur before a rejected candidate runs.

## Related Docs

- [Tensor Specs](tensor_specs.md)
- [Data API](data.md)
- [Models API](models.md)
- [Annotations](annotations.md)
