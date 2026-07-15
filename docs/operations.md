# DRYML Operations

`dryml.operations` defines portable operation-call specs as canonical JSON metadata. Operation specs are regular `dryml.records` specs in the existing `operation` family. They are not `Object` instances, are not stored under `objects/`, and do not change `ConcreteDefinition` identity or object save/load behavior.

The `operation` spec family remains an open metadata family at the lower `dryml.records` layer. `dryml.operations.validate_operation_spec()` defines the portable Sprint 2 call-spec subset that future dispatch code should require explicitly.

## Function Calls

Use `make_function_call_spec()` to describe a future function call without importing or executing the target:

```python
from dryml.operations import attach_operation_id, make_function_call_spec

spec = attach_operation_id(
    make_function_call_spec(
        "my_pkg.eval:accuracy",
        args=[model_cdef_id, "ref(" + dataset_cdef_id + ")"],
        kwargs={"split": "test", "batch_size": 64},
    )
)
```

The canonical payload shape is:

```json
{
  "schema": "dryml.operation.v1",
  "schema_version": 1,
  "id": "op-v1-...",
  "kind": "function_call",
  "payload": {
    "function": "my_pkg.eval:accuracy",
    "args": [],
    "kwargs": {}
  },
  "metadata": {}
}
```

`function` must be a non-empty `module:qualname` import path. `args` defaults to `[]`, `kwargs` defaults to `{}`, and nested values must be canonical JSON-compatible. `kwargs` keys must be strings.

## Method Calls

Use `make_method_call_spec()` to describe a method call on a materialized subject CDef:

```python
from dryml.operations import make_method_call_spec

spec = make_method_call_spec(model_cdef_id, "train", kwargs={"epochs": 10})
```

The canonical payload shape is:

```json
{
  "schema": "dryml.operation.v1",
  "schema_version": 1,
  "kind": "method_call",
  "payload": {
    "subject": "cdef-v4-...",
    "method": "train",
    "args": [],
    "kwargs": {}
  }
}
```

`subject` must be a raw `cdef-v...` string, not `ref(cdef-v...)`. `method` must be a dotted Python attribute path such as `train` or `module_like.method_name`.

## Operation IDs

Operation IDs are spec IDs from the `operation` family:

```text
op-v1-<sha256>
```

The ID is computed by the existing `dryml.records` spec-family machinery from `schema`, `schema_version`, `kind`, and `payload`. It excludes `id`, `metadata`, store paths, indexes, and timestamps unless the timestamp is deliberately placed in `payload`. This does **not** make arbitrary metadata variants safe to publish: operation sidecars are immutable by their whole canonical bytes. Dispatch strips reserved planning/trace metadata before normalization, so traced and untraced equivalent operations keep byte-identical, trace-free operation sidecars.

## CDef, Ref, And Literal Semantics

Inside operation `args`, `kwargs`, and method-call `subject`:

```text
"cdef-v4-..."       means materialize that CDef and pass the resulting object.
"ref(cdef-v4-...)"  means pass the CDef identity/reference without materializing it.
{"$literal": "cdef-v4-..."} means pass the literal string.
```

`resolve_call_arguments()` implements only a resolver skeleton. By default it returns `MaterializeCDefArg` and `CDefRefArg` placeholders. Future dispatch code can provide explicit `materialize_cdef` and `make_cdef_ref` callbacks. Sprint 2 does not import function targets, select environments, launch workers, or execute operations.

## Storage

Store operation specs through the records API:

```python
located = store.records.write_spec(spec, family="operation")
assert located.spec_id.startswith("op-v1-")
```

Operation specs remain sidecar metadata under `records/specs/operation/`.

Dispatch metadata wraps operation IDs without changing operation identity. `DispatchSpec` records request policy/override intent with `dispatch-v1-*` IDs, while `ExecutionRecipe` records resolved plan metadata with `recipe-v1-*` IDs. Neither executes the operation. When explicit current-process dynamic tracing is requested, its per-run input/run identity and bounded provenance belong only to the dispatch/recipe/envelope/explanation carriers; they are never written to an `OperationSpec` or its operation sidecar.

For `pickle_small`, canonical operation arguments retain the one internal
`{"$literal": "dryml.pickled_callable.sha256:..."}` identity marker. Dispatch
validates its exact suffix and `identity_arg_count`, strips it only from the
private current-process trace invocation, and leaves the operation payload and
ID unchanged. A final post-trace same-Python rejection cleans the launch-only
pickle and returns the completed trace carrier; it does not publish or execute
the operation.
