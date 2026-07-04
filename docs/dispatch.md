# DRYML Dispatch Metadata

`dryml.dispatch` currently defines metadata only. It does not launch workers, import targets, materialize objects, create environments, send cancellation signals, or replace `dryml.execute`.

## DispatchSpec

`DispatchSpec` is request intent: the operation the user asked for plus policies and overrides that should affect dispatch identity.

```python
import dryml.operations as ops
from dryml.dispatch import attach_dispatch_id, make_dispatch_spec

operation = ops.attach_operation_id(ops.make_function_call_spec("my_pkg.train:run"))
dispatch = attach_dispatch_id(
    make_dispatch_spec(
        operation_id=operation["id"],
        operation=operation,
        records={"record_policy": "descriptive", "provenance": True},
        execution={"backend": "local_subprocess"},
    )
)
```

Dispatch specs are store specs in family `dispatch`, schema `dryml.dispatch.v1`, and use `dispatch-v1-*` IDs. Operation identity remains separate: changing operation payload changes the `op-v1-*` ID, while changing dispatch policy changes the `dispatch-v1-*` ID.

Function-call and method-call operation specs may be embedded for self-contained tests and examples, but embedded operation IDs must match `operation_id`.

Method-call dispatch uses the same dispatch wrapper:

```python
method_operation = ops.attach_operation_id(
    ops.make_method_call_spec("cdef-v4-...", "train", kwargs={"epochs": 10})
)
method_dispatch = attach_dispatch_id(make_dispatch_spec(operation_id=method_operation["id"], operation=method_operation))
```

## ExecutionRecipe

`ExecutionRecipe` is resolved plan metadata. It records choices that a future backend can consume, such as backend identity, selected environment/world/runtime IDs, input plans, output plans, probe reports, and constraints.

```python
from dryml.dispatch import attach_recipe_id, make_execution_recipe

recipe = attach_recipe_id(
    make_execution_recipe(
        dispatch_id=dispatch["id"],
        operation_id=operation["id"],
        backend={"name": "dryml.local_subprocess", "kind": "local_subprocess"},
        input_plan={"materialize_cdefs": []},
        output_plan={"record_policy": "descriptive"},
    )
)
```

Recipes are store specs in family `execution_recipe`, schema `dryml.execution_recipe.v1`, and use `recipe-v1-*` IDs. A recipe is not an execution history record and does not prove anything actually ran.

## Boundaries

`ExecutionRecord` is the optional history record for what happened. Dispatch specs and recipes are not DRYML Objects, are not stored under `objects/`, and do not affect `ConcreteDefinition` identity.

Real local subprocess dispatch v2 is a later sprint. That sprint can consume `DispatchSpec` and `ExecutionRecipe` and emit `ExecutionRecord` provenance.
