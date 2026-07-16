# DRYML Documentation

Status: current.

This documentation is the user-facing guide to DRYML. It complements API docstrings by explaining concepts, workflows, and common usage patterns.

## Recommended Reading Order

1. [Create or open a Store](repos.md)
2. [Declare requirements and defaults](annotations.md)
3. [Dispatch a module-level function](dispatch.md#python-shaped-dispatch)
4. [Dispatch a stored CDef method](dispatch.md#python-shaped-dispatch)
5. [Explain a plan before launching](dispatch.md#requirement-aware-planning)
6. [Set notebook environment/world defaults](world_runtime.md#requested-defaults-allocation-and-plain-mode)
7. [Use `runtime.plain()` for inline work](world_runtime.md#requested-defaults-allocation-and-plain-mode)
8. [Analyze code without invoking it](architecture/code_analysis.md#fact-oriented-api)
9. [Opt into trusted current-process tracing](architecture/code_analysis.md#dynamic-trace-contract)
10. [Use Operations as advanced canonical IR](operations.md)

## Additional Documentation

- [Introduction](intro.md)
- [Objects and Definitions](objects_and_defs.md)
- [Immutable Definition Graph](immutable_definition_graph.md)
- [Formats](formats.md)
- [Records](records.md)
- [Representations and Adapters](representations_adapters.md)
- [Environments and Resolution](environments.md)
- [Runtime, Worlds, and Dispatch overview](context.md)
- [Tensor Specs](tensor_specs.md)
- [Data API](data.md)
- [Models API](models.md)
- [Artifacts API](artifacts.md)
- [Query Index Backend Contracts](query_index_backend_contracts.md)
- [Migration from legacy context/execute APIs](migration/legacy_context_execute_removal.md)
- [Release Notes](release_notes.md)
- [Testing Workflow](testing.md)

## Core Concepts

- DRYML programs are built from object graphs.
- A `Definition` is a deferred construction recipe.
- A `ConcreteDefinition` is a fully resolved, stable identity for an object.
- `Ref` records a non-materializing exact or selector reference in a definition graph.
- `dryml.formats` provides canonical JSON, content IDs, generic envelopes, and reserved-reference parsing for metadata layers.
- `dryml.records` provides optional store-owned JSON record/spec sidecars without changing object identity.
- Representation specs and fake/local adapter plans describe product formats and conversions without dispatch v2.
- `dryml.annotations` owns requirement/default collection and resolution; dispatch consumes its result.
- `dryml.dispatch` normalizes Python functions and CDef methods, resolves candidates, explains plans, and launches local workers.
- `dryml.code` is explicitly imported with `import dryml.code as code`; analyze is non-invoking and trace is a separate trusted execution opt-in.
- An environment record describes observed Python/software facts without changing object identity.
- An `Object` is the runtime instance associated with a concrete definition.
- A `Repo` manages live objects, persistent stores, aliases, queries, saves, and loads.
- A `Store` owns persisted object state.
- A `TensorSpec` describes tensor-like values independently from a specific ML backend.
- Current environment/world values are context-local planning defaults; runtime allocation is process-local actual state.
- `Dataset`, `Model`, and `Artifact` are higher-level APIs built on the core object/repo system.
- Store-owned query indexes accelerate stored and nested queries without changing object identity.
- Tests are grouped by feature category and automatically bucketed into smoke, medium, and heavy speed tiers.

## Documentation Status

These files are intentionally incremental. Each page should be updated when the corresponding API changes.

Use this rule when adding features:

1. Update docstrings for exact API behavior.
2. Update the relevant user-facing guide for workflow and concepts.
3. Add or update a small example when behavior is user visible.
4. Mark experimental or backend-specific behavior clearly.

## Focused Runnable Workflows

- [Requirements and explain](../examples/requirements/requirements_and_explain.py)
- [Python-shaped dispatch](../examples/dispatch/python_shaped_dispatch.py)
- [Notebook defaults and plain mode](../examples/notebooks/local_defaults_and_plain_mode.ipynb)
- [Static and dynamic analysis](../examples/code_analysis/static_and_dynamic_analysis.py)
