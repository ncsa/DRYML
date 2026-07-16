# DRYML Documentation

Status: current.

This documentation is the user-facing guide to DRYML. It complements API docstrings by explaining concepts, workflows, and common usage patterns.

## Recommended Reading Order

1. [Introduction](intro.md)
2. [Objects and Definitions](objects_and_defs.md)
3. [Immutable Definition Graph](immutable_definition_graph.md)
4. [Formats](formats.md)
5. [Records](records.md)
6. [Representations and Adapters](representations_adapters.md)
7. [Requirements and Defaults](annotations.md)
8. [Dispatch and Explain](dispatch.md)
9. [Operations (advanced IR)](operations.md)
10. [Environments and Resolution](environments.md)
11. [Worlds and Runtime](world_runtime.md)
12. [Runtime, Worlds, and Dispatch overview](context.md)
13. [Code Analysis and Trace Architecture](architecture/code_analysis.md)
14. [Repos and Stores](repos.md)
15. [Tensor Specs](tensor_specs.md)
16. [Data API](data.md)
17. [Models API](models.md)
18. [Artifacts API](artifacts.md)
19. [Query Index Backend Contracts](query_index_backend_contracts.md)
20. [Migration from legacy context/execute APIs](migration/legacy_context_execute_removal.md)
21. [Release Notes](release_notes.md)
22. [Testing Workflow](testing.md)

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
