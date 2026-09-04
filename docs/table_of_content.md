# DRYML Documentation

Status: draft.

This documentation is the user-facing guide to DRYML. It complements API docstrings by explaining concepts, workflows, and common usage patterns.

## Recommended Reading Order

1. [Introduction](intro.md)
2. [Objects and Definitions](objects_and_defs.md)
3. [Immutable Definition Graph](immutable_definition_graph.md)
4. [Graph Querying](graph_querying.md)
5. [Annotations](annotations.md)
6. [Code Analysis](code_analysis.md)
7. [Formats](formats.md)
8. [Environments](environments.md)
9. [World And Runtime](world_runtime.md)
10. [Sessions](session.md)
11. [Repos and Stores](repos.md)
12. [Tensor Specs](tensor_specs.md)
13. [Methods](methods.md)
14. [Contexts](context.md)
15. [Data API](data.md)
16. [Models API](models.md)
17. [Artifacts API](artifacts.md)
18. [Query Index Backend Contracts](query_index_backend_contracts.md)
19. [Testing Workflow](testing.md)
20. [Release Notes](release_notes.md)

## Core Concepts

- DRYML programs are built from object graphs.
- A `Definition` is a deferred construction recipe.
- A `ConcreteDefinition` is a fully bound V2 structural identity; graph topology is available through graph equality/hash.
- `Ref` records a non-materializing exact or selector reference in a definition graph.
- An environment record describes observed Python/software facts without changing object identity.
- An `Annotation` is passive process-local key/value metadata; consumers own its meaning.
- `dryml.code` provides closed, local static analysis and bounded in-process tracing; its results are ephemeral and consumer-owned.
- A requested world describes roles and resources; an allocation binds one exact role-qualified process.
- `dryml.session` publishes persistent `python`, `managed`, or definition-only `orchestrator` state.
- An `Object` is the runtime instance associated with a concrete definition.
- A `Repo` manages live objects, persistent stores, aliases, queries, saves, and loads.
- `ObjectRef` adds durable ObjectId lineage and `StateRef` adds immutable snapshots.
- A `Store` owns immutable graph records and local checkpoint state.
- A `TensorSpec` describes tensor-like values independently from a specific ML backend.
- A `Method` is a logical callable with inspectable local implementations and optional process-local preparation.
- A `Context` describes runtime resource and backend compatibility constraints.
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

## Planned Follow-Up Documents

- `quickstart.md`: a minimal end-to-end first example.
- `queries.md`: full query and result-set semantics.
- `glossary.md`: short definitions of recurring terms.
- `backend_integrations.md`: TensorFlow, PyTorch, JAX, NumPy, sklearn, and XGBoost notes.
