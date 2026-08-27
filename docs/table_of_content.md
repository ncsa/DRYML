# DRYML Documentation

Status: draft.

This documentation is the user-facing guide to DRYML. It complements API docstrings by explaining concepts, workflows, and common usage patterns.

## Recommended Reading Order

1. [Introduction](intro.md)
2. [Objects and Definitions](objects_and_defs.md)
3. [Immutable Definition Graph](immutable_definition_graph.md)
4. [Graph Querying](graph_querying.md)
5. [V1.1 Formats](formats.md)
6. [Environments](environments.md)
7. [World And Runtime](world_runtime.md)
8. [Sessions](session.md)
9. [Repos and Stores](repos.md)
10. [Tensor Specs](tensor_specs.md)
11. [Contexts](context.md)
12. [Data API](data.md)
13. [Models API](models.md)
14. [Artifacts API](artifacts.md)
15. [Query Index Backend Contracts](query_index_backend_contracts.md)
16. [Testing Workflow](testing.md)
17. [Release Notes](release_notes.md)

## Core Concepts

- DRYML programs are built from object graphs.
- A `Definition` is a deferred construction recipe.
- A `ConcreteDefinition` is a fully bound, versioned exact identity for an object; new identities use V2 semantic parameters.
- `Ref` records a non-materializing exact or selector reference in a definition graph.
- An environment record describes observed Python/software facts without changing object identity.
- Explicit annotations declare environment, world, and runtime requirements without activation.
- A requested world describes roles and resources; an allocation binds one exact role-qualified process.
- `dryml.session` publishes persistent `python`, `managed`, or definition-only `orchestrator` state.
- An `Object` is the runtime instance associated with a concrete definition.
- A `Repo` manages live objects, persistent stores, aliases, queries, saves, and loads.
- A `Store` owns persisted object state.
- A `TensorSpec` describes tensor-like values independently from a specific ML backend.
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
