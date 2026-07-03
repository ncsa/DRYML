# DRYML Documentation

Status: draft.

This documentation is the user-facing guide to DRYML. It complements API docstrings by explaining concepts, workflows, and common usage patterns.

## Recommended Reading Order

1. [Introduction](intro.md)
2. [Objects and Definitions](objects_and_defs.md)
3. [Immutable Definition Graph](immutable_definition_graph.md)
4. [Formats](formats.md)
5. [Records](records.md)
6. [Operations](operations.md)
7. [Environments](environments.md)
8. [Repos and Stores](repos.md)
9. [Tensor Specs](tensor_specs.md)
10. [Contexts](context.md)
11. [Data API](data.md)
12. [Models API](models.md)
13. [Artifacts API](artifacts.md)
14. [Query Index Backend Contracts](query_index_backend_contracts.md)
15. [Testing Workflow](testing.md)
16. [Release Notes](release_notes.md)

## Core Concepts

- DRYML programs are built from object graphs.
- A `Definition` is a deferred construction recipe.
- A `ConcreteDefinition` is a fully resolved, stable identity for an object.
- `Ref` records a non-materializing exact or selector reference in a definition graph.
- `dryml.formats` provides canonical JSON, content IDs, generic envelopes, and reserved-reference parsing for metadata layers.
- `dryml.records` provides optional store-owned JSON record/spec sidecars without changing object identity.
- `dryml.operations` provides canonical operation-call specs and CDef/ref argument semantics without dispatching work.
- An environment record describes observed Python/software facts without changing object identity.
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
- `configuration.md`: session and repo configuration patterns.
- `backend_integrations.md`: TensorFlow, PyTorch, JAX, NumPy, sklearn, and XGBoost notes.
