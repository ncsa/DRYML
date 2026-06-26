# DRYML Documentation

Status: draft.

This documentation is the user-facing guide to DRYML. It complements API docstrings by explaining concepts, workflows, and common usage patterns.

## Recommended Reading Order

1. [Introduction](intro.md)
2. [Objects and Definitions](objects_and_defs.md)
3. [Repos and Stores](repos.md)
4. [Tensor Specs](tensor_specs.md)
5. [Contexts](context.md)
6. [Data API](data.md)
7. [Models API](models.md)
8. [Artifacts API](artifacts.md)

## Core Concepts

- DRYML programs are built from object graphs.
- A `Definition` is a deferred construction recipe.
- A `ConcreteDefinition` is a fully resolved, stable identity for an object.
- An `Object` is the runtime instance associated with a concrete definition.
- A `Repo` manages live objects, persistent stores, aliases, queries, saves, and loads.
- A `Store` owns persisted object state.
- A `TensorSpec` describes tensor-like values independently from a specific ML backend.
- A `Context` describes runtime resource and backend compatibility constraints.
- `Dataset`, `Model`, and `Artifact` are higher-level APIs built on the core object/repo system.

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
