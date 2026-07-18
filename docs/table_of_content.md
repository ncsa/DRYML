# DRYML Documentation

Status: current.

This documentation is the user-facing guide to DRYML. It complements API docstrings by explaining concepts, workflows, and common usage patterns.

## Tutorials

This is the canonical tutorial sequence for both new DRYML learners and people
migrating from the legacy context, `Trainable`, or generated-ID APIs. Follow it
in order for the complete progression, or open any lesson independently after
meeting that lesson's prerequisites.

Use notebooks from the same DRYML version as the installed package. In a
repository checkout, use the notebooks under `examples/notebooks/`. In an
extracted source distribution, use the included notebooks from that matching
sdist. For an individual download, take the notebook from the exact matching
release tag or sdist and install that same DRYML version. The first three
lessons need a notebook-capable Python 3.10+ environment and the base
installation. The final three currently require Python 3.10-3.13 and the
`sklearn` extra, installed as `dryml[sklearn]`; that extra is not installed on
Python 3.14.

Each notebook is standalone: it uses installed public APIs, does not import
sibling support modules or inject a repository path, runs offline, and creates
any Store state only in temporary directories. The first lesson's initial
successful workflow saves mutable object state to a Store, closes and reopens
it, then verifies exact and alias loads restore that state.

1. [Objects, definitions, and repositories](../examples/notebooks/objects_definitions_and_repos.ipynb) (`base`): construction identity, mutable state, Store save/load, aliases, and query domains.
2. [Datasets and transforms](../examples/notebooks/datasets_and_transforms.ipynb) (`base`): re-iterable sources, tensor specs, method nodes, and structural transforms.
3. [Local defaults and plain mode](../examples/notebooks/local_defaults_and_plain_mode.ipynb) (`base`): environments, worlds, runtime allocation, trusted inline work, and worker dispatch.
4. [Models, experiments, and metrics](../examples/notebooks/models_experiments_and_metrics.ipynb) (`sklearn`): the maintained sklearn model, training, experiment, metric, and persistence path.
5. [Definition-driven experiments](../examples/notebooks/definition_driven_experiments.ipynb) (`sklearn`): immutable experiment variants and reproducible construction identity.
6. [Local hyperparameter search](../examples/notebooks/local_hyperparameter_search.ipynb) (`sklearn`): finite search spaces, bounded execution, deterministic selection, and temporary publication.

The sklearn lessons establish the current backend workflow. Continue with the
maintained TensorFlow and Torch APIs through the
[Models API backend progression](models.md#backend-progression); those backends
have their own extras and runtime requirements and are not prerequisites for
this sequence.

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
- [Static and dynamic analysis](../examples/code_analysis/static_and_dynamic_analysis.py)
