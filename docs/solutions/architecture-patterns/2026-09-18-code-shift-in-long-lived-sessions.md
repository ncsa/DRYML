---
title: Code Shift in Long-Lived Python and Notebook Sessions
date: 2026-09-18
problem_type: architecture_pattern
module: core, code, dispatch, environments
tags: [accepted-limitation, code-shift, hot-reload, notebooks, class-identity, selectors]
---

# Code Shift in Long-Lived Python and Notebook Sessions

## Decision

The project owner accepts source-versus-loaded-code drift as a limitation of the initial Stage 8 analysis/dispatch work.
Operation-long immutability of an externally managed Python/Conda/venv environment is also an accepted limitation.
These are accepted risks to revisit, not defects claimed to have been fixed or new hot-reload guarantees.

This decision does not waive ordinary detected target changes during submission, malformed requirement data, known requirement conflicts, mismatched environment evidence before payload authorization, or core Object/Store identity invariants.
It does not authorize implicit migration of live Objects or mutation of authoritative saved definitions/state.

## Source And Loaded-Code Drift

A Python function can retain its loaded code object after its backing file has been edited.
Reading that file for AST analysis can then observe a different body under the same name and source location.
A guard on the function's current code-object identity, globals, or attached declarations does not establish that newly read source is the source originally compiled into that object.

Consequently, static discovery is not a guarantee that source-derived dependencies describe the code a long-lived process will actually execute across a source edit or reload.
The initial Stage 8 work need not solve general source-to-loaded-code correspondence or coordinate live code updates.
Known incomplete discovery and detected changes must still be reported honestly; accepting this limitation must not be described as validating historical code/source identity.

## Notebook And Class-Redefinition Scenarios

Preserve these scenarios for a later focused code-evolution design:

- Redefine a class in a Jupyter cell after creating live instances. The cell name may now refer to a new class while existing instances remain associated with the earlier Python class.
- Change methods, bases, descriptors, constructor parameters, or state hooks progressively while Objects from earlier definitions remain in a Repo's live cache.
- Reload an imported module while retaining old function objects, bound methods, closure references, and `from module import ...` bindings.
- Materialize a persisted definition after the class currently bound at its import path has changed.
- Compare an old live Object or saved definition with a selector constructed from a newly defined class, including inheritance and exact-class matching.
- Capture new source-backed definitions from notebook `__main__` while old `SourceSpec` definitions remain authoritative for existing references.

Ordinary class redefinition does not update every live instance to the new class.
Manual Python `__class__` reassignment is not a DRYML migration contract and cannot be assumed to preserve CDefs, ObjectIds, bindings, state codecs, caches, or query semantics.

## Current Ownership And Behavior

- **CDef authority:** A CDef retains its symbolic class reference and bound parameters. It does not carry a universal loaded-module/session generation that reconciles later code edits.
- **Import references:** `ImportRef` identity uses module and qualified name. Resolution obtains the object currently bound at that path; identity validation when creating a reference is not an operation-long code-version guarantee.
- **Source specifications:** `SourceSpec` captures normalized source and import references. It is distinct from a path-only import reference; a changed captured class body can yield a different source-backed definition. Do not generalize import-path limitations into a claim that source-backed definitions contain no code information.
- **Materialization:** Core resolves a definition's class reference and projects its recorded parameters when constructing an Object. A changed resolved constructor can fail or behave differently; this is not automatic migration of earlier instances.
- **Live caches and state:** Repo caches retain live Objects with exact definition-node/realization associations. Exact StateRef restoration concerns state, topology, ObjectIds, and retained bindings, not arbitrary replacement of the live Python class.
- **Selector matching:** Equal symbolic class references match before runtime inheritance lookup. With unequal references, the default policy may resolve classes and use `issubclass`; exact/strict matching rejects unequal references. Therefore a stable import path does not imply historical-code equality, while changed SourceSpecs need not match merely because their visible class name is unchanged.
- **Reload helper:** `dryml.devtools.reload.reload_and_patch` reloads a module and rebinds selected caller globals. It does not migrate existing Objects, rewrite their classes/CDefs, reconcile Repo caches, or republish Store state.

These observations come from source inspection, not a runtime qualification of live-reload or migration behavior.
Future agents should verify the current implementation before relying on a particular edge case.

## Questions For Future Work

1. What code/version provenance should distinguish an import path from the particular class/function definition loaded at a point in time?
2. Should notebook/module redefinition be detected, rejected, diagnosed, or supported through an explicit session-generation boundary?
3. Can an opt-in live-Object migration preserve ObjectId and runtime bindings, or must some changes produce a new Object lineage and new definitions?
4. How should constructor/state-codec changes migrate persisted state, and how is partial or failed migration recovered without overwriting authoritative state?
5. Should selectors expose symbolic-path, current-runtime-inheritance, and historical-definition policies explicitly when code has changed?
6. Which live caches and derived query indexes need invalidation or rebuilding after a supported code update, and which authoritative identities must remain unchanged?
7. How should old bound methods, closures, and raw targets retained by decorators behave after a module reload or class redefinition?
8. Can source provenance be captured at definition/import time without executing analysis targets or introducing a new persistence authority?

These questions are future work, not additional Stage 8 completion requirements.
Any implementation needs its own product contract and must preserve the destination CDef V2 identity and Store authority rules.

## Related Environment Limitation

An exact environment selector identifies the requested existing interpreter/environment and admission checks its evidence before payload authorization.
A trusted external process can still install, remove, or modify packages in that environment afterward.
The initial work does not freeze or lock an entire Conda/venv prefix throughout workload execution.
This point-in-time verification limitation is accepted separately from code/source drift; observed pre-authorization mismatch still rejects rather than falling back to another environment.

## Evidence

- `docs/objects_and_defs.md`: structural definitions, current class resolution, and exact state restoration.
- `src/dryml/core/definition.py`: class references, parameter projection, and class matching.
- `src/dryml/core/symbol.py`: `ImportRef`, `SourceSpec`, and symbol resolution.
- `src/dryml/core/materialization.py`: class resolution during Object construction.
- `src/dryml/core/repo.py`: live-object caches and state-reference restoration.
- `src/dryml/core/query/query.py`: symbolic-reference equality and runtime inheritance matching.
- `src/dryml/devtools/reload.py`: bounded module/global rebinding, not Object migration.
- `src/dryml/code/source.py`: file-backed source retrieval and its distinction from retained Python code objects.
- `docs/execute.md`: existing-environment selection and pre-execution admission boundaries.
