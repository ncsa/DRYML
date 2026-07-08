# Code Analysis Architecture

## Status

Proposed Sprint 0 baseline note for review before the code-analysis API is introduced.

## Current State

`dryml.code` contains small helper modules for callable inspection, source extraction, AST access collection, `Method`, and method traits. `dryml.core2.symbol` owns import references and source-backed symbol references. These helpers are useful but do not yet form a single fact-oriented analysis API.

## Problem Statement

Dispatch, probes, annotations, and later analyzers need shared answers about Python targets. Those answers should not be duplicated inside dispatch or hidden inside annotation merging.

## Guiding Principle

`dryml.code` should collect reusable algorithms that discover facts about code. It should not decide where operations run. Dispatch should consume code facts when selecting and checking candidates. Probes should be able to run code-analysis algorithms outside the orchestrator when useful.

## What Belongs in dryml.code

- Callable identity and signature inspection.
- Importability analysis.
- Source location and source text extraction.
- Source-backed fallback data for non-importable functions/classes.
- Symbol/import dependency discovery.
- Direct annotation-fragment discovery as facts.
- Method contract facts.
- Shape hints where they are code-derived.
- AST access and method-call hints.
- Optional future dynamic trace call facts.
- Structured diagnostics.

## What Does Not Belong in dryml.code

`dryml.code` should not select environments, allocate worlds, enforce runtime policy, launch workers, or decide dispatch compatibility. Those responsibilities belong to dispatch, worlds, runtime, and provider/probe layers.

## Proposed Fact-Oriented API Direction

Sprint 1 should introduce a public `dryml.code.analyze(...)`-style API that returns a structured result object. The result should contain facts and diagnostics rather than a launch decision. Facts should be serializable enough for dispatch and probes to persist or pass between processes.

## Relationship to core2.symbol

`core2.symbol` already provides stable `ImportRef` and `SourceSpec` primitives. `core2` must not depend on `dryml.code`; `dryml.code` may depend on `core2`. This keeps the core semantic model independent of higher-level analysis algorithms.

## Relationship to Method and Method Handles

`Method`, method handles, `Traits`, and `CompilerInfo` likely belong closer to stable semantic model primitives, for example a future `core2.methods` area. Sprint 0 does not move them; it records the direction for review.

## Relationship to dispatch and probes

Dispatch should ask `dryml.code` for code facts and then apply requirement/candidate logic. Code probes should reuse the same algorithms in a lightweight `RuntimeMode.PROBE` process when orchestrator-local analysis is insufficient or risky.

## Non-Goals

- This note does not implement `CodeFact` or `CodeAnalyzer`.
- This note does not move `Method`.
- This note does not add code probes.
- This note does not add dynamic tracing.

## Source Anchors

- `src/dryml/code/callable_info.py`
- `src/dryml/code/source.py`
- `src/dryml/code/ast_tools.py`
- `src/dryml/code/method.py`
- `src/dryml/code/traits.py`
- `src/dryml/core2/symbol.py`
- `src/dryml/core2/tensor_spec.py`

## Open Questions

- Should facts be dataclasses, records specs, or both?
- Which diagnostics must be JSON-compatible in Sprint 1?
- How much source-backed fallback should be accepted before probes are required?

## Follow-Up Sprints

- Sprint 1: fact-oriented code analyzer API.
- Sprint 2: Method/method-handle placement review.
- Sprint 5: code probe worker.
- Sprint 9: optional dynamic tracing algorithm.
