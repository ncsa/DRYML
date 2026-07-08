# Annotation Requirement Collection

## Status

Sprint 3 implementation note for authoritative requirement collection and resolution under `dryml.annotations`.

## Public API

`dryml.annotations` owns these collection and resolution helpers:

- `own_fragments(target, namespace=None, kind=None)` reads only fragments directly attached to one target.
- `fragments_for_method(cls, method_name, ...)` collects class MRO fragments followed by the concrete method implementation fragments.
- `fragments_for_definition_method(defn, method_name, ...)` resolves a live class from `.cls` or `.definition.cls` without building the object, then delegates to method collection.
- `resolve_fragments(...)`, `resolve_target_requirements(...)`, `resolve_method_requirements(...)`, and `resolve_definition_method_requirements(...)` return `RequirementResolution`.

## Semantics

Class fragments are inherited through MRO in base-to-subclass order. Method fragments describe the implementation body that owns them. If a subclass inherits a method unchanged, the inherited implementation fragments are included. If a subclass overrides a method, base method fragments are excluded by default while class-level fragments remain inherited.

Method lookup uses `inspect.getattr_static` so `classmethod` and `staticmethod` annotations are collected in both decorator orders. Collection inspects the raw descriptor and the underlying `.__func__`, deduplicating shared fragment objects while preserving order.

Provider fragments are appended after target fragments. This preserves existing annotation ordering while making externally supplied fragments visible in the final resolution and source traces.

## RequirementResolution

`RequirementResolution` contains merged environment, world, and runtime requirements/defaults where the current model supports them. It also preserves raw fragments, source traces, diagnostics, and merge report data through `to_data()` for future dispatch/probe provenance.

## Boundaries

Sprint 3 does not change dispatch behavior. Requirement declaration remains separate from dispatch candidate selection, environment/world compatibility checks, probes, runtime enforcement policy, and worker launch behavior.
