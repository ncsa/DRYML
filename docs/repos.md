# Repos and Stores

`Repo` coordinates live realizations and one or more Stores. `manage_repo(None)` uses an explicitly active Repo or creates a temporary Repo for the operation; no process-global fallback exists.

`Repo.save_object()` and `Object.save()` publish a graph `StateRef`. Direct save keywords are `main`, `store`, `alias`, `deep_capture`, `federated`, and `report_stores`; no options object, revision, or traversal-depth control exists. Default non-federated saves copy verified immutable dependency state to the selected Store. Federated saves may retain dependencies in connected Stores. The returned optional `StoreReport` is diagnostic only.

`Repo.load(cdef)` and `load_object(cdef)` are structural operations and do not infer state. `Repo.load_or_build(x)` may create missing structure. `Repo.load_state_ref(state_ref, reuse_live="matching")` is the only exact snapshot load. `matching`, `greedy`, and `never` are exact live-reuse policies; no structural cache match can substitute for an ObjectId and binding match.

Object aliases resolve with `get_alias()` to complete `ObjectRef` authority. State aliases resolve through `resolve_state_selector()` from an `ObjectRef`-scoped `StateSelectorRef` to a `StateRef`. There is no generic object-returning alias load because an ObjectRef does not select a snapshot. Declarations and claims reserve first construction; `build_object_ref()` requires a registered declaration and a valid claim. `fork_object_ref()` and `fork_state_ref()` are Repo-owned rekey operations.

`DirStore` is the supported directory checkpoint backend. It publishes immutable definition, local-state, declaration, and StateRef records, plus mutable aliases and claims. SQLite indexes and dirty markers are derived state. Rebuild is visible and may take time; it never replaces authoritative records. Supported concurrency relies on local filesystem atomic replacement, locks, and SQLite behavior. Distributed filesystems and cross-host coordination are unsupported.

Old Store layouts, format generations, and mutable current-state records reject before catalog registration, row decoding, restore, or index-ready activation. There is no migration or fallback reader.
