# Repos and Stores

`Repo` coordinates live realizations and one or more Stores. `manage_repo(None)` uses an explicitly active Repo or creates a temporary Repo for the operation; no process-global fallback exists.

`Repo.save_object()` and `Object.save()` publish a graph `StateRef`. Direct save keywords are `main`, `store`, `alias`, `deep_capture`, `federated`, and `report_stores`; no options object, revision, or traversal-depth control exists. Default non-federated saves copy verified immutable dependency state to the selected Store. Federated saves may retain dependencies in connected Stores. With `report_stores=True`, `StoreReport` exposes ordered root targets, confirmed local-state Stores, a deterministic sufficient recovery set, independently confirmed snapshots, and immutable per-boundary publication outcomes. `RepoSaveError.report` preserves the same partial evidence after planned capture/publication, index, name, or commit failure; interruption exceptions retain their original type and carry available report evidence.

`SaveRouting` is an immutable ordered `Selector` to connected-`Store` policy.
`Repo(..., save_routing=...)` and `set_save_routing()` accept it, `None`, or the
`"per-object"`/`"closure"` placement shorthands. A configured empty policy falls
back to the Repo default Store; `None` retains legacy closure behavior. A
configured per-object save captures each local state once, assigns payload only
to that Object's selected Stores, and publishes exact child `StateRef.at(path)`
projections. Closure mode makes every selected root Store self-contained; an
explicit `store=` always selects one complete closure. Installing routing
rejects distinct built-in Store handles for one physical destination, while
repeated use of the same handle deduplicates. A retained internal save context
snapshots the policy, Store order, and default together; later configuration
changes do not alter it and `Repo.close()` rejects while it is active.

Snapshots, stored-root membership, and initial claim completion are read back before a live `last_state_ref` receipt advances. A failed replica therefore leaves a prior root receipt intact, while an index, alias, main-reference, or later archive-commit failure leaves an already confirmed receipt inspectable. Root names are applied only after every selected root snapshot is confirmed and never propagate to independently published children. `Repo.save_object()` can leave a path-backed `ZipStore` buffered; `Repo.save()` and temporary convenience Repos report their required commit boundaries before returning.

`Repo.load(cdef)` and `load_object(cdef)` are structural operations and do not infer state. `Repo.load_or_build(x)` may create missing structure. `Repo.load_state_ref(state_ref, reuse_live="matching")` is the only exact snapshot load. `matching`, `greedy`, and `never` are exact live-reuse policies; no structural cache match can substitute for an ObjectId and binding match.

`Repo.reserve_state_graph(obj)` returns an active `StateGraphReservation` for the exact live graph's stateful ObjectIds and identities. It is process/thread local, Store-neutral, nonblocking, and all-or-nothing; use it as a context manager and pass it only to `Repo.save_object(..., reservation=...)` or `Repo.restore_state_ref_into(..., reservation=...)`. `restore_state_ref_into()` preflights complete authority and retained bindings, restores the supplied instances dependency-first without candidate search, and updates only the supplied root's `last_state_ref` on success. A restore-hook failure invalidates that live graph for later framework state IO; recover with a fresh `load_state_ref(..., reuse_live="never")`.

`Repo._for_state_io(stores)` is a private non-owning authority-only view for callers that select exact state Stores. It has an independent memory overlay but never opens or registers a persistent query index, commits, closes, or otherwise takes ownership of the caller's Stores.

Object aliases resolve with `get_alias()` to complete `ObjectRef` authority. State aliases resolve through `resolve_state_selector()` from an `ObjectRef`-scoped `StateSelectorRef` to a `StateRef`. There is no generic object-returning alias load because an ObjectRef does not select a snapshot. Declarations and claims reserve first construction; `build_object_ref()` requires a registered declaration and a valid claim. `fork_object_ref()` and `fork_state_ref()` are Repo-owned rekey operations.

`DirStore` is the supported directory checkpoint backend. It publishes immutable definition, local-state, declaration, and StateRef records, plus mutable aliases and claims. SQLite indexes and dirty markers are derived state. Rebuild is visible and may take time; it never replaces authoritative records. Supported concurrency relies on local filesystem atomic replacement, locks, and SQLite behavior. Distributed filesystems and cross-host coordination are unsupported.

Managed operations select a `DirStore` for immutable Object state and independently
select one for mutable lifecycle control. Their authority-only Repo view does not
open a query index or change session configuration. Callers own Store lifecycle
and must retain explicitly selected control-store locations for inspection:
managed has no locator or control journal. See [Managed Operations](managed_operations.md)
for reconciliation, interruption, callback, and invalid-target recovery rules.

Old Store layouts, format generations, and mutable current-state records reject before catalog registration, row decoding, restore, or index-ready activation. There is no migration or fallback reader.
