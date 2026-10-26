# Repo/Store API v3 Retirement

The compound-reference and Store-3 APIs have been removed. This is a breaking
Python API change: importing a retired name raises `ImportError`, importing
`dryml.core.reference_json` raises `ModuleNotFoundError`, and calling a removed
Repo/Store method raises `AttributeError`. There are no forwarding aliases,
deprecation wrappers, or automatic persisted-data conversions.

## Current APIs

| Removed surface | Current operation |
| --- | --- |
| `ObjectRef`, `StateRef`, `StateSelectorRef` | Use `Node` / `ConcreteNode` with `object_id` and `LocalState` facets; validate saved completeness with `require_saved(node)`. |
| `Object.object_ref` | Use `repo.describe(obj)`; call `.object_view()` for identity-preserving, state-weakened metadata. |
| `ObjectSelector`, `ObjectSelectorSet`, CDef `object_projection()` | Use Node patterns, `QueryPredicate`, `node.object_view()`, and `IdentitySet.object_projection()`. |
| Declaration lookup/build/claim methods | Concretize, optionally `allocate_identities()`, then `repo.build(node)`; use explicit queries for discovery and `reserve_state_graph` for live ownership. |
| Soft state aliases / `resolve_state_selector` | Query explicitly, choose a saved Node or LocalState, and bind that selection before realization. |
| `get_alias`, `set_alias`, old state aliases and `save_object_if_current` | Use `save(..., alias=...)`, `query().alias(...)`, and Store `NodeNameRecord` compare-and-set operations. |
| `fork_object_ref`, `fork_state_ref` | Explicitly replace Node identity/state bindings, then build/save as needed. Metadata copying is a separate deliberate operation. |
| `get_snapshot_directory` and snapshot-local readers | Borrow a full `StateKey` through `store.read_state(key)` or grouped `read_states`; use only the leased `data_directory`. |
| Reference JSON codecs and compound-reference transport tags | Use the authenticated shared Node graph codec for Node metadata. Old tags are rejected, not converted. |
| Old metadata classes/codecs | Use `NodeLineageMetadata`, `NodeSnapshotMetadata`, Repo metadata methods, and the bounded mapping codec. |

`Ref[...]`, `Mat[...]`, `ObjectId`, and `object_namespace` remain supported.
`SavedNode` is a typing refinement, not an `isinstance` target. Completeness is not
payload availability: loads freshly validate selected full keys and compatibility.

Some current names deliberately retain the word “reference”: `load_state_ref`,
`Object.last_state_ref`, query `object_refs()` / `state_refs()`, selection/report
`object_ref` / `state_ref` fields, and managed checkpoint/final receipts carry
Nodes. Ray's own `ObjectRef` is unrelated to the removed DRYML type.

## Removed Persistence And Claim Capabilities

The old `DefinitionRecord`, `StoredRootRecord`, `LocalStateManifest`,
`StateRefRecord`, `DeclarationRecord`, `ClaimRecord`, `MainRefRecord`,
`ObjectAliasRecord`, and `StateAliasRecord` classes and their Store inventory,
publication, snapshot, alias, and claim methods are absent. They are no longer
callable refusal stubs on built-in Stores. Current format-4 Node records,
declaration anchors, state variants, native reservations, journals, and retention
grants remain distinct supported contracts.

Repo no longer accepts `clock`, `lease_duration`, or `owner_token_factory`.
`RepoDefinition` uses the closed **version 2** envelope without claim settings;
version 1 is rejected. Its Node-selector carrier is still version 1. Shared Node
identity remains 3, Store format remains 4, and SQLite schema/coverage remain
11/3. Standalone manually populated CDef indexes have no authoritative Store
source for rebuilding; Store maintenance uses current format-4 projections.

Managed argument identity now represents an unannotated live Object with the
`live-node-object/v1` atom and its Node object-projection digest. It no longer
hashes a compound-reference digest. Supplied Node metadata continues to use its
exact Node digest; live payload mutation alone does not change object identity.

Old Store/reader fixture bytes remain frozen rejection or reader vectors. Their
obsolete Store-3/compound-reference generators have been retired; tests do not
regenerate those bytes with a current writer. Existing Stores are not rewritten.

Restart a long-lived interpreter after upgrading. Re-importing modules does not
migrate live Objects, cached class definitions, or already-created references.
See the [code-shift limitation](solutions/architecture-patterns/2026-09-18-code-shift-in-long-lived-sessions.md),
[Node values](ref_selector_values.md), [Repos and Stores](repos.md), and
[persistent formats](formats.md).
