# SQLite Query Lowering

## Current Store Projection Boundary

Format-4 Stores maintain SQLite schema **10**, coverage contract **1** synchronously
through their journaled public mutation boundary. `store_projection_rows` retains
source contributions and block/table/overlay correlation for typed node, edge and
facet postings. Recipe/saved membership is separate from known creation metadata,
full-StateKey manifests, data availability, declarations and names. The covering
source index makes incremental replacement proportional to changed sources rather
than total Store size. Coverage, mutation receipts and changed rows commit in the
same transaction; old schema 9 or incomplete coverage never becomes a ready empty
index.

The current read seam is `index.projection_rows(family, refresh=False)`, with
fresh Store UUID/epoch/pending and index UUID/schema/coverage checks. Disabled
refresh does no maintenance and refuses stale or missing coverage. Unknown or
not-yet-integrated metadata, retention and lifecycle families are refused rather
than treated as empty. See [Query Index Backend Contracts](query_index_backend_contracts.md)
for the closed family list, recovery and exact-acknowledgement protocol.

Repo query lowering, bounded pages, continuations and local-state discovery await
their owning consumer cutover. Format-4 Stores do not admit legacy Query V3 views
or advertise that those plans can consume the new projection already.

## Retained Legacy Lowering And Future Qualification

The retained legacy SQLite compiler lowers safe CDef and reference predicates to candidate relations, then verifies results against authoritative V2 values. Graph paths use the typed `Parameter` codec. A mismatched query schema or semantic-version bundle is rejected before row decode, not interpreted as an empty index. These legacy terminals are not enabled against format-4 authority.

The sidecar is derived. Dirty, missing, corrupt, or incompatible SQLite files are rebuilt visibly from current Store records; no rebuild mutates definitions, StateRefs, aliases, or local-state directories. Read transactions are short and connections are process-local. WAL behavior depends on the selected local filesystem and SQLite runtime.

## Million-Definition Qualification

The scale qualification dataset contains 1,000,000 stored root definitions with
one stable root class, a categorical `name` or `bucket` feature spread across at
least 1,000 values, one nested child on a fixed path for 10% of roots, and at
least one broad unindexed selector case.

For exact, selective, and broad Query V3 expressions, record the analyzed
explanation's source cuts, identities decoded, Python verifications, index-page
fetches, retained evidence size, and scan reason. Measure wall-clock time and
peak memory separately for `exists()`, `one()`, `count()`, and `take(1)`.

The qualification must demonstrate these contracts:

- Exact stored identity lookup uses the selective authority path and does not
  enumerate the broad definition inventory.
- Selective indexed candidates use a proved derived relation only when its
  projection coverage matches the captured authority cut.
- Broad `exists()` stops after a fully validated match when no later validation
  can invalidate that answer.
- `take(1)` retains only the canonical bounded prefix rather than building a
  complete final `IdentitySet`; eligible stored-CDef plans use graph-complete
  `(graph digest, encoded root)` keyset pages and a bounded cross-Store merge with
  one lookahead row per page. Every cursor is generation-bound, and candidates
  lacking direct Store proof fall back to complete authority or fail under an
  explicit no-scan/indexed-only policy.
- Broad unindexed expressions report the required authority scan, and
  `require_indexed()` rejects them rather than returning partial results.
- `count()` streams validated identities without constructing a complete final
  result set.

Run this benchmark separately from routine tests. It is a qualification recipe,
not a currently supported command-line entry point.
