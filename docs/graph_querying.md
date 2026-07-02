# DRYML Graph Querying Explained

**Suggested repository location:** `docs/graph_querying.md`  
**Alternative if the docs directory grows:** `docs/querying/graph_querying.md`  
**Current code reference:** `new-obj-system-4` around commit `6695a21`  
**Audience:** DRYML contributors and agents implementing, reviewing, or using Repo query/index code.

---

## 1. Purpose

DRYML graph querying exists to answer questions about saved and embedded object definitions without loading every object, importing heavy ML backends, or scanning every stored object directory. The key workflow is:

```python
with dryml.config(object_mode="definition"):
    experiment_def = Experiment(...)

already_run = repo.query(experiment_def).categorical(recursive=True).stored().count()

experiment_defs = (
    repo.query(experiment_def)
        .categorical(recursive=True)
        .stored()
        .defs()
)

models = (
    repo.query(autoencoder_selector)
        .nested()
        .definitions()
        .defs()
)

owners = (
    repo.query(autoencoder_selector)
        .nested()
        .owners()
        .defs()
)
```

The query system must distinguish:

```text
stored roots
    Definitions with their own persisted object directory / root def.pkl.

nested definitions
    Definitions embedded inside stored roots, possibly not independently saved.

owners
    Stored roots that contain a matching nested definition.

occurrences
    Specific root-to-nested-definition paths.
```

This distinction matters because many DRYML objects are structural. For example, an `Experiment` may be saved as a root, its encoder and decoder may be saved because they own model state, while an `AutoEncoder` composition may be embedded but not independently stored.

The query system is designed around this constraint:

> Searching should be definition-only. Loading or materializing objects is a separate explicit operation.

---

## 2. Suggested file name and location

Use:

```text
docs/graph_querying.md
```

Reasons:

1. The topic is broader than SQLite. It covers `Definition`, `ConcreteDefinitionGraph`, in-memory indexing, SQL-backed indexing, and result semantics.
2. The name is readable to users and contributors.
3. It can be linked from SQLite-specific docs such as `docs/sqlite_lowering.md` and backend contract docs.
4. It avoids implying that Graph Querying only means the SQLite backend.

If the docs tree becomes larger, move it to:

```text
docs/querying/graph_querying.md
```

and keep a short redirect or index link from `docs/graph_querying.md`.

---

## 3. The conceptual origin: GraphGrep, inverted indexes, and database query planning

DRYML graph querying borrows ideas from three older traditions:

1. **Subgraph search systems such as GraphGrep.**
2. **Information-retrieval inverted indexes.**
3. **Relational query planning and lowering.**

DRYML is not a direct reimplementation of any one system. It adapts their useful parts to a specific problem: searching canonical object-definition graphs.

### 3.1 GraphGrep-style idea: filter first, verify later

GraphGrep was designed to find occurrences of a query graph inside a database of graphs. Its high-level approach was to represent graphs using indexed fingerprints so a query can cheaply discard impossible candidates, then perform exact matching on the survivors. The GraphGrep paper describes the system as application-independent graph querying using hash-based fingerprinting to represent graphs in an abstract form and filter the graph database before exact matching.

DRYML uses the same broad strategy:

```text
selector graph
    ↓
cheap indexed features and edges
    ↓
small candidate set
    ↓
authoritative Definition matching
    ↓
verified results
```

Important difference:

```text
GraphGrep
    General graph/subgraph database search.

DRYML
    Canonical object definition DAG search with typed constructor paths,
    exact CDef identities, stored-root membership, and final Python selector semantics.
```

DRYML does not need a fully general graph-isomorphism engine. It uses exact `ConcreteDefinition` nodes, constructor paths, local structural features, and direct parent/child edges.

### 3.2 Inverted-index idea: dictionary + postings

The word **posting** comes from information retrieval. In an inverted index, there is a dictionary of terms, and each term points to the documents that contain it. The list of documents for one term is called a postings list.

DRYML reuses that structure:

```text
Information retrieval
    term       → documents containing that term
    posting    = one document occurrence of a term

DRYML query index
    feature    → CDef nodes containing that feature
    posting    = one CDef node has one local structural feature
```

For example:

```text
FeatureToken(CLASS_KEY, AutoEncoder)
    → CDef IDs whose root class key is AutoEncoder

FeatureToken(KWARG_PRESENT, "encoder")
    → CDef IDs whose local kwargs include encoder

FeatureToken(SCALAR_VALUE, path="$.kwargs.units", value=32)
    → CDef IDs with that local scalar value
```

A selector requiring several features can intersect postings lists:

```text
CLASS_KEY(AutoEncoder)
∩ KWARG_PRESENT("encoder")
∩ KWARG_PRESENT("decoder")
```

The rarest feature is usually the best anchor because it has the shortest postings list.

### 3.3 Database planning idea: lower query semantics to backend work

SQLite has its own query planner and optimizer, which chooses algorithms and index usage for SQL statements. DRYML uses SQLite's planner indirectly by lowering parts of a `DefinitionQuery` into SQL relations and indexed joins.

DRYML still does not push all semantics into SQL. SQL is used for safe candidate filtering:

```text
SQLite can safely do:
    stable-hash lookup
    local feature posting lookup
    direct-edge traversal
    stored-root membership filtering
    owner projection
    keyset paging

Python still does:
    final ConcreteDefinition equality confirmation
    full Definition selector matching
    callable selector leaves
    class policies that require Python semantics
```

This division is critical. SQL may return false positives. It must not return false negatives for supported indexed selectors. Python verification is the authority.

---

## 4. DRYML object/query terminology

### 4.1 Object

An `Object` is a runtime Python object. It can be materialized, trained, evaluated, saved, loaded, or restored.

Querying tries to avoid `Object` materialization. A query returns `ConcreteDefinition`s unless the user explicitly asks to load objects.

### 4.2 Definition

A `Definition` is a mutable, selector-capable object description. Users write or transform it. It can be partial.

A `Definition` can be used as:

```text
construction description
selector expression
projection source
categorical/base template
```

### 4.3 ConcreteDefinition / CDef

A `ConcreteDefinition` is the exact canonical identity of an object. It is immutable and hashable. Stores use it to locate object state.

In this document, **CDef** means `ConcreteDefinition`.

### 4.4 CDef stable hash

The stable hash is a deterministic content hash of a `ConcreteDefinition`. It is used for storage paths and exact-hash candidate lookup.

DRYML treats stable hash as an accelerator, not the full identity proof. Candidate hashes must still confirm full CDef equality because hash collisions are possible.

### 4.5 ConcreteDefinitionGraph

A `ConcreteDefinitionGraph` is the exact graph structure reachable from one or more root CDefs.

It has:

```text
nodes
    unique exact CDefs

edges
    direct parent CDef → direct child CDef
    labeled by a typed GraphPath

roots
    CDefs supplied as graph roots
```

Only direct CDef boundaries become edges. The parent does not flatten every descendant into itself.

Example:

```python
Experiment(model=AutoEncoder(encoder=Encoder(), decoder=Decoder()))
```

Graph:

```text
Experiment
    -- Kwarg("model") --> AutoEncoder

AutoEncoder
    -- Kwarg("encoder") --> Encoder
    -- Kwarg("decoder") --> Decoder
```

### 4.6 GraphPath

A `GraphPath` is a typed path through constructor/canonical values.

Common segments:

```text
Arg(i)
Kwarg(name)
Index(i)
Key(value)
SetMember(...)
```

Typed paths prevent ambiguity such as:

```text
Index(5)        sequence index
Key(5)          mapping key with integer value 5
```

### 4.7 Local feature

A local feature is a structural fact owned by one CDef node, stopping at nested CDef boundaries.

Examples:

```text
root class key
kwarg name exists
sequence length
mapping key exists
scalar value at local path
direct child exists at path
direct child class key
direct child stable hash
```

Local features avoid recursive duplication. The parent indexes that it has a direct child, not every feature inside the child.

### 4.8 Feature token

A feature token is the canonical encoded representation of one local feature.

It is the equivalent of a search term in an inverted index.

### 4.9 Posting

A posting means:

```text
feature token F appears in CDef node D with multiplicity M
```

A postings list is:

```text
all CDef nodes containing feature F
```

In SQLite, postings are rows. In memory, postings are maps.

### 4.10 Selector graph

A `SelectorGraph` is the query-side graph compiled from a user `Definition` or `ConcreteDefinition` selector.

It has:

```text
selector nodes
    local requirements and/or exact CDef constraints

selector edges
    required direct child relationships

root selector node
    the semantic query root
```

### 4.11 Anchor

The anchor is the selector node used to start indexed candidate search.

Good anchors are selective:

```text
exact CDef stable hash bucket
rare class key
rare local feature posting
rare nested child constraint
```

A query root does not have to be the anchor. A rare nested encoder can anchor an experiment query.

### 4.12 Candidate

A candidate is a possible matching CDef ID before authoritative matching.

Candidates are allowed to be false positives. They must not be false negatives.

### 4.13 Verification

Verification means running final DRYML selector semantics against candidate CDefs.

This includes:

```text
full CDef equality
Definition structural matching
container matching
class policy
callable selector leaves
```

Verification is the authority.

### 4.14 Stored root

A stored root is a CDef with its own persisted object directory / root definition.

It is independently loadable from a Store.

### 4.15 Nested definition

A nested definition is a CDef embedded inside a stored root's graph. It may not have its own object directory.

### 4.16 Owner

An owner is a stored root that contains a matching nested definition.

### 4.17 Occurrence

An occurrence is a specific path from a stored root to a nested definition.

The same nested CDef can have:

```text
one logical definition
many owners
many occurrences
```

### 4.18 Replica

A replica is the physical fact that a CDef is stored in a particular Store.

Two Stores can contain the same exact CDef. Repo should return one logical definition with multiple replicas.

### 4.19 Read view

A read view is a short-lived backend context for consistent index reads.

In memory, this is a context-bound view over in-memory structures.

In SQLite, this is a read transaction.

### 4.20 CandidateRelation

A `CandidateRelation` is an opaque backend-owned candidate set.

It carries metadata such as:

```text
source key
generation
relation id
relation kind
ordering
keyset support
estimated rows
exact-safe flag
debug label
```

The federation layer may inspect metadata and page CDefs, but it should not inspect backend-private SQL or temp table state.

### 4.21 Lowering

Lowering means converting the high-level DRYML selector graph into backend operations.

For SQLite, lowering produces SQL relations and relation operations.

### 4.22 Pushdown

Pushdown means doing safe filtering work in the backend instead of pulling large candidate sets into Python.

Examples:

```text
stable-hash lookup in SQLite
feature posting lookup in SQLite
edge traversal in SQLite
stored-root membership filtering in SQLite
keyset paging in SQLite
```

Final verification is deliberately not fully pushed down.

### 4.23 Semijoin

A semijoin filters one relation by the existence of matching rows in another relation.

In DRYML terms:

```text
parent candidates
    keep only those having a child edge to some child candidate
```

### 4.24 CTE

A CTE, or common table expression, is a named temporary result set inside one SQL statement.

SQLite uses `WITH` clauses for ordinary and recursive CTEs.

### 4.25 Keyset paging

Keyset paging fetches the next page after the last seen ordering key rather than using offset.

It is more stable for large result sets because it can use indexes and avoid skipping many rows.

---

## 5. How DRYML graph querying differs from general graph databases

DRYML's graph is specialized:

```text
General graph DB
    arbitrary nodes and edges
    arbitrary graph patterns
    often label/property matching
    often graph traversal language

DRYML graph index
    exact CDef nodes
    direct constructor-path edges
    local structural postings
    stored-root membership
    final Python Definition matching
```

DRYML does not need Cypher, SPARQL, or a graph database engine. It needs a fast persistent index over canonical object-definition graphs.

The result is simpler and safer:

```text
1. Find possible CDefs with indexed features and direct edges.
2. Verify with existing DRYML semantics.
3. Return definitions, owners, occurrences, or objects only when requested.
```

---

## 6. Index data model

The in-memory and SQLite indexes model the same logical facts.

### 6.1 Definitions

A definition record stores:

```text
backend-local definition id
stable hash
collision ordinal
class key
CDef blob or CDef object
local fingerprint
```

The backend-local ID is not DRYML identity. It is only a handle inside one backend view.

### 6.2 Feature dictionary

The feature dictionary maps feature tokens to feature IDs.

In memory:

```python
feature_token -> posting map
```

In SQLite:

```text
feature_tokens(feature_id, token_hash, token_blob, document_frequency)
```

### 6.3 Postings

A posting links:

```text
feature_id → def_id, multiplicity
```

In memory, this is typically a dictionary of dictionaries.

In SQLite, this is a `postings` table.

### 6.4 Edges

A direct edge links:

```text
parent_def_id, path, child_def_id
```

Paths are typed and versioned.

### 6.5 Stored roots

Stored roots are separate from graph nodes:

```text
definitions table
    all known graph nodes

stored_roots table
    subset independently stored in this Store
```

This separation is why DRYML can find ephemeral nested definitions without pretending they are directly loadable roots.

### 6.6 Replicas

For a Store-local SQLite index, every stored root row is one replica in that Store.

Repo federation merges replicas from multiple Stores.

---

## 7. Index build and save-time update

When an object is saved, DRYML already has or can compute its exact `ConcreteDefinitionGraph`.

Save-time indexing does this:

```text
1. Build exact CDef graph.
2. For each unique CDef node:
       compute local feature fingerprint
       register CDef record
       register feature postings
3. For each direct CDef edge:
       register typed edge
4. For each successfully stored root:
       activate stored-root membership
5. Increment index generation.
```

The object Store remains authoritative. SQLite index update happens after object files are committed.

If object files commit but index update fails, the Store is dirty/stale but recoverable. Reconciliation rebuilds or repairs the index from Store definitions.

---

## 8. Query execution overview

A normal query goes through these phases:

```text
User selector
    ↓
projection and query-builder transformations
    ↓
SelectorGraph compilation
    ↓
backend candidate planning/lowering
    ↓
backend candidate relation or candidate IDs
    ↓
paged CDef retrieval
    ↓
Python verification
    ↓
terminal sink
    ↓
ResultSet / count / exists / one / explanation
```

### 8.1 Example: stored exact query

```python
repo.query(exact_cdef).stored().exists()
```

Flow:

```text
exact stable hash lookup
    ↓
confirm full CDef equality
    ↓
filter stored-root membership
    ↓
exact-safe terminal path
    ↓
return True/False
```

### 8.2 Example: selective nested query

```python
repo.query(autoencoder_selector).nested().definitions().defs()
```

Flow:

```text
compile selector graph
    ↓
choose rare anchor
    ↓
find matching CDef nodes
    ↓
filter to nodes reachable from stored roots
    ↓
verify selector semantics
    ↓
return distinct nested CDefs
```

### 8.3 Example: owner query

```python
repo.query(autoencoder_selector).nested().owners().defs()
```

Flow:

```text
find matching nested nodes
    ↓
verify nested node CDefs
    ↓
project reverse through incoming edges to stored roots
    ↓
return owner root CDefs
```

### 8.4 Example: occurrence query

```python
repo.query(autoencoder_selector).nested().occurrences()
```

Flow:

```text
find matching nested nodes
    ↓
verify nested node CDefs
    ↓
capture relevant reverse ancestor graph
    ↓
lazily enumerate root-to-node paths
```

---

## 9. Selector graph compilation

A user selector is not executed directly. It is compiled to a graph representation.

Example selector:

```python
Experiment(
    model=AutoEncoder(
        encoder=exact_encoder,
        decoder=decoder_selector,
    )
)
```

Selector graph:

```text
node 0: Experiment local requirements
    -- Kwarg("model") --> node 1

node 1: AutoEncoder local requirements
    -- Kwarg("encoder") --> node 2
    -- Kwarg("decoder") --> node 3

node 2: exact encoder CDef
node 3: decoder local requirements
```

The compiler emits:

```text
local requirements
exact node constraints
direct selector edges
residual constraints if needed
```

Unsupported or unindexable pieces do not disappear. They are handled by final verification or scan policy.

---

## 10. Candidate planning and anchors

The planner estimates selector nodes and chooses an anchor.

Possible anchors:

```text
exact stable-hash node
rare local posting
class key
specific child edge
```

The anchor need not be the selector root.

Example:

```text
stored Experiments:     1,000,000
AutoEncoder nodes:        100,000
exact encoder nodes:            3
```

The planner should start from the exact encoder, not from every Experiment.

Then it propagates:

```text
encoder candidates
    → parent AutoEncoder candidates
    → parent Experiment candidates
```

This is how DRYML avoids root scans for selective nested queries.

---

## 11. In-memory Python indexing

The in-memory backend is the reference implementation for query semantics.

It stores Python data structures for:

```text
definition records
stable-hash buckets
feature postings
direct edges
incoming and outgoing adjacency
stored-root membership
cache membership
Store replica metadata
```

### 11.1 Read view

A memory read view is a context-bound object over the current catalog state.

It exposes operations such as:

```python
exact_ids(cdef)
local_candidates(requirements)
parents(child_relation, edge)
children(parent_relation, edge)
filter_domain(relation, domain)
filter_nested_ids(ids)
cdefs_by_id(ids)
replica_map(ids)
project_owners(ids)
capture_occurrences(ids)
```

The query planner can run against this without knowing whether the backend is memory or SQLite.

### 11.2 Exact lookup

In memory:

```text
stable_hash -> candidate definition IDs
```

Then full CDef equality confirms the match.

### 11.3 Local candidate lookup

For local requirements:

```text
feature token -> postings map
```

The memory backend chooses the rarest requirement and checks remaining requirements against the surviving IDs.

### 11.4 Edge traversal

Edges are indexed both directions:

```text
parent → edges → child
child → edges → parent
```

Parent/child relation operations use these adjacency structures.

### 11.5 Owner projection

Owners are found by reverse reachability:

```text
matching nested IDs
    ↑ incoming edges
stored roots
```

The owner terminal needs distinct stored roots, not occurrence paths.

### 11.6 Occurrence traversal

Occurrences require paths. The memory backend captures the relevant ancestor graph and lazily composes root-to-target paths.

This is more expensive than definitions or owners and is explicit.

### 11.7 Cache overlay

Cached definitions are process-local. They participate in `cached` and `known` domains.

They are not persisted to Store SQLite indexes.

---

## 12. SQL-backed indexing

The SQLite backend persists the same logical index beside each Store.

### 12.1 Store-owned sidecar

A directory Store has a sidecar database, conceptually:

```text
store/
    objects/
    aliases/
    .dryml/query-index-v1.sqlite3
```

The sidecar is rebuildable acceleration metadata. It is not object state.

### 12.2 Core SQLite tables

The schema mirrors the graph model:

```text
definitions
    unique CDef rows, stable hash, class key, CDef blob

feature_tokens
    token dictionary

postings
    inverted index: feature_id -> def_id

definition_edges
    direct CDef graph edges

stored_roots
    active independently stored roots in this Store

catalog_state
    schema versions, generation, dirty/build state
```

### 12.3 SQLite read view

A SQLite read view begins a short read transaction, exposes query operations, then closes before Python verification or ResultSet iteration.

This avoids retaining SQLite connections, cursors, or transactions in user-facing results.

### 12.4 CandidateRelation

SQLite lowers selectors to backend-owned relations:

```text
CandidateRelation(
    source_key,
    generation,
    relation_id,
    relation_kind,
    ordering,
    supports_keyset,
    estimated_rows,
    exact_safe,
    debug_label,
)
```

Federation sees metadata and asks the backend to page CDefs. It does not inspect SQL internals.

### 12.5 LoweredGraphPlan

A lowered plan describes:

```text
anchor node
anchor reason
anchor estimate
propagation steps
root projection
physical strategy
```

SQLite can represent the relation as:

```text
inline CTE
read-view-local temp relation
other backend-private representation
```

### 12.6 Inline versus temp relation

Small, single-use relations stay inline.

Large or reused relations may be materialized as read-view-local temporary tables.

Page terminals stay inline by default because query-backed ResultSets fetch each page in a fresh read view; a temp table would be rebuilt for every page.

### 12.7 Keyset paging

SQLite pages relation results using an ordering key such as:

```text
stable_hash, collision_ordinal, def_id
```

A `PagedResultCursor` records the last key for the next page.

### 12.8 Exact-safe paths

Some queries can be exact-safe in the backend after equality confirmation, such as exact stored CDef lookups. These can use optimized count/exists/defs paths without general selector verification.

### 12.9 Final verification remains in Python

Even with SQL lowering, DRYML verifies candidates in Python.

Reasons:

```text
hash collision confirmation
full Definition matching
callable selector leaves
complex class policies
future semantics changes
```

SQL filters candidates. Python verifies truth.

---

## 13. Repo federation across Stores

A Repo may have multiple Stores and a cache overlay.

Each Store has its own query index. Repo federation executes the query per source and merges results.

```text
Store A SQLite index
Store B SQLite index
Memory cache overlay
        ↓
FederatedQueryIndex
        ↓
verified logical CDefs + replicas
```

### 13.1 Deduplication

Two Stores can contain the same exact CDef.

Repo merges by:

```text
stable hash bucket
    full CDef equality
        combine replicas
```

### 13.2 Source-order paging

Current query-backed ResultSet order is stable source order:

```text
Store A pages
then Store B pages
then cache overlay pages
```

Within one SQLite source, ordering is keyset-based.

Global cross-Store keyset merge is future work.

### 13.3 Generation vectors

A federated ResultSet records a vector of source generations:

```text
Store A generation
Store B generation
cache generation
```

This makes result provenance explicit.

---

## 14. Why postings help performance

Suppose a Store has one million CDef nodes.

Without postings:

```text
for every CDef:
    load CDef
    inspect class/args/kwargs
    run selector match
```

With postings:

```text
feature: CLASS_KEY(AutoEncoder)
    postings list maybe 100,000

feature: KWARG_PRESENT("encoder")
    postings list maybe 80,000

feature: SCALAR_VALUE(units=32)
    postings list maybe 500
```

The rarest posting gives a much smaller anchor.

```text
candidate IDs = postings[SCALAR_VALUE(units=32)]
```

Then remaining feature checks and graph edges shrink the candidate set further.

This is the same core advantage as an inverted index: do not scan every document/node for every query.

---

## 15. Why direct edges help performance

Local features alone can identify a node that looks like an AutoEncoder. Edges identify where that node sits in a larger graph.

Example query:

```python
Experiment(model=AutoEncoder(encoder=exact_encoder))
```

The rare anchor may be `exact_encoder`. Direct edges let the backend climb:

```text
exact_encoder
    ← Kwarg("encoder") -- AutoEncoder
    ← Kwarg("model")   -- Experiment
```

This avoids scanning every Experiment and inspecting its nested model.

---

## 16. Why SQL lowering helps performance

The memory backend already knows how to choose rare anchors and traverse graph edges. The SQLite backend makes those operations persistent and backend-owned.

Instead of:

```python
ids = sqlite.local_candidates(...)
ids = sqlite.parents(ids, edge)
ids = sqlite.filter_domain(ids, stored)
```

with large Python sets, SQLite can represent intermediate relations inside SQL:

```sql
WITH anchor AS (...),
     parent AS (... JOIN definition_edges ...)
SELECT ...
```

or materialize a large relation as a read-view-local temp table.

The goals are:

```text
reduce CDef blobs decoded
reduce Python candidate sets
reduce object-directory scans
reduce Python verification count
use SQLite indexes and joins
```

---

## 17. Query terminals and performance

Different terminals need different amounts of work.

### 17.1 `defs()`

Collect verified CDefs.

### 17.2 `objects()`

Load/materialize verified CDefs. This is explicit and outside definition-only search.

### 17.3 `exists()`

Stop after the first verified match.

### 17.4 `one()` / `one_or_none()`

Stop after enough matches to know cardinality.

### 17.5 `count()`

Count verified logical CDefs while deduplicating across Store replicas and stable-hash collisions.

DRYML uses a witness-ref strategy: keep lightweight witness references for first-seen stable hashes and load/retain CDefs only when a stable hash repeats.

### 17.6 `explain()`

Explain planning without running full verification by default.

### 17.7 `explain(analyze=True)`

Run the query and report actual counters.

### 17.8 Query-backed ResultSets

A query-backed ResultSet fetches pages lazily rather than eagerly materializing all definitions.

It stores no SQLite cursor or transaction.

---

## 18. Owner and occurrence performance

### 18.1 Owners

Owners are distinct stored roots containing matched nested definitions.

Efficient owner lookup uses reverse graph traversal:

```text
nested target relation
    → reverse definition_edges
    → stored_roots
```

SQLite can do this with recursive CTEs or relation operations.

### 18.2 Occurrences

Occurrences are actual paths.

They are more expensive because a shared DAG can have many root-to-node paths.

Current design:

```text
capture relevant ancestor backing
lazily enumerate paths in Python
respect path limits where configured
```

Full SQL-native occurrence path enumeration is not necessary for current semantics.

---

## 19. Scan policy

Some selectors are not indexable.

Example:

```python
repo.query(lambda cdef: expensive_predicate(cdef)).stored()
```

or a graph-shaped selector with no exact node and no local requirements.

Scan policy controls this:

```python
query.scan_policy("allow")
query.scan_policy("warn")
query.require_indexed()
query.max_verify(10_000)
```

The query explanation should show when scanning is required and why.

---

## 20. Diagnostics

Good query explanations include:

```text
selector graph shape
anchor node
anchor reason
anchor estimate
anchor relation kind
propagation steps
physical relation strategy
inline/temp relation names
pages fetched
candidate rows read
CDef blobs decoded
Python verifications
scan fallback reason
terminal stop reason
SQLite EXPLAIN QUERY PLAN rows when requested
```

SQLite `EXPLAIN QUERY PLAN` is useful for diagnostics because it reports a high-level strategy, especially index usage. It is not a public semantic contract and should not be tested too brittlely.

---

## 21. Exactness and false positives

The index can return false positives.

Examples:

```text
same stable hash but different CDef collision
feature-level approximation
SQL subtree EXISTS filter that is necessary but not sufficient
callable selector not representable in SQL
```

The index must not return false negatives for supported indexed selectors.

Every returned definition must pass final verification unless the path is explicitly exact-safe.

---

## 22. In-memory versus SQLite summary

| Concept | In-memory backend | SQLite backend |
|---|---|---|
| CDef record | Python object/dict entry | `definitions` row |
| Feature dictionary | Python dict | `feature_tokens` table |
| Postings | dict[token][id] | `postings` table |
| Edges | adjacency maps | `definition_edges` table |
| Stored roots | Python set/map | `stored_roots` table |
| Query source | process-local catalog | Store-owned sidecar DB |
| Read consistency | lock/context view | read transaction |
| Candidate relation | Python sets or relation metadata | backend-owned SQL relation |
| Paging | Python iteration | keyset SQL page fetch |
| Persistence | none | durable sidecar |
| Multi-process visibility | process-local only | committed updates visible to next read |
| Cache membership | native | memory overlay only |

---

## 23. Performance summary

DRYML achieves performance through layered filtering:

```text
1. Do not scan object directories on normal queries.
2. Use exact stable-hash lookup for exact CDefs.
3. Use local inverted postings for structural features.
4. Start from the rarest anchor.
5. Traverse direct CDef edges instead of recursive Python search.
6. Apply stored/nested/owner domain filters in the backend.
7. Page CDef blobs instead of loading everything.
8. Verify candidates in Python only after backend filtering.
9. Short-circuit terminals like exists/one.
10. Avoid materializing Objects unless explicitly requested.
```

This makes the common case fast:

```text
many stored roots
few structurally matching definitions
small verified result set
```

It cannot make inherently huge result sets cheap to return. If the user asks for one million definitions, DRYML must eventually yield one million definitions.

---

## 24. End-to-end example

Query:

```python
selector = Experiment(
    model=AutoEncoder(
        encoder=exact_encoder,
    )
)

results = repo.query(selector).stored().defs()
```

Execution:

```text
1. Compile selector graph:
       Experiment → AutoEncoder → exact_encoder

2. Choose anchor:
       exact_encoder stable-hash bucket

3. SQLite relation:
       definitions.stable_hash = encoder_hash

4. Parent projection:
       definition_edges child=edge_encoder → AutoEncoder

5. Parent projection:
       definition_edges child=AutoEncoder at path model → Experiment

6. Domain filter:
       Experiment def_id in stored_roots

7. Page CDef blobs:
       fetch only candidate Experiments

8. Verify:
       Python Definition matching confirms true matches

9. ResultSet:
       verified CDefs plus replica metadata
```

---

## 25. Practical debugging guide

When a query is slow, ask:

```text
Did it have an indexable selector graph?
What anchor was chosen?
How many rows did the anchor estimate?
Did SQLite start from definitions or postings?
Did edge propagation happen in SQL?
Did the query fall back to scan?
How many CDef blobs were decoded?
How many Python verifications ran?
Did a terminal stop early?
Did a query-backed ResultSet page lazily?
Did owner or occurrence expansion dominate?
```

Useful API:

```python
explanation = query.explain(sql=True)
analysis = query.explain(analyze=True, sql=True)
```

Look for:

```text
scan_required
anchor_reason
anchor_relation_kind
physical_strategy
sqlite_plan
pages_fetched
cdef_blobs_decoded
verified_count
```

---

## 26. What not to over-optimize yet

Avoid prematurely optimizing:

```text
multi-anchor global optimizer
SQL-native occurrence path enumeration
global cross-Store keyset merge
perfect SQL count for arbitrary structural selectors
arbitrary callable pushdown
backend-import-aware subclass SQL matching
```

Optimize these only when benchmarks or real workflows show a problem.

---

## 27. References and further reading

### Graph querying and filtering

- GraphGrep paper: `GraphGrep: A Fast and Universal Method for Querying Graphs`  
  https://cs.nyu.edu/shasha/papers/graphgrep/icpr2002.pdf
- GraphGrep project page  
  https://cs.nyu.edu/shasha/papers/graphgrep/
- GraphFind, which discusses later improvements to GraphGrep-style graph search  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2367637/

### Inverted indexes and postings

- Stanford IR book, inverted index introduction  
  https://nlp.stanford.edu/IR-book/html/htmledition/a-first-take-at-building-an-inverted-index-1.html

### SQLite query execution

- SQLite query planner overview  
  https://sqlite.org/queryplanner.html
- SQLite optimizer overview  
  https://sqlite.org/optoverview.html
- SQLite `EXPLAIN QUERY PLAN`  
  https://sqlite.org/eqp.html
- SQLite common table expressions  
  https://sqlite.org/lang_with.html
- SQLite isolation and snapshot behavior  
  https://sqlite.org/isolation.html
- Python `sqlite3` module documentation  
  https://docs.python.org/3/library/sqlite3.html

### DRYML code references at `6695a21`

- Commit `6695a21`: Complete SQLite query optimizer relation planning  
  https://github.com/ncsa/DRYML/commit/6695a21
- Query package tree  
  https://github.com/ncsa/DRYML/tree/6695a21/src/dryml/core2/query
- SQLite query package tree  
  https://github.com/ncsa/DRYML/tree/6695a21/src/dryml/core2/query/sqlite
- `query/federation.py`  
  https://raw.githubusercontent.com/ncsa/DRYML/6695a21/src/dryml/core2/query/federation.py
- `query/sqlite/lowering.py`  
  https://raw.githubusercontent.com/ncsa/DRYML/6695a21/src/dryml/core2/query/sqlite/lowering.py
- `query/protocols.py`  
  https://raw.githubusercontent.com/ncsa/DRYML/6695a21/src/dryml/core2/query/protocols.py

---

## 28. Short version for contributors

DRYML graph querying is:

```text
GraphGrep-style filter/verify
+ inverted-index postings
+ exact CDef graph edges
+ SQLite relation lowering
+ Python authoritative verification
```

The query index answers:

```text
Which exact CDefs might match this selector?
Which stored roots contain this nested definition?
Which paths are occurrences?
Which Stores physically contain each result?
```

It deliberately does not answer:

```text
Load this Object.
Import TensorFlow to check class inheritance.
Run arbitrary selector callables in SQL.
Materialize every result eagerly.
```

The performance principle is simple:

> Push safe candidate filtering into the index. Keep final DRYML semantics in Python. Load Objects only when explicitly requested.
# Immutable Definition Graph Terminology

Graph-query code uses `Ref` / `EdgeKind.REF` for non-materializing edges.

Use `Selector(Definition(...))` for semantic query matching. `Definition.__eq__` is structural equality, not selector matching. Use `QuotedDef` or `SelectorSpec` when a selector/expression is stored as constructor data and must not emit `definition_edges` rows.
