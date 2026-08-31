# Graph Querying

`Repo.query()` inspects CDef and reference authority without materializing objects or importing optional backends. Structural queries operate on CDef class and semantic parameters. `Repo.references()` returns ObjectRef and StateRef result sets for lineage, namespace, primary-path, alias, and exact-state questions.

Graph traversal records typed V2 `Parameter` and container paths. Materializing edges participate in owned object topology; `Ref` edges remain inspectable but are not constructed or saved as dependencies. Final query verification compares authoritative CDef/ObjectRef/StateRef values, not a CDef projection of an exact reference.

Store indexes are acceleration only. Rebuild scans authoritative definition and reference records, announces visible progress, and can safely replace a missing or stale derived index. A query may fail closed when current metadata is incompatible; it never treats an incompatible index as empty or current authority.
