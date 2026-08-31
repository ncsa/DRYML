# Reference Values

`ObjectId` is a namespace plus framework-generated UUID nonce for one `Serializable` lineage. `ObjectRef(definition, objects)` combines V2 graph topology with all owned ObjectIds. `StateRef(object, states)` adds immutable local-state hashes. These values are import-free canonical leaves and their equality includes complete topology and IDs.

`Ref(ObjectRef | StateRef)` is non-materializing. Bare or `Mat(ObjectRef | StateRef)` expands owned materializing topology. Materializing exact state references load through the exact StateRef path; a raw ObjectRef may be built only through its registered declaration and claim.

`StateSelectorRef(object, alias)` is soft only. Canonicalization resolves it once through a Repo to an exact StateRef before a CDef or finalized query exists. Object aliases name ObjectRefs; state aliases are scoped by complete ObjectRef. Alias movement never mutates a finalized CDef.
