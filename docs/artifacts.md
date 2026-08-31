# Artifacts

Artifacts are `Serializable` computed outputs. Their CDef records construction semantics, while their payload is immutable local state published by `save()`. A saved artifact returns a StateRef and exact restoration uses `Repo.load_state_ref()`.

Artifact code owns payload completeness in codec hooks. DRYML hashes the completed directory manifest but does not infer application mutation or semantic validity. Keep large computed output in state, not CDef parameters. Artifact references may be inspected and queried without importing backend payloads.
