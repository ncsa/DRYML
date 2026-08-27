# V1.1 Formats

The v1.1 metadata contract uses closed canonical JSON envelopes. Every emitted envelope has exactly `contract_version`, `schema`, `kind`, `payload`, `id`, and optional `metadata`; `contract_version` is exactly `"1.1"`. Parsers reject missing or unknown fields, source-v1 and future versions, duplicate textual JSON keys, invalid UTF-8, non-finite values, invalid bounds, and attached IDs that do not validate.

Canonical JSON has sorted string keys, compact UTF-8 encoding, finite floats, and detached deeply immutable projections. Defaults are depth 8, 1,024 nodes, 64 entries/container, 4,096-codepoint strings/keys, 4,096-bit integers, and 4,194,304-byte envelopes. Environment records allow 4,096 distributions, 65,536 nodes, and 16,777,216-byte envelopes. Public diagnostics are redacted and bounded to 512 codepoints with at most 64 entries per report.

| Family | Schema | Kind | ID prefix | Identifying payload | Non-identifying payload fields |
| --- | --- | --- | --- | --- | --- |
| Environment record | `dryml.environment_record.v1.1` | `environment_record` | `envrec` | Python version/implementation, platform fields, normalized distribution names/versions, DRYML version/protocol/schema/features, kind, tags | Interpreter paths/prefixes, distribution location/installer/editable facts, DRYML git revision, `details` |
| Environment requirement | `dryml.environment_requirement.v1.1` | `environment_requirement` | `envreq` | Normalized Python, PEP 508, exclusions, capabilities, tags, protocol, and schema constraints | `details` |
| Environment spec | `dryml.environment_spec.v1.1` | `environment_spec` | `envspec` | Every tagged spec payload field | None |
| Environment lock | `dryml.environment_lock.v1.1` | `environment_lock` | `envlock` | Every lock payload field | None |
| Annotation fragment | `dryml.annotation.v1.1` | `annotation_fragment` | `annotation` | Target, namespace, declaration kind, priority, merge policy, fragment, and source | None |
| World requirement | `dryml.world_requirement.v1.1` | `world_requirement` | `worldreq` | Entire normalized role requirement graph | None |
| Requested world | `dryml.world.v1.1` | `world_spec` | `world` | Entire normalized requested shape | None |
| Exact world allocation | `dryml.world_allocation.v1.1` | `local_allocation` | `worldalloc` | Backend plus role, replica, rank, resources, and environment assignment | Per-process diagnostic `metadata` |
| Runtime context | `dryml.runtime.v1.1` | `runtime_context` | `runtime` | Mode, visibility, frameworks, limits, environment, and allocation association | Payload `metadata` |
| Session configuration | `dryml.session_configuration.v1.1` | `session_configuration` | `sessioncfg` | Mode, resources, selected role-qualified allocation, environment requirement, and requirement axes | Derived `controls` |

IDs are `<prefix>-v1.1-<64 lowercase hex>` SHA-256 hashes over a domain-separated canonical preimage containing the prefix, version, schema, kind, and identifying payload. Envelope metadata never changes an ID. Record paths/prefixes, distribution installation facts, DRYML git revision, and `details` are non-identifying; requirement `details` are non-identifying.

The `current`, `python`, `conda`, and `container` spec kinds preserve their exact tagged fields. A Conda spec requires exactly one of `prefix` or `name`. World-family values allow 4,096 roles, 4,096 total processes, 256 entries per resource map, 65,536 nodes, and 16,777,216-byte envelopes. Other family-specific exceptions must not relax the 64-entry ordinary-container bound.

Typed public snapshots and DRYML-authored session/runtime diagnostics redact recognizable secret values and direct local paths before construction. Keys and bounded control status remain inspectable. This boundary does not claim exhaustive third-party exception/log redaction or `file:` URI masking.

No v1.1 family stores CDef references, Store authority, Object state, records/provenance sidecars, or query-index data. Environment values and the shared codec are implemented first; annotations, worlds, runtime, and session values adopt the table in their owning implementation units.
