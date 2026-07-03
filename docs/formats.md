# DRYML Formats

`dryml.formats` contains generic, dependency-light helpers used by DRYML metadata layers. It is not an environment-specific API and does not implement record stores, operation specs, worlds, runtime dispatch, or object-state persistence.

## Canonical JSON

Canonical JSON helpers live in `dryml.formats.canonical`.

- Mapping keys must be strings.
- Mapping keys are serialized in lexicographic order.
- Lists and tuples serialize as JSON arrays.
- Sets and frozensets are accepted by normalization helpers and sorted by `repr`.
- `MappingProxyType` values can be converted back to mutable JSON-ready dictionaries with `json_ready`.
- `deep_freeze_json` returns deeply immutable structures suitable for dataclass fields.
- Non-finite floats, bytes, datetimes, and arbitrary Python objects are rejected.
- Canonical dumps use `sort_keys=True`, compact separators, `allow_nan=False`, and UTF-8 bytes.

Use `canonical_json_bytes(data)` when computing stable hashes. Use `deep_freeze_json(data)` when storing JSON-compatible metadata on immutable objects.

## Content IDs

Content-ID helpers live in `dryml.formats.ids`.

The stable ID shape is:

```text
<prefix>-v<schema_version>-<sha256>
```

`content_id(prefix, schema_version, data)` hashes this exact payload with SHA-256 over canonical JSON bytes:

```python
{
    "id_prefix": prefix,
    "schema_version": schema_version,
    "data": data,
}
```

Prefixes must match `^[a-z][a-z0-9_]*$`. Schema versions are positive integers. Existing environment prefixes remain valid and stable, including `envrec`, `envreq`, `envspec`, and `envlock`.

## Envelopes

Generic envelope helpers live in `dryml.formats.envelope`.

An envelope contains `schema`, `kind`, and `payload`, with optional `schema_version`, `id`, and `metadata` fields. `payload` defaults to `{}`. The helpers are intentionally generic; they do not define records, operation specs, or store layout.

`envelope_payload_for_id(envelope)` selects stable fields for content-ID hashing and excludes `id` and `metadata` by default. Put semantically identifying metadata inside `payload` when it should affect identity.

## Reserved References

Reserved reference helpers live in `dryml.formats.refs`.

Supported forms include:

```text
cdef-v4-<lowercase-hex-digest>
ref(cdef-v4-<lowercase-hex-digest>)
record-v1-<sha256>
spec-v1-<sha256>
env-v1-<sha256>
envrec-v1-<sha256>
envreq-v1-<sha256>
envspec-v1-<sha256>
envlock-v1-<sha256>
world-v1-<sha256>
worldreq-v1-<sha256>
runtime-v1-<sha256>
repr-v1-<sha256>
op-v1-<sha256>
annotation-v1-<sha256>
blob-v1-<sha256>
```

Use a literal escape when a value looks like a reserved reference but must remain ordinary JSON data:

```json
{"$literal": "cdef-v4-abcdef0123456789"}
```

Literal escapes must contain exactly one key, `"$literal"`, and may wrap any JSON-compatible value.
