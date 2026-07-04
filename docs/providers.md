# DRYML Providers And Target-Environment Probes

`dryml.providers` is the dynamic counterpart to DRYML's static annotation decorators. Static decorators remain the lightweight way for Python code to declare known requirements and defaults. Providers are for facts that require importing framework code, user modules, or target-environment packages.

Provider reports are JSON-ready metadata records. They are not DRYML Objects, and they do not change `ConcreteDefinition` identity or object save/load behavior.

## Process Split

The orchestrator plans probes with `ProviderRef` objects and canonical request payloads. It does not instantiate provider classes or import target framework modules.

The probe worker runs in a child process selected by an `EnvironmentSpec`. Even `CurrentEnvironmentSpec` launches `sys.executable -m dryml.providers.probe_worker --json`; “current” means the same Python environment, not the current orchestrator process.

Before provider imports, the worker enters:

```text
RuntimeMode.PROBE + NoAllocation + device_visibility=none
```

Provider import failures, timeouts, malformed worker output, nonzero worker exits, unsupported methods, and provider exceptions are returned as structured `ProbeReport` diagnostics or provider reports.

## Provider Example

```python
import dryml.annotations as ann
import dryml.providers as providers


class Provider(providers.DrymlProvider):
    identity = providers.ProviderIdentity(
        "example",
        version="1",
        module=__name__,
        qualname="Provider",
        capabilities=("operation_inspection",),
    )

    def inspect_operation(self, request):
        fragment = ann.AnnotationFragment(
            "world",
            "requirement",
            {"roles": {"main": {"resources": {"cpus": {"min": 1}}}}},
            ann.SourceTrace("provider"),
        )
        return providers.OperationInspectionReport(
            provider_identity=self.identity,
            status="ok",
            request_key=request.key,
            operation_id=request.operation_id,
            fragments=(fragment,),
        )
```

Register import-path references in the orchestrator without importing provider modules:

```python
registry = providers.ProviderRegistry()
registry.register_ref(providers.ProviderRef("example", "my_pkg.provider", "Provider"))
```

## Probing An Operation

```python
from dryml.environments import CurrentEnvironmentSpec
from dryml.operations import make_function_call_spec

operation = make_function_call_spec("my_pkg.training:train")

report = providers.probe_operation(
    operation,
    environment=CurrentEnvironmentSpec(),
    providers=("example",),
    registry=registry,
    timeout=30.0,
)
```

For `PythonExecutableSpec` and `CondaEnvironmentSpec`, provider probes use the selected interpreter and `dryml.environments.build_probe_env(...)` to apply environment overrides and `PYTHONPATH` policy. Container probes currently return a structured unsupported report.

## Records

Probe reports are persisted through the existing generic record envelope:

```json
{
  "schema": "dryml.record.v1",
  "schema_version": 1,
  "kind": "probe_report",
  "payload": {
    "schema": "dryml.provider_probe_report.v1",
    "schema_version": 1,
    "request": { "request_kind": "operation_inspection" },
    "operation_id": "op-v1-...",
    "runtime_id": "runtime-v1-...",
    "reports": [
      { "report_kind": "operation_inspection", "status": "ok" }
    ],
    "status": "ok",
    "diagnostics": []
  }
}
```

Use `make_probe_report_record(...)`, `write_probe_report(...)`, and `probe_report_from_record(...)` for round trips through `RecordStoreIO`.

`ProbeReport` remains the authoritative provider/probe output. `ExecutionRecord` can be written separately as optional provenance for how a probe attempt ran, including failed or unsupported probe execution. Execution provenance may point at produced probe-report record IDs through `probe_report_ids`, but it does not replace the probe report payload.

## Annotation Integration

Provider output feeds the existing annotation resolver as ordinary `AnnotationFragment` objects:

```python
result = dryml.annotations.resolve(
    train,
    provider_fragments=report.annotation_fragments(),
)
```

Fresh reports rewrite fragment sources to `SourceTrace(kind="provider")`. Reports loaded from records should use `report.annotation_fragments(cached=True)`, which rewrites sources to `SourceTrace(kind="cached_probe")` and preserves provider name, version, operation ID, environment IDs, runtime ID, and probe report ID in source metadata.

## Representation And Adapter Payloads

Provider reports also carry a generic JSON-ready `report_payload`. Annotation fragments remain separate from representation/adapter payloads.

Representation inspection reports can include observed or supported representation specs:

```json
{
  "representations": [
    {
      "representation_spec": {
        "schema": "dryml.representation.v1",
        "kind": "fake.raw_state",
        "payload": {}
      },
      "applies_to": {"record_kinds": ["stored_state"]},
      "notes": []
    }
  ]
}
```

Adapter planning reports can include adapter descriptors:

```json
{
  "adapters": [
    {
      "name": "fake.normalize_state",
      "version": "1",
      "source": {"kind": "fake.raw_state"},
      "target": {"kind": "fake.normalized_state"},
      "cost": 1.0
    }
  ]
}
```

The orchestrator can deserialize these descriptors without importing provider modules or framework packages.

## Cache Hooks

`ProbeCacheKey` and `ProbeCache` provide an explicit exact-match cache. Store-backed lookup helpers scan `probe_report` records and validate JSON payloads. Cached reports are never authoritative unless the caller asks for them.

## Limitations

This sprint intentionally does not implement real Torch, TensorFlow, JAX, or DeepSpeed providers; adapter execution; compiler/JIT lowering; distributed backends; container execution; or probe-time materialization of arbitrary DRYML Objects.
