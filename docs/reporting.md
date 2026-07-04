# DRYML Reporting

`dryml.reporting` is DRYML's structured progress-event layer. It is backed by Python-compatible reporter objects, but DRYML internals emit `DrymlEvent` records instead of printing directly or calling raw loggers at every call site.

The library default is quiet:

```python
import dryml

dryml.configure(reporting="steps")

with dryml.config(reporting="details"):
    ...
```

Environment defaults are read by `dryml.reset_config()` and initial import:

```bash
DRYML_REPORT=steps
DRYML_REPORT=details
DRYML_REPORT=debug
DRYML_REPORT=quiet
DRYML_REPORT_STREAM=stdout
DRYML_REPORT_STREAM=stderr
DRYML_REPORT_FORMAT=text
DRYML_REPORT_FORMAT=json
```

`DRYML_REPORT` is user-facing progress verbosity. `DRYML_LOG_LEVEL` remains available for normal Python logging configuration and developer diagnostics.

## Levels

`quiet` emits no progress events.

`steps` emits one-line lifecycle messages such as probing a target environment or resolving metadata.

`details` includes compact result summaries and selected IDs.

`debug` includes verbose structured payloads for cache keys, reports, source traces, and subprocess details.

## Reporters

Use the default stdout/stderr renderer for interactive progress, `LoggingReporter` to forward events to Python logging, `NullReporter` to suppress events, and `CaptureReporter` for tests or notebooks.

```python
capture = dryml.reporting.CaptureReporter()
with dryml.config(reporting={"level": "debug", "reporter": capture}):
    dryml.reporting.step("dryml.example", "Doing work")
```

Passing a reporter directly enables `debug` reporting when the current reporting level is `quiet`:

```python
dryml.configure(reporting=dryml.reporting.CaptureReporter())
```

Progress reporting is fail-soft by default: malformed event payloads or reporter failures are dropped and logged at debug level so reporting does not change DRYML semantics. Use `strict=True` in tests or diagnostics when reporting failures should raise.

## Subprocess Protocols

Provider probes and future workers must keep protocol stdout machine-readable. Reporting events in child processes are either suppressed, captured, sent through structured protocol fields, or emitted by the parent after decoding worker results. The provider probe worker redirects internal stdout/stderr while constructing its JSON response.

## Record And Adapter Events

Record resolution and local fake adapter execution emit structured events at orchestration boundaries:

| Event | Purpose |
|---|---|
| `dryml.records.state.find` | Scan/select stored-state records. |
| `dryml.records.representation.check` | Check representation compatibility. |
| `dryml.records.adapter.plan` | Plan zero-, one-, or multi-step adapter paths. |
| `dryml.records.adapter.run` | Run one local fake adapter step. |
| `dryml.records.product.write` | Commit product bytes and record sidecar. |
| `dryml.records.adapter.record` | Write adapter lineage records. |
| `dryml.dispatch.spec.build` | Build dispatch request-intent metadata. |
| `dryml.dispatch.recipe.build` | Build resolved execution recipe metadata. |
| `dryml.records.execution.write` | Write execution provenance records. |
| `dryml.records.execution.query` | Query execution provenance records. |
| `dryml.records.execution.export` | Include execution provenance in an explicit provenance export closure. |
| `dryml.dispatch.plan.start` | Build a dispatch plan. |
| `dryml.dispatch.requirements.gather` | Gather dispatch requirements/defaults. |
| `dryml.dispatch.requirements.merge` | Merge explicit overrides and defaults. |
| `dryml.dispatch.store.prepare` | Prepare same-host DirStore marshalling. |
| `dryml.dispatch.worker.launch` | Launch the local subprocess worker. |
| `dryml.dispatch.worker.handshake` | Validate worker protocol and store accessibility. |
| `dryml.dispatch.worker.execute` | Run the operation in the worker. |
| `dryml.dispatch.worker.cancel` | Cancel a local subprocess worker. |
| `dryml.dispatch.result.save` | Save dispatch outputs or output refs. |
| `dryml.dispatch.execution_record.write` | Write execution provenance for dispatch. |
| `dryml.dispatch.complete` | Complete dispatch and return a compact result. |

`quiet` remains silent. `steps` shows lifecycle messages. `details` includes selected record IDs, representation IDs, adapter names, and output record IDs. Reporting remains fail-soft unless strict mode is enabled.
