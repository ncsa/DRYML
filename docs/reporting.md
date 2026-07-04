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
