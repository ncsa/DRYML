# Generic Execute

`dryml.execute` runs ordinary trusted Python callables through an explicitly
selected backend configuration. It does not provide DRYML `Repo`, `Store`,
Object, StateRef, managed-operation, or runtime/session transport.

Use `Executor` for a reusable backend lifetime, or pass a configuration to the
one-off `run` and `submit` helpers. The common facade is dependency-light and
does not eagerly import the subprocess or Ray specialization modules.

```python
from dryml.execute import Executor
from dryml.execute.subprocess import SubProcessConfig

with Executor(SubProcessConfig()) as executor:
    result = executor.run(sum, [1, 2, 3])
```

`submit` returns an `ExecutionFuture`, which provides result, failure,
cancellation, cleanup, output, and snapshot inspection. Pass ordinary workload
keyword arguments as `kwargs={...}` to `run` or `submit`; use
`executor.with_options(...)` when workload keywords must include control names.

The prior string backend selectors, implicit default orchestrator, and
`repo`, `update`, `transfer_store`, `result_store`, `env`, and `requirements`
transport options are removed. Select `SubProcessConfig` from
`dryml.execute.subprocess` or `RayBackendConfig` from `dryml.execute.ray`.
Ray connects only when the selected executor starts and requires an existing
same-host deployment; this migration note does not claim full Stage 7 or the
deferred core adapter is complete.
