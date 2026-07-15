# ADR 0006: Sprint 9C Maintained-Full Verification Deviation

## Status

Accepted by the user on 2026-07-15 for Sprint 9C closeout.

## Context

Sprint 9C required one green maintained-full `./tests.sh full` run. The one
closeout run passed its smoke/medium phase with 2,441 tests and one pre-existing
xfail, then failed two heavy cancellation tests while planning an explicit
Python environment. Both failures occurred before worker submission.

Focused investigation reproduced the failure only under severe shared-host
resource pressure. A final code probe and an environment probe each reached
their independent 30-second deadlines. At reproduction time the host had load
above 80, effectively exhausted 32 GiB of swap, more than 3,200 threads, and
I/O full-pressure above 70 percent. The planner correctly failed closed.

The cancellation tests mixed cancellation lifecycle coverage with unrelated
probe-worker startup. They now use an explicit current environment and the
standard-library `time:sleep` target. This retains real DRYML worker launch,
cancellation, timeout provenance, response retention, and pickle cleanup while
leaving code-probe and environment-probe integration to their dedicated tests.
The corrected cancellation file passed under the same host pressure both
without coverage and with the maintained-full coverage mode.

The shared host remained too saturated for a meaningful maintained-full rerun.
Running it again would knowingly exercise unrelated probe deadlines under the
same unsuitable conditions rather than provide useful release evidence.

## Decision

Accept Sprint 9C without a green maintained-full rerun on this host. Record the
failed run and focused diagnosis as an explicit verification deviation; do not
describe maintained-full verification as passing.

Do not skip, xfail, or remove the cancellation tests. They remain selected in
the heavy suite after their deterministic isolation correction. The skipped
item is only the second maintained-full closeout invocation.

This decision accepts the residual integration risk for Sprint 9C. It does not
weaken runtime probe deadlines, planner fail-closed behavior, cancellation
semantics, or the maintained-full requirement for future sprints and releases.

## Consequences

Sprint 9C can close with green focused cancellation coverage, green focused and
rolling evidence recorded before the host became saturated, and an explicit
exception for final maintained-full evidence. The final candidate has not been
demonstrated green across every maintained heavy test in one invocation.

Future audits must treat this as accepted verification debt, not proof that the
full suite passed. A failure reproduced on a suitable host remains actionable
and is not covered by this acceptance.

## Revisit Conditions

Run `./tests.sh full` and retire this deviation when a suitable local or CI host
is available with all of the following:

- no competing DRYML or pytest suite;
- material swap headroom rather than exhausted swap;
- sustained I/O full-pressure below 10 percent before launch; and
- enough uninterrupted time for the maintained heavy tier.

Record exact counts, duration, warnings, skips, xfails, host-pressure evidence,
and the candidate commit. If probe or cancellation tests fail on that host,
diagnose the structured planning/probe context and correct the product or test
contract rather than extending this deviation automatically.

## Alternatives Considered

Skipping or xfail-marking the cancellation tests would hide supported lifecycle
behavior and was rejected. Increasing production probe deadlines would mask host
contention without correcting cancellation and was rejected. Repeating the full
suite on the saturated host would consume substantial resources while producing
predictably unreliable evidence and was rejected.

## Source Anchors

- `tests/dispatch/test_cancellation.py`
- `tests/code/test_probe_worker.py`
- `tests/environments/test_environment_probe.py`
- `tests.sh`
- `docs/testing.md`
- `docs/adr/0001-code-analysis-boundaries.md`
