# ADR 0007: Sprint 9C Final-Candidate Verification Deviation

## Status

Accepted by the user on 2026-07-15 for Sprint 9C closeout.

## Context

The first final read-only audit found that the recorded green governing focused
and rolling runs predated nested candidate
`2ef3992c273f97fc9a6a5901eff8e4c51a8c301f`. Later implementation repairs
changed rejected-evidence admission and diagnostic projection, and a test repair
isolated cancellation lifecycle coverage from unrelated probes. The final nested
commit then added the exact provenance-boundary matrix required by Sprint 9C.

The final candidate has green narrow evidence: all 88 dynamic-trace integration
tests pass, including construction and restoration at every exact provenance
bound and N+1, and all three cancellation lifecycle tests pass under their heavy
selection. Earlier post-repair evidence also includes a green 217-test governing
focused run and a green 678-test dispatch/code/annotations run, but those results
do not cover every later commit together. The latest green rolling run covered
`2dc0783a5c333b0ea944e8aed2e2e9ed892b8bdb`, before the later repairs.

The shared host is not suitable for meaningful probe-dependent reruns. At the
decision point its load average was `98.15, 99.18, 97.44`, all 32 GiB of swap was
used, and I/O full-pressure had `avg60=81.59`. Earlier governing tests had already
failed in unrelated probe paths under the same pressure. Repeating those suites
would primarily retest saturated-host deadlines rather than candidate behavior.

ADR 0006 covers only the missing green maintained-full rerun. It does not cover
this final-candidate governing focused and rolling evidence gap, so the user made
a separate explicit disposition.

## Decision

Accept Sprint 9C without rerunning the governing focused set or rolling selection
against nested candidate `2ef3992c273f97fc9a6a5901eff8e4c51a8c301f` on this
host. Preserve the prior green suite evidence, final-candidate narrow evidence,
and host snapshot as an explicit verification deviation. Do not describe the
governing focused set or rolling selection as passing on the final candidate.

This decision dispositions `AUD-9C-FINAL-001` for Sprint 9C closeout. It does not
waive implementation findings, the exact-boundary coverage requirement, or the
fresh final no-unresolved-P0/P1 audit. It does not skip, xfail, or deselect any
test and does not weaken focused or rolling requirements for future sprints and
releases.

## Consequences

Sprint 9C retains residual compatibility risk because all final-candidate changes
have not passed together through the governing focused and rolling selections.
The strongest final-candidate evidence is the complete dynamic-trace integration
file and cancellation lifecycle file, supplemented by earlier green governing
and rolling results. Future audits must treat this as accepted verification debt,
not as a green final-candidate suite result.

A focused or rolling failure reproduced on a suitable host remains actionable
and is not covered by this acceptance. Any product correction after such a
failure creates a new candidate and requires verification under the then-current
contract.

## Revisit Conditions

Run both commands and retire this deviation when a suitable local or CI host is
available with all of the following:

- no competing DRYML or pytest suite;
- material swap headroom rather than exhausted swap;
- sustained I/O full-pressure below 10 percent before launch; and
- enough uninterrupted time for the rolling selection.

Use the governing focused command:

```bash
pytest -q \
  tests/dispatch/test_dynamic_trace_integration.py \
  tests/dispatch/test_explain.py \
  tests/dispatch/test_planner_baseline.py \
  tests/dispatch/test_normalize_user_operation.py \
  tests/code/test_dynamic_trace.py \
  tests/annotations/test_definition_method_fragments.py \
  tests/annotations/test_requirement_resolution.py
```

Then use the rolling command:

```bash
pytest -q \
  tests/annotations \
  tests/code \
  tests/core \
  tests/dispatch \
  tests/environments \
  tests/notebook \
  tests/runtime
```

Record the candidate commit, host-pressure evidence, exact counts, duration,
warnings, skips, deselections, and xfails. Diagnose any failure rather than
extending this deviation automatically.

## Alternatives Considered

Running immediately on the saturated host was rejected because prior unrelated
probe deadlines and current I/O pressure make the result predictably unreliable.
Treating ADR 0006 as covering all verification gaps was rejected because its
scope is explicitly limited to maintained-full. Leaving the final audit finding
unresolved was rejected because the user chose an explicit, separately recorded
disposition instead.

## Source Anchors

- `tests/dispatch/test_dynamic_trace_integration.py`
- `tests/dispatch/test_cancellation.py`
- `docs/testing.md`
- `docs/adr/0006-sprint-9c-maintained-full-verification-deviation.md`
