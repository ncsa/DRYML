"""JSON command-line worker for lightweight code probes.

Protocol: ``python -m dryml.code.probe_worker --json`` reads one JSON
``CodeProbeRequest`` object from stdin and writes one JSON ``CodeProbeResult``
object to stdout. User-code output produced during import/analysis is captured
inside the result so it cannot corrupt protocol stdout.
"""

from __future__ import annotations

import argparse
import json
import sys

from dryml.code.probe import _InvalidTimeoutError, CodeProbeRequest, CodeProbeResult, diagnostic, run_probe_request


def main(argv: list[str] | None = None) -> int:
    """Run the probe worker CLI and return a process exit code."""

    parser = argparse.ArgumentParser(prog="python -m dryml.code.probe_worker")
    parser.add_argument("--json", action="store_true", required=True, help="use stdin/stdout JSON protocol")
    args = parser.parse_args(argv)
    if not args.json:
        return 2

    try:
        raw = sys.stdin.read()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            result = CodeProbeResult(
                ok=False,
                analysis=None,
                environment_record=None,
                diagnostics=(diagnostic("code_probe.invalid_json", "Probe worker received invalid JSON.", data={"error": str(exc)}),),
            )
        else:
            try:
                request = CodeProbeRequest.from_data(payload)
            except _InvalidTimeoutError:
                result = CodeProbeResult(
                    ok=False,
                    analysis=None,
                    environment_record=None,
                    diagnostics=(diagnostic(
                        "code_probe.invalid_timeout",
                        "Code probe timeout must be a finite positive number of seconds.",
                    ),),
                )
            except Exception as exc:
                result = CodeProbeResult(
                    ok=False,
                    analysis=None,
                    environment_record=None,
                    diagnostics=(diagnostic("code_probe.invalid_request", "Probe worker received an invalid request.", data={"error": repr(exc)}),),
                )
            else:
                result = run_probe_request(request, require_stable_import_path=True)
        sys.stdout.write(json.dumps(result.to_data(), sort_keys=True))
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        try:
            result = CodeProbeResult(
                ok=False,
                analysis=None,
                environment_record=None,
                diagnostics=(diagnostic("code_probe.unexpected_error", "Probe worker failed unexpectedly.", data={"error": repr(exc)}),),
            )
            sys.stdout.write(json.dumps(result.to_data(), sort_keys=True))
            sys.stdout.write("\n")
            return 0
        except Exception as fatal:
            sys.stderr.write(f"fatal code probe worker failure: {fatal!r}\n")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
