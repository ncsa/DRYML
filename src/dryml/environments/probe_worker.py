"""CLI worker for environment probes.

Run with ``python -m dryml.environments.probe_worker --json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from .introspection import inspect_current
from .schema import ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION
from .serialization import canonical_json_dumps


def main(argv: list[str] | None = None) -> int:
    """Run the environment probe worker CLI."""

    parser = argparse.ArgumentParser(description="Inspect a DRYML Python environment")
    parser.add_argument("--json", action="store_true", help="print a JSON probe payload")
    args = parser.parse_args(argv)
    if not args.json:
        parser.error("only --json is supported in this sprint")
    try:
        record = inspect_current()
        payload = {
            "kind": "dryml.environment_probe_result",
            "schema_version": ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION,
            "ok": True,
            "record": record.to_data(),
        }
        sys.stdout.write(canonical_json_dumps(payload))
        sys.stdout.write("\n")
        return 0
    except Exception as exc:  # pragma: no cover - exercised through subprocess failures
        payload = {
            "kind": "dryml.environment_probe_result",
            "schema_version": ENVIRONMENT_PROBE_RESULT_SCHEMA_VERSION,
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        sys.stdout.write(json.dumps(payload, sort_keys=True))
        sys.stdout.write("\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
