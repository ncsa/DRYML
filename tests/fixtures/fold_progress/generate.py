"""Print the synthetic Fold compatibility fixture patch; never run in reader tests.

Run with the declared big_env interpreter from the repository root. The recipe
uses explicit schema records, NumPy 2.x scalar carry, and pickle protocol 5, not
the production Fold writer. Output is an apply_patch addition packet for review;
existing fixture bytes must never be regenerated during qualification.
"""

import base64
import copy
import hashlib
import json
import pickle
import textwrap

import numpy as np


def fixture_records():
    """Return five explicit progress envelopes and one completed Value envelope."""

    carry_spec = {
        "kind": "tensor", "dtype": "float64", "shape": (), "batch": None,
        "backend": "numpy", "layout": "dense", "axis_names": None,
        "batch_axis_name": None, "ragged_rank": None,
        "row_splits_dtype": None, "sparse_format": None,
    }
    source_spec = dict(carry_spec, batch="dynamic")
    progress = {
        "state": "active", "count": 1,
        "carry": {"kind": "tensor", "backend": "numpy", "device": "host",
                  "dtype": "float64", "shape": (), "data": np.asarray(1.0)},
        "source_binding": {"kind": "concrete-definition", "digest": "synthetic-source"},
        "source_spec": source_spec, "observed_spec": source_spec,
        "carry_spec": carry_spec, "observation_devices": (None,),
        "program": {"initializer": "synthetic-initial", "accumulator": "synthetic-sum", "finalizer": None},
        "backend_devices": (("numpy", "host", "float64", ()),),
        "operation_id": "synthetic-operation", "attempt_id": "synthetic-attempt",
    }
    records = {}
    for version, state in ((1, "active"), (1, "exhausted"), (1, "completed"), (2, "active"), (2, "exhausted")):
        item = copy.deepcopy(progress)
        item["state"] = state
        if version == 2 and state == "exhausted":
            item["eof"] = "root-output-eof"
        records[f"v{version}-{state}.pkl.b64"] = {
            "format": "dryml.artifacts.fold-progress", "version": version,
            "processed_count": 2 if state == "completed" else 1,
            "progress": None if state == "completed" else item,
        }
    records["v1-completed-value.pkl.b64"] = {
        "format": "dryml.artifacts.value", "version": 1, "present": True, "result": 3.0,
    }
    return records


def main():
    """Emit base64 byte-preserving payloads and their single provenance manifest."""

    manifest = {
        "schema": "dryml-test-fold-progress", "version": 1,
        "provenance": "Synthetic explicit records; schema frozen from fold.py at ba368afc (v1); approved KTD02 grammar (v2). No user Store/data.",
        "encoding": "base64-wrapped pickle protocol 5; SHA256 covers decoded authoritative bytes and encoded files",
        "identity_binding": "Managed tests rebind only source_binding, program, operation_id, attempt_id to a synthetic receiver; carry/spec/state/EOF bytes originate here.",
        "expectations": {
            "v1-active": "resume saved carry 1 at position 1 over (1,2) -> 3; no initializer execution",
            "v1-exhausted": "restore succeeds; continuation requires explicit managed rerun; saves preserve v1",
            "v1-completed": "count 2 and completed Value 3 remain readable without source",
            "v2-active": "no EOF field; compatible positional recovery",
            "v2-exhausted": "root-output-eof field required; source-free finalization of saved carry 1",
            "downgrade": "frozen ba368afc reader rejects v2 at version gate before carry validation",
        },
        "files": {},
    }
    print("*** Begin Patch")
    for name, record in fixture_records().items():
        data = pickle.dumps(record, protocol=5)
        encoded = ("\n".join(textwrap.wrap(base64.b64encode(data).decode("ascii"), 76)) + "\n").encode("ascii")
        manifest["files"][name] = {
            "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
            "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        print(f"*** Add File: tests/fixtures/fold_progress/{name}")
        for line in encoded.decode("ascii").splitlines():
            print("+" + line)
    print("*** Add File: tests/fixtures/fold_progress/manifest.json")
    for line in json.dumps(manifest, indent=2).splitlines():
        print("+" + line)
    print("*** End Patch")


if __name__ == "__main__":
    main()
