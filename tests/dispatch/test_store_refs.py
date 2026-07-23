import dataclasses

import pytest

from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher, WorkerStoreRef, compute_dispatch_id
from dryml.dispatch.stores import open_worker_store, select_marshal_plan, validate_worker_store_access, worker_store_ref_from_dir_store
from dryml.formats.refs import format_cdef_id
from dryml.operations import attach_operation_id, make_function_call_spec


def test_worker_store_ref_from_dirstore_opens(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    ref = worker_store_ref_from_dir_store(store)

    assert ref.kind == "dir_store"
    assert open_worker_store(ref).base_dir == store.base_dir


def test_missing_store_path_fails_handshake(tmp_path):
    ref = WorkerStoreRef("dir_store", "shared", str(tmp_path / "missing"))
    with pytest.raises(Exception, match="missing"):
        validate_worker_store_access(ref)


def test_store_paths_stay_out_of_dispatch_identity(tmp_path):
    op = attach_operation_id(make_function_call_spec("operator:add", args=[format_cdef_id("a" * 64)]))
    left = Dispatcher(store=DirStore(tmp_path / "left", query_index="none")).plan(op, requirement_policy="ignore")
    right = Dispatcher(store=DirStore(tmp_path / "right", query_index="none")).plan(op, requirement_policy="ignore")

    assert compute_dispatch_id(left.dispatch_spec) == compute_dispatch_id(right.dispatch_spec)
    assert str(tmp_path / "left") in left.envelope.to_json()["store_refs"][0]["path"]
    assert str(tmp_path / "left") not in repr(left.dispatch_spec)


def test_unsupported_non_dirstore_transfer_strategy():
    assert select_marshal_plan(object()).strategy == "unsupported"
