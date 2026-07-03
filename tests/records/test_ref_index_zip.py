from io import BytesIO

from dryml.core2.store.zip import ZipStore
from dryml.formats.refs import format_cdef_id
from dryml.operations import make_function_call_spec
from dryml.records import RecordStoreIO


def cdef():
    return format_cdef_id("a" * 64)


def test_zip_store_persists_ref_index_after_commit():
    buf = BytesIO()
    store = ZipStore(buf)
    io = RecordStoreIO(store)
    spec_ref = io.write_spec(make_function_call_spec("pkg.mod:run", args=[cdef()]), family="operation")
    io.rebuild_ref_index()
    store.commit()
    store.close()

    reopened = ZipStore(buf)
    reopened_io = RecordStoreIO(reopened)
    assert reopened_io.read_ref_index().mention_count == 1
    assert [ref.spec_id for ref in reopened_io.find_operation_specs_for_cdef(cdef(), refresh=False)] == [spec_ref.spec_id]
    reopened.close()
