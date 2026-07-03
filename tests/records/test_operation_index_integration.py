from dryml.core2.store.dir import DirStore
from dryml.formats.refs import format_cdef_id, format_ref_cdef
from dryml.operations import attach_operation_id, make_function_call_spec, make_method_call_spec
from dryml.records import RecordStoreIO


def cdef(char="a"):
    return format_cdef_id(char * 64)


def test_operation_specs_are_indexed_by_cdef_semantics(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))
    materializing = attach_operation_id(make_function_call_spec("pkg.mod:run", args=[cdef()]))
    reference = attach_operation_id(make_function_call_spec("pkg.mod:run", args=[format_ref_cdef(cdef())]))
    method = attach_operation_id(make_method_call_spec(cdef(), "train"))
    literal = attach_operation_id(make_function_call_spec("pkg.mod:run", args=[{"$literal": cdef()}]))

    refs = [io.write_spec(spec, family="operation") for spec in (materializing, reference, method, literal)]
    io.rebuild_ref_index()

    materializing_refs = io.find_operation_specs_for_cdef(cdef(), cdef_semantics="materialize", refresh=False)
    reference_refs = io.find_operation_specs_for_cdef(cdef(), cdef_semantics="reference", refresh=False)

    assert {ref.spec_id for ref in materializing_refs} == {refs[0].spec_id, refs[2].spec_id}
    assert {ref.spec_id for ref in reference_refs} == {refs[1].spec_id}
    assert refs[3].spec_id not in {ref.spec_id for ref in materializing_refs + reference_refs}
    assert io.read_spec(refs[0].spec_id, family="operation")["id"] == materializing["id"]
