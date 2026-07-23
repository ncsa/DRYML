from dryml.core.store.dir import DirStore
from dryml.formats.ids import content_id
from dryml.records import RecordStoreIO, attach_record_id, make_record, scan_record_refs
import dryml.runtime as runtime
import dryml.worlds as worlds


def test_world_runtime_specs_write_read_and_record_refs(tmp_path):
    io = RecordStoreIO(DirStore(tmp_path / "store"))
    world_req = worlds.attach_world_requirement_id(worlds.make_world_requirement_spec({"trainer": {"resources": {"accelerators": {"gpu": {"min": 1}}}}}))
    world = worlds.attach_world_id(worlds.make_world_spec({"trainer": {"replicas": 1, "process": {"resources": {"accelerators": {"gpu": 1}}}}}))
    runtime_spec = runtime.attach_runtime_id(runtime.make_runtime_spec(mode="worker", device_visibility={"policy": "assigned"}))
    allocation = worlds.attach_world_allocation_id(
        worlds.make_world_allocation_spec({"trainer": [{"replica": 0, "rank": 0, "local_rank": 0, "resources": {"accelerators": {"gpu": [0]}}}]})
    )

    for spec, family in ((world_req, "world_requirement"), (world, "world"), (runtime_spec, "runtime"), (allocation, "world_allocation")):
        ref = io.write_spec(spec, family=family)
        assert io.read_spec(ref.spec_id, family=family) == spec

    record = attach_record_id(
        make_record(
            kind="execution",
            payload={
                "execution_kind": "python",
                "operation_id": content_id("op", 1, {"op": "world-runtime"}),
                "backend": {"name": "dryml.fake"},
                "status": "ok",
                "world_requirement_id": world_req["id"],
                "world_id": world["id"],
                "world_allocation_id": allocation["id"],
                "runtime_id": runtime_spec["id"],
            },
        )
    )
    mentions = scan_record_refs(record)
    assert {mention.typed_key for mention in mentions} >= {"operation_id", "world_requirement_id", "world_id", "world_allocation_id", "runtime_id"}
    io.write_record(record)
    report = io.rebuild_ref_index()
    indexed_mentions = io.find_mentions(source_kind="record", target_kind="content_id", refresh=False)
    assert report.mention_count >= 4
    assert {mention.typed_key for mention in indexed_mentions} >= {"world_requirement_id", "world_id", "world_allocation_id", "runtime_id"}
    assert not hasattr(worlds.WorldSpec, "__dryml_object__")
