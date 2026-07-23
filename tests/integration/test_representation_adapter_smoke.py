import core_objects as objects

from dryml.core.repo import Repo
from dryml.core.store.dir import DirStore
from dryml.formats.refs import format_cdef_id
from dryml.records import (
    AdapterDescriptor,
    AdapterRegistry,
    RepresentationRequirement,
    StorageRef,
    StoredStateRecord,
    make_representation_spec,
    resolve_state_record,
    run_adapter_plan,
)


def test_descriptive_save_fake_adapter_and_normal_load(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(stores=[store])
    obj = objects.TestClassC2(10)
    obj.set_val(20)
    cdef_id = format_cdef_id(obj.definition.stable_hash())

    repo.save_object(obj, record_policy="descriptive")
    source = next(StoredStateRecord.from_envelope(store.records.read_record(record_id)) for record_id in store.records.iter_record_ids())
    assert source.subject_cdef_id == cdef_id

    normalized = make_representation_spec("fake.normalized_state", storage_kinds=("product-dir",))
    store.records.write_spec(normalized, family="representation")
    registry = AdapterRegistry()

    def runner(context):
        context.session.write_text("normalized.txt", "normalized")
        return {}

    registry.register(
        AdapterDescriptor("fake.normalize", RepresentationRequirement(kind="dryml.object_state"), RepresentationRequirement(representation_id=normalized["id"], kind="fake.normalized_state")),
        runner=runner,
    )
    result = resolve_state_record(repo, cdef_id, RepresentationRequirement(representation_id=normalized["id"]), adapters=registry)
    assert result.status == "requires_adapter"
    assert result.adapter_plan is not None
    executed = run_adapter_plan(result.adapter_plan, repo=repo, store=store, registry=registry)
    assert executed.status == "ok"
    target = store.records.read_record(executed.target_records[-1].record_id)
    target_state = StoredStateRecord.from_envelope(target)
    assert target_state.storage[0] == StorageRef.self_product(role="target-state")
    assert store.records.resolve_storage_ref(target_state.storage[0], record_id=target["id"]).joinpath("normalized.txt").exists()
    assert store.records.read_record(executed.adapter_records[-1].record_id)["payload"]["target_record_id"] == target["id"]

    loaded = repo.load_object(obj.definition)
    assert loaded.data == 20
    store.records.ref_index_path.unlink(missing_ok=True)
    assert resolve_state_record(repo, cdef_id, RepresentationRequirement(representation_id=normalized["id"])).status == "ok"
