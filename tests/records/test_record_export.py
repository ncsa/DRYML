import zipfile
from io import BytesIO

import core2_objects as objects
from dryml.core2.repo import Repo
from dryml.core2.store.dir import DirStore
from dryml.core2.store.zip import ZipExportStore, ZipStore
from dryml.formats.refs import format_cdef_id
from dryml.records import (
    AdapterRecord,
    RecordPolicyOptions,
    RecordStoreIO,
    StorageRef,
    StoredStateRecord,
    copy_record_closure,
    default_object_state_representation_spec,
    make_record,
    make_spec,
    plan_record_closure,
    record_export_include_paths,
)


def _cdef(char="a"):
    return format_cdef_id(char * 64)


def _seed_store(tmp_path):
    store = DirStore(tmp_path / "source")
    io = RecordStoreIO(store)
    repr_ref = io.write_spec(make_spec(family="representation", kind="repr", payload={"x": 1}))
    op_ref = io.write_spec(make_spec(family="operation", kind="function_call", payload={"args": [_cdef()], "kwargs": {}}))
    env_ref = io.write_spec(make_spec(family="environment_requirement", kind="requirement", payload={"x": 1}))
    unrelated_spec = io.write_spec(make_spec(family="runtime", kind="runtime", payload={"x": 1}))
    ancestor = io.write_record(make_record(kind="stored_state", payload={"subject_cdef_id": _cdef("b")}))
    seed = io.write_record(
        make_record(
            kind="stored_state",
            payload={
                "subject_cdef_id": _cdef(),
                "representation_id": repr_ref.spec_id,
                "operation_id": op_ref.spec_id,
                "environment_requirement_id": env_ref.spec_id,
                "derived_from": [ancestor.record_id],
            },
        )
    )
    execution = io.write_record(
        make_record(
            kind="execution",
            payload={
                "execution_kind": "python",
                "operation_id": op_ref.spec_id,
                "backend": {"name": "dryml.fake"},
                "status": "ok",
                "consumed_records": [{"record_id": seed.record_id, "required": True}],
            },
        )
    )
    adapter = io.write_record(make_record(kind="adapter", payload={"record_id": seed.record_id}))
    unrelated_record = io.write_record(make_record(kind="data", payload={"subject_cdef_id": _cdef("c")}))
    return store, seed, ancestor, execution, adapter, unrelated_record, (repr_ref, op_ref, env_ref, unrelated_spec)


def test_closure_includes_referenced_specs_and_excludes_provenance_products(tmp_path):
    store, seed, ancestor, execution, adapter, unrelated_record, specs = _seed_store(tmp_path)
    plan = plan_record_closure(store, seed_records=[seed.record_id], policy="closure")

    assert plan.records == (seed.record_id,)
    assert set(plan.specs) == {(ref.kind, ref.spec_id) for ref in specs[:3]}
    assert specs[3].spec_id not in {spec_id for _, spec_id in plan.specs}
    assert ancestor.record_id not in plan.records
    assert execution.record_id not in plan.records
    assert adapter.record_id not in plan.records
    assert unrelated_record.record_id not in plan.records
    assert plan.products == ()


def test_closure_works_with_missing_and_dirty_ref_index(tmp_path):
    store, seed, *_ = _seed_store(tmp_path)
    io = RecordStoreIO(store)
    io.rebuild_ref_index()
    io.write_record(make_record(kind="data", payload={"subject_cdef_id": _cdef("d")}))
    assert io.ref_index_is_dirty()

    plan = plan_record_closure(store, seed_records=[seed.record_id], policy="closure")
    assert plan.records == (seed.record_id,)
    io.ref_index_path.unlink()
    plan = plan_record_closure(store, seed_records=[seed.record_id], policy="closure")
    assert plan.records == (seed.record_id,)


def test_none_and_descriptive_export_policies_are_bounded(tmp_path):
    store, seed, *_ = _seed_store(tmp_path)
    io = RecordStoreIO(store)
    product_path = io.resolve_storage_ref(StorageRef.product_dir(seed.record_id, path="artifact"), create=True)
    (product_path / "data.txt").write_text("product", encoding="utf-8")
    none_plan = plan_record_closure(store, seed_records=[seed.record_id], policy="none")
    assert none_plan.records == ()
    assert none_plan.specs == ()
    assert none_plan.products == ()

    descriptive_plan = plan_record_closure(store, seed_records=[seed.record_id], policy="descriptive")
    assert descriptive_plan.records == (seed.record_id,)
    assert descriptive_plan.specs == ()
    assert descriptive_plan.products == ()

    descriptive_products = plan_record_closure(
        store,
        seed_records=[seed.record_id],
        policy="descriptive",
        options=RecordPolicyOptions(include_products=True),
    )
    assert descriptive_products.records == (seed.record_id,)
    assert descriptive_products.specs == ()
    assert descriptive_products.products == (seed.record_id,)

    product_dest = DirStore(tmp_path / "dest_descriptive_products")
    product_report = copy_record_closure(
        store,
        product_dest,
        seed_records=[seed.record_id],
        policy="descriptive",
        options=RecordPolicyOptions(include_products=True),
    )
    assert product_report.products_copied == (seed.record_id,)
    copied = RecordStoreIO(product_dest).resolve_storage_ref(StorageRef.product_dir(seed.record_id, path="artifact"))
    assert copied.joinpath("data.txt").read_text(encoding="utf-8") == "product"

    dest = DirStore(tmp_path / "dest_none")
    report = copy_record_closure(store, dest, seed_records=[seed.record_id], policy="none")
    assert report.policy == "none"
    assert report.records_written == ()
    assert report.specs_written == ()
    assert list(RecordStoreIO(dest).iter_record_ids()) == []


def test_provenance_and_all_are_explicit(tmp_path):
    store, seed, ancestor, execution, adapter, unrelated_record, specs = _seed_store(tmp_path)
    provenance = plan_record_closure(store, seed_records=[seed.record_id], policy="provenance")
    assert set(provenance.records) == {seed.record_id, ancestor.record_id, execution.record_id, adapter.record_id}

    closure = plan_record_closure(store, seed_records=[seed.record_id], policy="closure")
    assert set(closure.records) == {seed.record_id}

    all_plan = plan_record_closure(store, policy="all")
    assert {seed.record_id, ancestor.record_id, execution.record_id, adapter.record_id, unrelated_record.record_id} <= set(all_plan.records)
    assert {ref.spec_id for ref in specs} <= {spec_id for _, spec_id in all_plan.specs}


def test_all_includes_existing_products_by_default_and_indexes_never_exported(tmp_path):
    store, seed, *_ = _seed_store(tmp_path)
    io = RecordStoreIO(store)
    product_path = io.resolve_storage_ref(StorageRef.product_dir(seed.record_id, path="artifact"), create=True)
    (product_path / "data.txt").write_text("product", encoding="utf-8")
    no_product = io.write_record(make_record(kind="stored_state", payload={"subject_cdef_id": _cdef("e")}))
    io.rebuild_ref_index()

    plan = plan_record_closure(store, policy="all")
    paths = record_export_include_paths(plan)
    assert f"products/{seed.record_id}/" in paths
    assert f"products/{no_product.record_id}/" not in paths
    assert all(not path.startswith("records/indexes/") for path in paths)


def test_copy_record_closure_dir_to_dir_and_rebuilds_destination_index(tmp_path):
    source_store = DirStore(tmp_path / "source")
    dest_store = DirStore(tmp_path / "dest")
    source_repo = Repo(stores=[source_store])
    dest_repo = Repo(stores=[dest_store])
    obj = objects.HelloStr(msg="test")
    source_repo.save(obj, record_policy="descriptive")
    dest_repo.save(obj, record_policy="none")

    source_io = RecordStoreIO(source_store)
    seed_id = next(source_io.iter_record_ids())
    report = copy_record_closure(
        source_store,
        dest_store,
        seed_records=[seed_id],
        options=RecordPolicyOptions(rebuild_index=True),
    )
    dest_io = RecordStoreIO(dest_store)
    copied = dest_io.read_record(seed_id)

    assert report.records_written[0].store_ref == dest_io._store_ref()
    assert copied["id"] == seed_id
    assert dest_io.read_ref_index().store_ref == dest_io._store_ref()
    assert dest_io.resolve_storage_ref(copied["payload"]["storage"][0]).is_dir()
    assert not (dest_io.records_dir / "indexes" / "source-index.json").exists()


def test_copy_record_closure_product_copy_ignores_records_without_products(tmp_path):
    store, seed, *_ = _seed_store(tmp_path)
    io = RecordStoreIO(store)
    product_path = io.resolve_storage_ref(StorageRef.product_dir(seed.record_id, path="artifact"), create=True)
    (product_path / "data.txt").write_text("product", encoding="utf-8")
    dest = DirStore(tmp_path / "dest")

    report = copy_record_closure(store, dest, seed_records=[seed.record_id], options=RecordPolicyOptions(include_products=True))
    dest_io = RecordStoreIO(dest)
    copied_path = dest_io.resolve_storage_ref(StorageRef.product_dir(seed.record_id, path="artifact"))
    assert copied_path.joinpath("data.txt").read_text(encoding="utf-8") == "product"
    assert report.products_copied == (seed.record_id,)

    missing_dest = DirStore(tmp_path / "missing_dest")
    missing_record = io.write_record(make_record(kind="stored_state", payload={"subject_cdef_id": _cdef("e")}))
    missing_report = copy_record_closure(
        store,
        missing_dest,
        seed_records=[missing_record.record_id],
        options=RecordPolicyOptions(include_products=True),
    )
    assert missing_report.products_copied == ()


def test_record_export_include_paths_and_zip_export(tmp_path):
    store, seed, *_ = _seed_store(tmp_path)
    plan = plan_record_closure(store, seed_records=[seed.record_id], policy="closure")
    paths = record_export_include_paths(plan)
    assert f"records/items/{seed.record_id}.json" in paths
    assert any(path.startswith("records/specs/representation/") for path in paths)
    assert all(not path.startswith("records/indexes/") for path in paths)

    buf = BytesIO()
    ZipExportStore(buf, store.base_dir, include_paths=paths).commit()
    with zipfile.ZipFile(buf, "r") as zf:
        names = set(zf.namelist())
    assert f"records/items/{seed.record_id}.json" in names
    assert all(not name.startswith("records/indexes/") for name in names)

    reopened = ZipStore(buf)
    reopened_io = RecordStoreIO(reopened)
    assert reopened_io.read_record(seed.record_id)["id"] == seed.record_id
    reopened_io.rebuild_ref_index()
    assert reopened_io.read_ref_index().store_ref == reopened_io._store_ref()
    reopened.close()


def test_legacy_zip_export_without_records_remains_compatible(tmp_path):
    store, seed, *_ = _seed_store(tmp_path)
    buf = BytesIO()
    ZipExportStore(buf, store.base_dir, include_paths={"objects"}).commit()
    with zipfile.ZipFile(buf, "r") as zf:
        names = set(zf.namelist())
    assert all(not name.startswith("records/") for name in names)
    assert f"records/items/{seed.record_id}.json" not in names


def test_typed_record_closure_and_self_product_export(tmp_path):
    source = DirStore(tmp_path / "source")
    dest = DirStore(tmp_path / "dest")
    io = RecordStoreIO(source)
    spec = default_object_state_representation_spec()
    io.write_spec(spec, family="representation")
    source_state = io.write_record(StoredStateRecord(_cdef(), spec["id"], (StorageRef.self_product(role="source"),)).to_envelope())
    io.product_root(source_state.record_id, create=True).joinpath("state.txt").write_text("source", encoding="utf-8")
    target_state = io.write_record(StoredStateRecord(_cdef(), spec["id"], (StorageRef.self_product(role="target"),), derived_from=(source_state.record_id,)).to_envelope())
    io.product_root(target_state.record_id, create=True).joinpath("state.txt").write_text("target", encoding="utf-8")
    adapter = AdapterRecord(
        adapter={"name": "fake.copy"},
        source_record_id=source_state.record_id,
        source_representation_id=spec["id"],
        target_record_id=target_state.record_id,
        target_representation_id=spec["id"],
        produced_records=(target_state.record_id,),
        derived_from=(source_state.record_id,),
    )
    adapter_ref = io.write_record(adapter.to_envelope())

    closure = plan_record_closure(source, seed_records=[target_state.record_id], policy="closure")
    assert ("representation", spec["id"]) in closure.specs
    provenance = plan_record_closure(source, seed_records=[target_state.record_id], policy="provenance")
    assert source_state.record_id in provenance.records
    assert adapter_ref.record_id in provenance.records
    assert target_state.record_id not in provenance.products

    report = copy_record_closure(source, dest, seed_records=[target_state.record_id], options=RecordPolicyOptions(include_products=True))
    dest_io = RecordStoreIO(dest)
    copied = dest_io.read_record(target_state.record_id)
    assert dest_io.resolve_storage_ref(copied["payload"]["storage"][0], record_id=target_state.record_id).joinpath("state.txt").read_text(encoding="utf-8") == "target"
    assert target_state.record_id in report.products_copied
