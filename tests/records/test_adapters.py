import dryml
import pytest

from dryml.core2.repo import Repo
from dryml.core2.store.dir import DirStore
from dryml.formats.refs import format_cdef_id
from dryml.records import (
    AdapterDescriptor,
    AdapterRegistry,
    RecordValidationError,
    RepresentationRequirement,
    StorageRef,
    StoredStateRecord,
    find_adapter_path,
    make_representation_spec,
    resolve_state_record,
    run_adapter_plan,
)


def _cdef():
    return format_cdef_id("a" * 64)


def _seed(repo_store, *, realization_id=None, output_slot=None):
    raw = make_representation_spec("fake.raw_state", storage_kinds=("product-dir",))
    repo_store.records.write_spec(raw, family="representation")
    source = StoredStateRecord(
        _cdef(),
        raw["id"],
        (StorageRef.self_product(role="source-state"),),
        realization_id=realization_id,
        output_slot=output_slot,
    )
    ref = repo_store.records.write_record(source.to_envelope())
    root = repo_store.records.product_root(ref.record_id, create=True)
    root.joinpath("state.txt").write_text("raw", encoding="utf-8")
    return raw, ref


def test_adapter_descriptor_registry_and_paths(tmp_path):
    store = DirStore(tmp_path / "store")
    raw, _ = _seed(store)
    repo = Repo(stores=[store])
    registry = AdapterRegistry()
    registry.register(AdapterDescriptor("fake.normalize", RepresentationRequirement(kind="fake.raw_state"), RepresentationRequirement(kind="fake.normalized_state"), version="1"))

    zero = resolve_state_record(repo, _cdef(), RepresentationRequirement(kind="fake.raw_state"), adapters=registry).adapter_plan
    assert zero is None
    planned_result = resolve_state_record(repo, _cdef(), RepresentationRequirement(kind="fake.normalized_state"), adapters=registry)
    assert planned_result.status == "requires_adapter"
    assert planned_result.selected is None
    assert planned_result.adapter_source.ref.record_id == _.record_id
    planned = planned_result.adapter_plan
    assert planned.steps[0].descriptor.name == "fake.normalize"
    assert registry.descriptors()[0].to_json()["source"] == {"kind": "fake.raw_state"}


def test_multi_step_cycle_avoidance_and_cost_ordering(tmp_path):
    store = DirStore(tmp_path / "store")
    _seed(store)
    repo = Repo(stores=[store])
    registry = AdapterRegistry()
    registry.register(AdapterDescriptor("cycle", RepresentationRequirement(kind="fake.raw_state"), RepresentationRequirement(kind="fake.raw_state"), cost=0.1))
    registry.register(AdapterDescriptor("to_mid", RepresentationRequirement(kind="fake.raw_state"), RepresentationRequirement(kind="fake.mid_state"), cost=1.0))
    registry.register(AdapterDescriptor("to_target", RepresentationRequirement(kind="fake.mid_state"), RepresentationRequirement(kind="fake.normalized_state"), cost=1.0))
    result = resolve_state_record(repo, _cdef(), RepresentationRequirement(kind="fake.normalized_state"), adapters=registry)
    assert result.status == "requires_adapter"
    assert [step.descriptor.name for step in result.adapter_plan.steps] == ["to_mid", "to_target"]


def test_fake_runner_writes_target_product_and_adapter_lineage(tmp_path):
    store = DirStore(tmp_path / "store")
    _seed(store)
    repo = Repo(stores=[store])
    registry = AdapterRegistry()

    def runner(context):
        context.session.write_text("normalized.txt", "normalized")
        return {}

    registry.register(
        AdapterDescriptor("fake.normalize", RepresentationRequirement(kind="fake.raw_state"), RepresentationRequirement(kind="fake.normalized_state"), version="1"),
        runner=runner,
    )
    plan = resolve_state_record(repo, _cdef(), RepresentationRequirement(kind="fake.normalized_state"), adapters=registry).adapter_plan
    capture = dryml.reporting.CaptureReporter()
    with dryml.config(reporting={"level": "details", "reporter": capture}):
        result = run_adapter_plan(plan, repo=repo, store=store, registry=registry)

    assert result.status == "ok"
    target_ref = result.target_records[-1]
    adapter = store.records.read_record(result.adapter_records[-1].record_id)["payload"]
    target = store.records.read_record(target_ref.record_id)["payload"]
    assert (store.records.products_dir / target_ref.record_id / "normalized.txt").read_text(encoding="utf-8") == "normalized"
    assert target["storage"][0]["kind"] == "product-dir"
    assert "record_id" not in target["storage"][0]
    assert adapter["produced_records"] == [target_ref.record_id]
    assert adapter["derived_from"] == [plan.source_record.ref.record_id]
    assert "dryml.records.adapter.run" in {event.name for event in capture.events}


def test_missing_runner_returns_unsupported(tmp_path):
    store = DirStore(tmp_path / "store")
    _seed(store)
    repo = Repo(stores=[store])
    registry = AdapterRegistry()
    registry.register(AdapterDescriptor("fake.normalize", RepresentationRequirement(kind="fake.raw_state"), RepresentationRequirement(kind="fake.normalized_state")))
    plan = resolve_state_record(repo, _cdef(), RepresentationRequirement(kind="fake.normalized_state"), adapters=registry).adapter_plan
    assert run_adapter_plan(plan, repo=repo, store=store, registry=registry).status == "unsupported"


def test_adapter_target_preserves_managed_realization_ownership(tmp_path):
    store = DirStore(tmp_path / "store")
    realization_id = "realization-v1-" + "1" * 32
    _seed(store, realization_id=realization_id, output_slot="result")
    repo = Repo(stores=[store])
    registry = AdapterRegistry()

    def runner(context):
        context.session.write_text("adapted.txt", "adapted")
        return {}

    registry.register(
        AdapterDescriptor(
            "fake.normalize",
            RepresentationRequirement(kind="fake.raw_state"),
            RepresentationRequirement(kind="fake.normalized_state"),
        ),
        runner=runner,
    )
    plan = resolve_state_record(
        repo,
        _cdef(),
        RepresentationRequirement(kind="fake.normalized_state"),
        adapters=registry,
    ).adapter_plan

    result = run_adapter_plan(plan, repo=repo, store=store, registry=registry)
    target = StoredStateRecord.from_envelope(
        store.records.read_record(result.target_records[-1].record_id)
    )

    assert target.realization_id == realization_id
    assert target.output_slot == "result"


def test_adapter_descriptors_from_report_rejects_string_sequences():
    from dryml.records import adapter_descriptors_from_report

    class Report:
        report_payload = {"adapters": "fake.normalize"}

    with pytest.raises(RecordValidationError):
        adapter_descriptors_from_report(Report())
