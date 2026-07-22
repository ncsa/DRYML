import pytest

import dryml.providers as providers
from dryml.formats import json_ready
from dryml.records import AdapterDescriptor, RepresentationRequirement, adapter_descriptors_from_report


def test_representation_inspection_payload_round_trip():
    identity = providers.ProviderIdentity("fake", "1")
    payload = {"representations": [{"representation_spec": {"schema": "dryml.representation.v1", "kind": "fake.raw_state", "payload": {}}, "applies_to": {"record_kinds": ["stored_state"]}, "notes": []}]}
    report = providers.RepresentationInspectionReport(identity, "ok", report_payload=payload)

    round_trip = providers.report_from_data(report.to_data())
    assert json_ready(round_trip.report_payload) == payload


def test_adapter_planning_payload_probe_round_trip_and_extraction():
    identity = providers.ProviderIdentity("fake", "1")
    descriptor = AdapterDescriptor("fake.normalize", RepresentationRequirement(kind="fake.raw_state"), RepresentationRequirement(kind="fake.normalized_state"))
    report = providers.AdapterPlanningReport(identity, "ok", report_payload={"adapters": [descriptor.to_json()]})
    probe = providers.ProbeReport(reports=(report,))

    round_trip = providers.ProbeReport.from_data(probe.to_data())
    extracted = adapter_descriptors_from_report(round_trip.reports[0])
    assert extracted[0].name == "fake.normalize"
    assert round_trip.annotation_fragments() == ()


def test_probe_report_rejects_string_sequence_fields():
    with pytest.raises(providers.ProviderReportError):
        providers.ProbeReport.from_data({"reports": "not-a-list"})
