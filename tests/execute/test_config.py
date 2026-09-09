from __future__ import annotations

import math
from pathlib import Path

import pytest

from dryml.execute.config import BackendConfig
from dryml.environments.specs import CurrentEnvironmentSpec


class FakeConfig(BackendConfig):
    """Inert test configuration with no concrete backend."""

    def create_backend(self):
        raise AssertionError("config construction must not create a backend")


def test_config_defaults_are_inert_and_nested_values_are_frozen(tmp_path: Path):
    """Common defaults validate without inspecting the filesystem or launching work."""
    env = {"TOKEN": "secret"}
    candidate = CurrentEnvironmentSpec()
    config = FakeConfig(env_vars=env, environment_candidates=[candidate], environment_search_roots=[tmp_path])

    env["TOKEN"] = "changed"
    assert config.admission_timeout == 30.0
    assert config.spool_limit_bytes == 4_294_967_296
    assert config.env_vars == {"TOKEN": "secret"}
    assert config.environment_candidates == (candidate,)
    assert config.environment_search_roots == (tmp_path,)
    with pytest.raises(TypeError):
        config.env_vars["NEW"] = "value"


@pytest.mark.parametrize("value", [0, -1, True, math.inf, math.nan, "1"])
def test_config_rejects_invalid_durations(value: object):
    """Duration controls accept only finite positive numeric seconds."""
    with pytest.raises((TypeError, ValueError)):
        FakeConfig(admission_timeout=value)


@pytest.mark.parametrize("field", ["spool_limit_bytes", "spool_file_limit", "preflight_limit", "invocation_limit_bytes"])
def test_config_rejects_boolean_or_invalid_operational_limits(field: str):
    """Count and byte controls reject booleans and non-positive values."""
    with pytest.raises((TypeError, ValueError)):
        FakeConfig(**{field: True})
    with pytest.raises((TypeError, ValueError)):
        FakeConfig(**{field: 0})


def test_config_validates_cross_field_limits_and_allows_nondefault_values():
    """Related public limits are checked without hidden fixed-size caps."""
    with pytest.raises(ValueError, match="spool_limit_bytes"):
        FakeConfig(spool_limit_bytes=9, invocation_limit_bytes=5, result_limit_bytes=5)
    with pytest.raises(ValueError, match="spool_file_limit"):
        FakeConfig(spool_file_limit=1)
    with pytest.raises(ValueError, match="output_frame_limit_bytes"):
        FakeConfig(output_frame_limit_bytes=9, live_output_queue_limit_bytes=8)
    with pytest.raises(ValueError, match="control_header_limit_bytes"):
        FakeConfig(control_header_limit_bytes=9, admission_message_limit_bytes=8)

    config = FakeConfig(
        spool_limit_bytes=100,
        spool_file_limit=4,
        preflight_limit=2,
        invocation_limit_bytes=40,
        result_limit_bytes=60,
        control_header_limit_bytes=32,
        admission_message_limit_bytes=64,
        output_frame_limit_bytes=16,
        live_output_queue_limit_bytes=16,
    )
    assert config.spool_limit_bytes == 100


@pytest.mark.parametrize("value", [True, math.inf, math.nan, pytest.param(10**10000, id="huge-int")])
def test_config_rejects_unrepresentable_or_nonfinite_duration_without_overflow(value: object):
    """Duration validation reports ordinary validation errors for huge integers."""
    with pytest.raises((TypeError, ValueError)):
        FakeConfig(admission_timeout=value)


@pytest.mark.parametrize("field", ["spool_limit_bytes", "result_limit_bytes", "control_header_limit_bytes", "owner_envelope_limit_bytes", "output_limit_bytes", "diagnostic_issue_limit"])
def test_every_numeric_setting_accepts_smaller_and_larger_valid_values(field: str):
    """Public numeric controls have no hidden default-sized validation ceiling."""
    value = 2 if field == "diagnostic_issue_limit" else 2_000_000
    kwargs = {field: value}
    if field == "spool_limit_bytes":
        kwargs.update(invocation_limit_bytes=1, result_limit_bytes=1)
    elif field == "result_limit_bytes":
        kwargs.update(spool_limit_bytes=2_000_000 + 67_108_864)
    elif field == "control_header_limit_bytes":
        kwargs.update(admission_message_limit_bytes=value)
    config = FakeConfig(**kwargs)
    assert getattr(config, field) == value


def test_config_rejects_non_spec_candidates_and_header_values_unrepresentable_by_wire():
    """Owner candidates and four-byte frame headers use their concrete contracts."""
    with pytest.raises(TypeError, match="EnvironmentSpec"):
        FakeConfig(environment_candidates=(object(),))
    with pytest.raises(ValueError, match="4-byte"):
        FakeConfig(control_header_limit_bytes=1 << 40, admission_message_limit_bytes=1 << 40)


@pytest.mark.parametrize(
    ("field", "small", "large", "related"),
    [
        ("spool_limit_bytes", 134_217_728, 400_000_000, {}),
        ("spool_file_limit", 2, 256, {}),
        ("preflight_limit", 1, 32, {}),
        ("invocation_limit_bytes", 1, 100_000_000, {"spool_limit_bytes": 200_000_000}),
        ("result_limit_bytes", 1, 100_000_000, {"spool_limit_bytes": 200_000_000}),
        ("control_header_limit_bytes", 1, 2_000_000, {"admission_message_limit_bytes": 2_000_000}),
        ("owner_envelope_limit_bytes", 1, 100_000_000, {}),
        ("admission_message_limit_bytes", 1_048_576, 100_000_000, {}),
        ("output_frame_limit_bytes", 1, 1_000_000, {"live_output_queue_limit_bytes": 1_000_000}),
        ("output_limit_bytes", 1, 100_000_000, {}),
        ("live_output_queue_limit_bytes", 65_536, 100_000_000, {}),
        ("diagnostic_text_limit_bytes", 1, 100_000_000, {}),
        ("diagnostic_issue_limit", 1, 100_000, {}),
        ("discovery_candidate_limit", 1, 100_000, {}),
        ("environment_search_depth", 0, 100_000, {}),
    ],
)
def test_every_count_and_byte_control_uses_the_supplied_valid_limits(field, small, large, related):
    """Every public count/byte control accepts smaller and larger valid values."""
    for value in (small, large):
        config = FakeConfig(**related, **{field: value})
        assert getattr(config, field) == value
