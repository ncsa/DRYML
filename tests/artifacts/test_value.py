"""U4 coverage for abstract Artifacts and versioned Value payloads."""

from __future__ import annotations

import inspect
import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dryml.artifacts import Value
from dryml.core import Repo
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore
from dryml.core.utils.general import pickle_load, pickle_save
from dryml.managed import ManagedConfig, managed_operation


_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "artifact_value_v1"
_ENVELOPE_FORMAT = "dryml.artifacts.value"


class BasicValue(Value):
    """Concrete Value implementation whose managed operation stores a result."""

    def __init__(self, source_path: str | None = None):
        self.source_path = source_path

    @property
    def ready(self) -> bool:
        """Return whether a complete Value payload is installed."""

        return self._value_is_present()

    @managed_operation()
    def compute(self, *, managed):
        """Read the optional source and publish its text as the result.

        Args:
            managed: Framework-provided managed operation context.

        Returns:
            ``None`` after installing the complete Value result.

        Raises:
            FileNotFoundError: If the declared source is no longer available.
        """

        if self.source_path is None:
            raise FileNotFoundError("Value source is unavailable.")
        result = Path(self.source_path).read_text(encoding="utf-8")
        self._install_value_payload({
            "format": _ENVELOPE_FORMAT,
            "version": 1,
            "present": True,
            "result": result,
        })


class DomainValue(BasicValue):
    """Value whose result domain is limited to non-negative integers."""

    def _validate_value_result(self, result) -> None:
        """Reject values outside this test Value's documented integer domain."""

        if type(result) is not int or result < 0:
            raise ValueError("Value result is outside the supported domain.")


class HookedValue(DomainValue):
    """Value with an independent Serializable payload contribution."""

    save_calls = 0
    restore_calls = 0

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Write this subclass's independent hook payload exactly once."""

        type(self).save_calls += 1
        Path(dest_dir, "subclass.txt").write_text("subclass", encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Restore this subclass's independent hook payload exactly once."""

        type(self).restore_calls += 1
        assert Path(src_dir, "subclass.txt").read_text(encoding="ascii") == "subclass"


def _install(value: BasicValue, result) -> None:
    """Install a complete test result through Value's protected boundary."""

    value._install_value_payload({
        "format": _ENVELOPE_FORMAT,
        "version": 1,
        "present": True,
        "result": result,
    })


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_artifact_and_value_stay_abstract_and_concrete_values_are_inert(tmp_path):
    """Artifact contracts require managed compute/readiness without loading inputs."""

    from dryml.artifacts import Artifact, ArtifactNotReadyError

    assert inspect.isabstract(Artifact)
    assert inspect.isabstract(Value)
    assert not inspect.isabstract(BasicValue)

    store = DirStore(tmp_path / "store")
    value = BasicValue(repo=Repo(store))
    assert value.ready is False
    with pytest.raises(ArtifactNotReadyError):
        value.value()
    with pytest.raises(FileNotFoundError, match="source is unavailable"):
        value.compute(managed=ManagedConfig(state_repo=store))


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_value_distinguishes_absence_from_a_present_none_result():
    """A complete ``None`` result is readable while an absent result is not."""

    from dryml.artifacts import ArtifactNotReadyError

    value = BasicValue()
    with pytest.raises(ArtifactNotReadyError):
        value.value()

    _install(value, None)
    assert value.ready is True
    assert value.value() is None


@pytest.mark.usefixtures("fixed_snapshot_environment")
@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    (
        ("absent", "absent"),
        ("none", None),
        ("float", 3.5),
        ("array", np.array([1.0, 2.5], dtype=np.float64)),
        ("matrix", np.array([[1, 2], [3, 4]], dtype=np.int64)),
        ("tree", {"labels": ("a", "b"), "scores": [0.25, 0.75]}),
    ),
)
def test_fixed_v1_fixtures_restore_known_value_payloads(fixture_name, expected):
    """Read checked-in v1 payloads rather than generating reader expectations."""

    from dryml.artifacts import ArtifactNotReadyError

    value = BasicValue()
    value.restore_state_from_dir(str(_FIXTURE_ROOT / fixture_name), codec="pkl")

    if fixture_name == "absent":
        assert value.ready is False
        with pytest.raises(ArtifactNotReadyError):
            value.value()
    elif isinstance(expected, np.ndarray):
        assert value.ready is True
        np.testing.assert_array_equal(value.value(), expected)
    else:
        assert value.ready is True
        assert value.value() == expected


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_value_envelope_validation_preserves_existing_result(tmp_path):
    """Invalid, incomplete, and unknown payloads fail before replacing content."""

    value = BasicValue()
    _install(value, "old")
    invalid_payloads = (
        {},
        {"format": _ENVELOPE_FORMAT, "version": True, "present": False, "result": None},
        {"format": _ENVELOPE_FORMAT, "version": 1, "present": 1, "result": None},
        {"format": "dryml.artifacts.other", "version": 1, "present": False, "result": None},
        {"format": _ENVELOPE_FORMAT, "version": 2, "present": False, "result": None},
        {"format": _ENVELOPE_FORMAT, "version": 1, "present": False, "result": "not-none"},
    )

    for index, payload in enumerate(invalid_payloads):
        directory = tmp_path / str(index)
        directory.mkdir()
        pickle_save(payload, directory / "value.pkl")
        with pytest.raises(ValueError):
            value.restore_state_from_dir(str(directory), codec="pkl")
        assert value.value() == "old"

    with pytest.raises(FileNotFoundError):
        value.restore_state_from_dir(str(tmp_path / "missing"), codec="pkl")
    assert value.value() == "old"

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "value.pkl").write_bytes(b"not a dill payload")
    with pytest.raises(Exception):
        value.restore_state_from_dir(str(corrupt), codec="pkl")
    assert value.value() == "old"


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_domain_validation_happens_before_value_payload_assignment(tmp_path):
    """A valid envelope with an invalid domain result leaves prior content intact."""

    value = DomainValue()
    _install(value, 3)
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    pickle_save({
        "format": _ENVELOPE_FORMAT,
        "version": 1,
        "present": True,
        "result": -1,
    }, payload_dir / "value.pkl")

    with pytest.raises(ValueError, match="supported domain"):
        value.restore_state_from_dir(str(payload_dir), codec="pkl")
    assert value.value() == 3


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_invalid_value_restore_uses_core_invalidation_after_hooks_begin(tmp_path):
    """A domain-invalid envelope runs its subclass hook once then invalidates."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    HookedValue.restore_calls = 0
    target = HookedValue(repo=repo)
    target._value_payload = {
        "format": _ENVELOPE_FORMAT,
        "version": 1,
        "present": True,
        "result": -1,
    }
    state = repo.save_object(target)
    _install(target, 3)

    with pytest.raises(RepoLoadError, match="supported domain"):
        repo.restore_state_ref_into(target, state)
    assert HookedValue.restore_calls == 1
    assert target.value() == 3
    assert target._restore_failed is True


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_value_serializable_hooks_write_and_restore_each_file_once(tmp_path):
    """Value's narrow hook and a subclass hook participate once each in the MRO."""

    HookedValue.save_calls = 0
    HookedValue.restore_calls = 0
    repo = Repo(DirStore(tmp_path / "store"))
    value = HookedValue(repo=repo)
    _install(value, 7)

    state = repo.save_object(value)
    assert HookedValue.save_calls == 1

    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")
    assert HookedValue.restore_calls == 1
    assert loaded.value() == 7


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_exact_state_restore_reads_result_without_its_missing_source(tmp_path):
    """Exact Value state remains readable after its non-materializing source disappears."""

    source = tmp_path / "source.txt"
    source.write_text("computed source", encoding="utf-8")
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    value = BasicValue(str(source), repo=repo)
    _install(value, "completed result")
    state = repo.save_object(value)
    source.unlink()

    loaded = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")
    assert loaded.value() == "completed result"
    with pytest.raises(FileNotFoundError):
        loaded.compute(managed=ManagedConfig(state_repo=store))


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_value_results_do_not_change_cdef_identity_or_capture_transient_fields(tmp_path):
    """Results live only in Value state and not in construction identity or attributes."""

    left = BasicValue()
    right = BasicValue()
    left.transient_input = object()
    _install(left, {"nested": [1, 2]})
    _install(right, {"nested": [3, 4]})

    assert left.definition == right.definition
    left.save_state_to_dir(str(tmp_path), codec="pkl")
    payload = pickle_load(tmp_path / "value.pkl")
    assert payload == {
        "format": _ENVELOPE_FORMAT,
        "version": 1,
        "present": True,
        "result": {"nested": [1, 2]},
    }


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_fixed_fixture_restores_in_a_fresh_process_without_sources_or_controls():
    """A new interpreter reads a result-only fixture without input or control authority."""

    script = """
from dryml.artifacts import Value
from dryml.managed import managed_operation
class FixtureValue(Value):
    @property
    def ready(self):
        return self._value_is_present()
    @managed_operation()
    def compute(self, *, managed):
        raise FileNotFoundError('source unavailable')
value = object.__new__(FixtureValue)
value.restore_state_from_dir(__import__('sys').argv[1], codec='pkl')
assert value.value() == {'labels': ('a', 'b'), 'scores': [0.25, 0.75]}
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(_FIXTURE_ROOT / "tree")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_fixed_fixture_manifest_records_independent_provenance():
    """Fixture metadata records the fixed v1 envelope source and byte hashes."""

    manifest = json.loads((_FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == _ENVELOPE_FORMAT
    assert manifest["version"] == 1
    assert manifest["generator"] == "generate.py"
    assert set(manifest["fixtures"]) == {"absent", "none", "float", "array", "matrix", "tree"}
    for fixture in manifest["fixtures"].values():
        fixture_path = _FIXTURE_ROOT / fixture["file"]
        assert sha256(fixture_path.read_bytes()).hexdigest() == fixture["sha256"]
