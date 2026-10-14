"""Persistent authority and validation for ML workflow qualification fixtures.

This module is intentionally import-safe: preparation is the only path that
opens a Store, and optional ML packages are never imported here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType

from dryml.core import StateRef
from dryml.locking import LockError, interprocess_lock


MANIFEST_FORMAT = "dryml-ml-workflow-fixtures"
MANIFEST_VERSION = 2
BASELINE_PATH = Path(__file__).with_name("ml_workflow_baseline.json")
REQUIRED_ENVIRONMENT_KEYS = (
    "python", "dryml", "flax", "jax", "jaxlib", "optax", "pandas", "pyarrow",
    "tensorflow", "tensorflow_datasets", "torch",
)
_MINIMUM_PYARROW = (25, 0, 1)


class FixtureManifestError(ValueError):
    """Report malformed, incompatible, or unsafe fixture authority."""


class QualificationUnrun(RuntimeError):
    """Report an explicitly requested qualification whose prerequisites are absent."""


def _canonical_json(value: object) -> bytes:
    """Return deterministic JSON bytes for a closed qualification record."""

    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _json_value(value: object) -> object:
    """Return mutable JSON-compatible data from immutable qualification values."""

    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _freeze_json(value: object) -> object:
    """Freeze closed JSON data recursively after validation."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def config_digest(config: Mapping[str, object]) -> str:
    """Return the deterministic digest of the fixed KTD11 baseline."""

    return hashlib.sha256(_canonical_json(config)).hexdigest()


def _closed_mapping(value: object, *, name: str) -> dict[str, object]:
    """Return a detached JSON-object mapping, rejecting non-finite values."""

    if not isinstance(value, dict):
        raise FixtureManifestError(f"{name} must be a JSON object.")
    try:
        copied = json.loads(_canonical_json(value))
    except (TypeError, ValueError) as error:
        raise FixtureManifestError(f"{name} is not closed JSON data.") from error
    if not isinstance(copied, dict):  # Defensive: json conversion above preserves dict.
        raise FixtureManifestError(f"{name} must be a JSON object.")
    return copied


def load_baseline(path: str | Path = BASELINE_PATH) -> dict[str, object]:
    """Load the fixed, closed KTD11 baseline.

    Args:
        path: Baseline JSON path; the checked-in authority is the default.

    Returns:
        A detached exact baseline mapping.

    Raises:
        FixtureManifestError: If the baseline is unavailable, malformed, or does
            not contain the complete fixed KTD11 schema.
    """

    try:
        value = json.loads(Path(path).read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureManifestError("ML workflow baseline is unavailable or malformed.") from error
    value = _closed_mapping(value, name="ML workflow baseline")
    required = {
        "format", "version", "mnist", "w1", "w2", "w3", "initialization",
        "batch_size", "checkpoint_every_steps", "cpu_budget_seconds",
        "cpu_peak_rss_bytes", "case_store_budget_bytes",
    }
    if set(value) != required or value.get("format") != "dryml-ml-workflow-baseline" or value.get("version") != 2:
        raise FixtureManifestError("ML workflow baseline format is unsupported.")
    if not all(isinstance(value[key], dict) for key in ("mnist", "w1", "w2", "w3", "initialization")):
        raise FixtureManifestError("ML workflow baseline nested schema is malformed.")
    return value


def _fixed_baseline(value: Mapping[str, object] | None) -> dict[str, object]:
    """Reject caller-selected hyperparameters instead of treating them as a baseline."""

    fixed = load_baseline()
    candidate = fixed if value is None else _closed_mapping(dict(value), name="ML workflow baseline")
    if candidate != fixed:
        raise FixtureManifestError("ML workflow qualification rejects baseline/config drift.")
    return fixed


def installed_environment() -> dict[str, str]:
    """Return required version evidence without importing optional frameworks.

    Raises:
        QualificationUnrun: If a required distribution is not installed.
    """

    distributions = {
        "python": None,
        "dryml": "dryml",
        "flax": "flax",
        "jax": "jax",
        "jaxlib": "jaxlib",
        "optax": "optax",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "tensorflow": "tensorflow",
        "tensorflow_datasets": "tensorflow-datasets",
        "torch": "torch",
    }
    values = {"python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"}
    missing = []
    for key, distribution in distributions.items():
        if distribution is None:
            continue
        try:
            values[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(key)
    if missing:
        raise QualificationUnrun("Qualification prerequisites are unavailable: " + ", ".join(missing))
    return values


def _version_tuple(value: str) -> tuple[int, ...]:
    """Return the numeric release prefix used by the narrow pyarrow gate."""

    try:
        return tuple(int(part) for part in value.split("+", 1)[0].split(".")[:3])
    except ValueError as error:
        raise QualificationUnrun("Installed pyarrow version is unsupported.") from error


def require_supported_pyarrow() -> str:
    """Require the supported Parquet fixture dependency before any builder work.

    Returns:
        The installed pyarrow distribution version.

    Raises:
        QualificationUnrun: If pyarrow is absent or older than the packaging
            contract. This function imports no optional framework and performs no
            Store or manifest mutation.
    """

    try:
        version = importlib.metadata.version("pyarrow")
    except importlib.metadata.PackageNotFoundError as error:
        raise QualificationUnrun("Qualification prerequisites are unavailable: pyarrow") from error
    if _version_tuple(version) < _MINIMUM_PYARROW:
        raise QualificationUnrun("Qualification requires pyarrow>=25.0.1.")
    return version


def _environment(value: Mapping[str, str]) -> dict[str, str]:
    """Validate the complete, exact environment evidence schema."""

    if not isinstance(value, Mapping) or set(value) != set(REQUIRED_ENVIRONMENT_KEYS):
        raise FixtureManifestError("Qualification environment evidence has missing or extra keys.")
    if not all(type(key) is str and type(item) is str and item for key, item in value.items()):
        raise FixtureManifestError("Qualification environment evidence must contain nonempty strings.")
    return dict(value)


def _absolute(path: str | Path) -> Path:
    """Return a stable absolute local path without creating it."""

    return Path(path).expanduser().resolve(strict=False)


def _reference_to_json(value: object) -> dict[str, object]:
    """Encode StateRef data through the manifest's legacy v2 grammar."""

    from dryml.core.reference_json import _encode_legacy_reference_json

    return _encode_legacy_reference_json(
        StateRef.from_data(value), version=MANIFEST_VERSION,
    )


def _reference_from_json(value: object) -> object:
    """Decode the manifest's legacy v2 StateRef tree to ordinary data."""

    from dryml.core.reference_json import _decode_legacy_reference_json

    return _decode_legacy_reference_json(
        value, kind="state_ref", version=MANIFEST_VERSION,
    ).to_data()


@dataclass(frozen=True, slots=True)
class FixtureReferences:
    """Two distinct exact completed W3 CachedDataset references.

    Args:
        numpy: Exact NumPy-codec cache StateRef used by every W3 case.
        parquet: Exact Parquet-codec cache StateRef used only for equivalence.

    Store-dependent codec/readiness validation is performed by
    :func:`validate_fixture_references` at preparation and execution boundaries.
    """

    numpy: StateRef
    parquet: StateRef

    def __post_init__(self) -> None:
        """Reject floating, malformed, or aliased reference authority."""

        if type(self.numpy) is not StateRef or type(self.parquet) is not StateRef:
            raise FixtureManifestError("W3 fixtures require exact StateRefs.")
        if self.numpy == self.parquet:
            raise FixtureManifestError("W3 NumPy and Parquet fixtures must be distinct complete states.")

    def to_data(self) -> dict[str, object]:
        """Encode exact references without opening Store payloads."""

        return {"numpy": _reference_to_json(self.numpy.to_data()), "parquet": _reference_to_json(self.parquet.to_data())}

    @classmethod
    def from_data(cls, value: object) -> "FixtureReferences":
        """Decode the closed two-reference record without opening a Store."""

        if not isinstance(value, dict) or set(value) != {"numpy", "parquet"}:
            raise FixtureManifestError("W3 fixture references require numpy and parquet records.")
        try:
            return cls(
                StateRef.from_data(_reference_from_json(value["numpy"])),
                StateRef.from_data(_reference_from_json(value["parquet"])),
            )
        except (TypeError, ValueError) as error:
            raise FixtureManifestError("W3 fixture references are malformed.") from error


@dataclass(frozen=True, slots=True)
class TFDSAuthority:
    """Resolved local MNIST builder identity bound to one fixture manifest.

    ``data_dir`` is local-only manifest authority. It is deliberately not copied
    into portable case or evidence records; execution supplies and verifies the
    same resolved root when loading this local manifest.
    """

    data_dir: Path
    builder: str
    config: str
    version: str
    content_digest: str

    def __post_init__(self) -> None:
        """Validate complete local builder/config/version/content identity."""

        if not all(type(value) is str and value for value in (
                self.builder, self.config, self.version, self.content_digest)):
            raise FixtureManifestError("TFDS authority requires nonempty builder identity fields.")
        if len(self.content_digest) != 64:
            raise FixtureManifestError("TFDS authority content digest is malformed.")
        object.__setattr__(self, "data_dir", _absolute(self.data_dir))

    def to_data(self) -> dict[str, str]:
        """Encode local root plus machine-readable builder identity for a manifest."""

        return {
            "data_dir": os.fspath(self.data_dir),
            "builder": self.builder, "config": self.config,
            "version": self.version, "content_digest": self.content_digest,
        }

    @classmethod
    def from_data(cls, value: object, *, data_dir: str | Path) -> "TFDSAuthority":
        """Decode portable identity with explicitly supplied local root authority."""

        if not isinstance(value, dict) or set(value) != {"data_dir", "builder", "config", "version", "content_digest"}:
            raise FixtureManifestError("TFDS authority record is malformed.")
        if type(value["data_dir"]) is not str or _absolute(value["data_dir"]) != _absolute(data_dir):
            raise FixtureManifestError("ML workflow fixture manifest does not authorize this TFDS root.")
        return cls(_absolute(data_dir), **{key: value[key] for key in ("builder", "config", "version", "content_digest")})


def _tfds_content_digest(builder) -> str:
    """Stream a domain-separated digest of prepared regular-file contents.

    Relative paths, declared sizes, and every file byte are hashed in sorted
    order. Symlinks and non-regular entries are rejected so the persisted TFDS
    authority cannot silently depend on ambiguous filesystem resolution.
    """

    root = Path(builder.data_path)
    try:
        root_mode = root.lstat().st_mode
    except OSError as error:
        raise QualificationUnrun("Prepared MNIST data is absent from the selected TFDS root.") from error
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise QualificationUnrun("Prepared MNIST data is absent from the selected TFDS root.")
    entries: list[tuple[str, Path, int]] = []
    for path in sorted(root.rglob("*")):
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise QualificationUnrun("Prepared MNIST data cannot be read consistently.") from error
        if stat.S_ISLNK(mode):
            raise QualificationUnrun("Prepared MNIST data contains an ambiguous symlink.")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise QualificationUnrun("Prepared MNIST data contains a non-regular entry.")
        entries.append((path.relative_to(root).as_posix(), path, path.stat().st_size))
    if not entries:
        raise QualificationUnrun("Prepared MNIST data has no content files in the selected TFDS root.")
    digest = hashlib.sha256()
    digest.update(b"dryml-ml-workflow-tfds-content-v1\0")
    for relative, path, size in entries:
        encoded = relative.encode("utf-8")
        digest.update(b"path\0" + len(encoded).to_bytes(8, "big") + encoded)
        digest.update(b"size\0" + size.to_bytes(8, "big"))
        digest.update(b"content\0")
        consumed = 0
        try:
            with path.open("rb") as file:
                while chunk := file.read(1024 * 1024):
                    digest.update(chunk)
                    consumed += len(chunk)
        except OSError as error:
            raise QualificationUnrun("Prepared MNIST data cannot be read consistently.") from error
        if consumed != size:
            raise QualificationUnrun("Prepared MNIST data changed while its content was being hashed.")
    return digest.hexdigest()


def prepare_tfds_authority(data_dir: str | Path, *, allow_download: bool) -> TFDSAuthority:
    """Prepare MNIST only when explicitly allowed and return exact local authority.

    The TFDS import and optional download are confined to explicit preparation.
    Execution uses :func:`validate_tfds_authority` with downloading disabled.
    """

    if type(allow_download) is not bool:
        raise TypeError("allow_download must be a bool.")
    root = _absolute(data_dir)
    if not allow_download:
        raise QualificationUnrun("TFDS preparation requires explicit allow_download=True.")
    import tensorflow_datasets as tfds

    builder = tfds.builder("mnist", data_dir=os.fspath(root))
    builder.download_and_prepare()
    config = getattr(builder.builder_config, "name", None) or "default"
    return TFDSAuthority(root, builder.name, config, str(builder.version), _tfds_content_digest(builder))


def validate_tfds_authority(authority: TFDSAuthority) -> None:
    """Require the exact prepared MNIST authority without permitting download."""

    import tensorflow_datasets as tfds

    try:
        builder = tfds.builder(authority.builder, data_dir=os.fspath(authority.data_dir))
    except Exception as error:
        raise QualificationUnrun("Selected TFDS root cannot build MNIST without download.") from error
    config = getattr(builder.builder_config, "name", None) or "default"
    observed = (builder.name, config, str(builder.version), _tfds_content_digest(builder))
    expected = (authority.builder, authority.config, authority.version, authority.content_digest)
    if observed != expected:
        raise QualificationUnrun("Selected TFDS root does not match manifest MNIST authority.")


def validate_tfds_content_authority(authority: TFDSAuthority) -> None:
    """Verify prepared TFDS bytes without importing TFDS in an orchestrator.

    The coordinator may inspect only the manifest-declared builder directory and
    its content digest.  A worker later uses :func:`validate_tfds_authority` to
    additionally reconstruct TFDS's builder metadata before it opens data.
    """

    data_path = authority.data_dir / authority.builder / authority.version
    builder = type("PreparedBuilderPath", (), {"data_path": data_path})()
    if _tfds_content_digest(builder) != authority.content_digest:
        raise FixtureManifestError("Selected TFDS content drifted from manifest authority.")


@dataclass(frozen=True, slots=True)
class FixtureManifest:
    """Closed persistent authority for one fixed ML workflow fixture set.

    Args:
        fixture_store: Caller-selected absolute Store containing retained caches.
        baseline: The exact checked-in KTD11 baseline.
        references: Distinct exact W3 cache StateRefs.
        environment: Complete required version evidence.
        tfds: Exact local MNIST builder/config/version/content authority.
    """

    fixture_store: Path
    baseline: Mapping[str, object]
    references: FixtureReferences
    environment: Mapping[str, str]
    tfds: TFDSAuthority

    def __post_init__(self) -> None:
        """Validate closed construction-time authority before publication."""

        store = _absolute(self.fixture_store)
        if not store.is_absolute():  # pragma: no cover - Path.resolve is absolute.
            raise FixtureManifestError("Fixture Store path must be absolute.")
        if not isinstance(self.references, FixtureReferences):
            raise FixtureManifestError("Fixture manifest lacks exact W3 references.")
        if not isinstance(self.tfds, TFDSAuthority):
            raise FixtureManifestError("Fixture manifest lacks prepared TFDS authority.")
        object.__setattr__(self, "fixture_store", store)
        object.__setattr__(self, "baseline", _freeze_json(_fixed_baseline(self.baseline)))
        object.__setattr__(self, "environment", _freeze_json(_environment(self.environment)))

    @property
    def digest(self) -> str:
        """Return stable identity for this manifest's closed authority."""

        return hashlib.sha256(_canonical_json(self.to_data())).hexdigest()

    def to_data(self) -> dict[str, object]:
        """Encode the closed manifest grammar for fresh-process consumption."""

        return {
            "format": MANIFEST_FORMAT, "version": MANIFEST_VERSION,
            "fixture_store": os.fspath(self.fixture_store), "baseline": _json_value(self.baseline),
            "config_digest": config_digest(self.baseline), "references": self.references.to_data(),
            "environment": dict(self.environment), "tfds": self.tfds.to_data(),
        }

    @classmethod
    def from_data(cls, value: object, *, fixture_store: str | Path, tfds_data_dir: str | Path, environment: Mapping[str, str]) -> "FixtureManifest":
        """Decode and validate manifest JSON against selected Store/environment authority."""

        required = {"format", "version", "fixture_store", "baseline", "config_digest", "references", "environment", "tfds"}
        if not isinstance(value, dict) or set(value) != required:
            raise FixtureManifestError("ML workflow fixture manifest fields are invalid.")
        if value["format"] != MANIFEST_FORMAT or value["version"] != MANIFEST_VERSION:
            raise FixtureManifestError("ML workflow fixture manifest version is unsupported.")
        if not isinstance(value["baseline"], dict):
            raise FixtureManifestError("ML workflow fixture baseline is malformed.")
        baseline = _fixed_baseline(value["baseline"])
        if type(value["config_digest"]) is not str or value["config_digest"] != config_digest(baseline):
            raise FixtureManifestError("ML workflow fixture configuration digest is invalid.")
        if type(value["fixture_store"]) is not str or _absolute(value["fixture_store"]) != _absolute(fixture_store):
            raise FixtureManifestError("ML workflow fixture manifest does not authorize this Store.")
        recorded = _environment(value["environment"])
        if recorded != _environment(environment):
            raise FixtureManifestError("ML workflow fixture environment evidence is incompatible.")
        return cls(_absolute(fixture_store), baseline, FixtureReferences.from_data(value["references"]), recorded, TFDSAuthority.from_data(value["tfds"], data_dir=tfds_data_dir))


def load_manifest(manifest_path: str | Path, *, fixture_store: str | Path, tfds_data_dir: str | Path, environment: Mapping[str, str] | None = None) -> FixtureManifest:
    """Load manifest authority and require complete compatible version evidence.

    This reads JSON only; it does not open the selected Store, import frameworks,
    download data, or construct datasets. Use :func:`preflight_manifest` before
    running an opted-in case.
    """

    expected_environment = installed_environment() if environment is None else _environment(environment)
    try:
        value = json.loads(Path(manifest_path).read_text(encoding="ascii"))
    except FileNotFoundError as error:
        raise FixtureManifestError("ML workflow fixture manifest is missing.") from error
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureManifestError("ML workflow fixture manifest is malformed.") from error
    return FixtureManifest.from_data(value, fixture_store=fixture_store, tfds_data_dir=tfds_data_dir, environment=expected_environment)


def _lock_paths(manifest_path: Path, store: Path) -> tuple[Path, Path]:
    """Return deterministic native locks for both manifest and Store authority.

    A manifest-only key would allow concurrent preparation through another
    manifest name, while a Store-only key would allow concurrent publication to
    one manifest. Acquiring both in lexical order prevents either overlap.
    """

    manifest_identity = hashlib.sha256(str(manifest_path).encode("utf-8")).hexdigest()
    store_identity = hashlib.sha256(str(store).encode("utf-8")).hexdigest()
    paths = (
        manifest_path.parent / f".ml_workflow_manifest_{manifest_identity}.lock",
        store.parent / f".ml_workflow_store_{store_identity}.lock",
    )
    return tuple(sorted(paths, key=os.fspath))


def _fsync_directory(path: Path) -> None:
    """Synchronize a POSIX directory entry after non-replacing publication."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as error:
        raise FixtureManifestError("Fixture manifest directory cannot be synchronized.") from error
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_manifest(path: Path, manifest: FixtureManifest) -> None:
    """Publish one fully fsynced private manifest without replacing a final path."""

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(_canonical_json(manifest.to_data()).decode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FixtureManifestError("Refusing to replace an existing fixture manifest.") from error
        _fsync_directory(path.parent)
    finally:
        # The temporary name is owned by this invocation only; final authority is never removed.
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def prepare_manifest(
        manifest_path: str | Path, *, fixture_store: str | Path,
        build: Callable[[Path, Mapping[str, object]], FixtureReferences],
        environment: Mapping[str, str], tfds_data_dir: str | Path, allow_download: bool = False,
        baseline: Mapping[str, object] | None = None,
) -> FixtureManifest:
    """Prepare, validate, and atomically publish caller-owned fixture authority.

    The manifest and Store are serialized under one native advisory lock. Existing
    manifests and nonempty Stores fail closed; an interrupted builder leaves no
    final manifest and the populated Store remains unavailable for replacement.
    """

    if type(allow_download) is not bool:
        raise TypeError("allow_download must be a bool.")
    if not callable(build):
        raise TypeError("build must be an explicit fixture preparation callable.")
    # This gate is intentionally before directory creation, locks, or builder
    # invocation so unsupported pyarrow leaves caller authority untouched.
    require_supported_pyarrow()
    path, store = _absolute(manifest_path), _absolute(fixture_store)
    selected_baseline, selected_environment = _fixed_baseline(baseline), _environment(environment)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ExitStack() as locks:
            for lock_path in _lock_paths(path, store):
                locks.enter_context(interprocess_lock(lock_path))
            if path.exists():
                raise FixtureManifestError("Refusing to replace an existing fixture manifest.")
            if store.exists() and any(store.iterdir()):
                raise FixtureManifestError("Refusing to prepare into a nonempty fixture Store.")
            if not allow_download:
                raise QualificationUnrun("Fixture preparation requires explicit allow_download=True.")
            tfds = prepare_tfds_authority(tfds_data_dir, allow_download=True)
            references = build(store, selected_baseline)
            if not isinstance(references, FixtureReferences):
                raise FixtureManifestError("Fixture builder did not return exact W3 StateRefs.")
            manifest = FixtureManifest(store, selected_baseline, references, selected_environment, tfds)
            preflight_manifest(manifest)
            # Validate the same serialized representation that future processes consume.
            FixtureManifest.from_data(manifest.to_data(), fixture_store=store, tfds_data_dir=tfds.data_dir, environment=selected_environment)
            _publish_manifest(path, manifest)
            return manifest
    except LockError as error:
        raise FixtureManifestError("Fixture preparation could not acquire native Store authority.") from error


def _tree_signature(value: object) -> object:
    """Return exact container order and leaf dtype/shape facts for cache comparison."""

    import numpy as np

    if isinstance(value, dict):
        return ("dict", tuple((key, _tree_signature(item)) for key, item in value.items()))
    if isinstance(value, tuple):
        return ("tuple", tuple(_tree_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_tree_signature(item) for item in value))
    array = np.asarray(value)
    return ("leaf", array.dtype.str, array.shape)


def _assert_exact_values(left: object, right: object) -> None:
    """Require equal ordered structure, dtype, shape, and leaf values."""

    import numpy as np

    if _tree_signature(left) != _tree_signature(right):
        raise FixtureManifestError("W3 cache leaf structure, order, dtype, or shape differs.")
    if isinstance(left, dict):
        for key in left:
            _assert_exact_values(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        for first, second in zip(left, right):
            _assert_exact_values(first, second)
    else:
        try:
            np.testing.assert_equal(np.asarray(left), np.asarray(right))
        except AssertionError as error:
            raise FixtureManifestError("W3 cache values differ.") from error


def validate_fixture_references(references: FixtureReferences, *, repo) -> None:
    """Validate completed distinct codecs and exact W3 cache equivalence in a Repo.

    Args:
        references: Manifest's distinct exact cache StateRefs.
        repo: Open Repo bound to the caller-selected fixture Store.

    Raises:
        FixtureManifestError: If completion, definition, codec, spec, structure,
            dtype, order, or values fail the fixed W3 fixture contract.
    """

    from dryml.artifacts import CachedDataset

    try:
        numpy_cache = repo.load_state_ref(references.numpy, reuse_live="never")
        parquet_cache = repo.load_state_ref(references.parquet, reuse_live="never")
    except Exception as error:
        raise FixtureManifestError("W3 fixture StateRefs cannot be restored from selected Store authority.") from error
    if not isinstance(numpy_cache, CachedDataset) or not isinstance(parquet_cache, CachedDataset):
        raise FixtureManifestError("W3 fixture references must restore CachedDataset states.")
    if not numpy_cache.ready or not parquet_cache.ready:
        raise FixtureManifestError("W3 fixture CachedDataset states must be completed.")
    try:
        numpy_codec = numpy_cache._read_generation().payload["codec"]
        parquet_codec = parquet_cache._read_generation().payload["codec"]
    except (AttributeError, KeyError, TypeError) as error:
        raise FixtureManifestError("W3 fixture cache completion evidence is malformed.") from error
    if numpy_codec != "numpy" or parquet_codec != "parquet":
        raise FixtureManifestError("W3 fixture StateRefs must be distinct NumPy and Parquet codec states.")
    if numpy_cache.spec != parquet_cache.spec:
        raise FixtureManifestError("W3 cache persisted specs differ.")
    numpy_values, parquet_values = tuple(numpy_cache), tuple(parquet_cache)
    if len(numpy_values) != len(parquet_values):
        raise FixtureManifestError("W3 NumPy and Parquet caches have different cardinalities.")
    for left, right in zip(numpy_values, parquet_values):
        _assert_exact_values(left, right)


def validate_fixture_reference_metadata(references: FixtureReferences, *, store) -> None:
    """Validate retained W3 receipt closure from Store metadata only.

    This is the coordinator-side preflight.  It deliberately checks immutable
    StateRef records, definition records, and snapshot metadata without opening
    local-state payloads, restoring CachedDatasets, or iterating fixture data.
    The worker repeats the payload/codec equivalence check immediately before it
    materializes a selected workload.
    """

    from dryml.core.metadata import read_snapshot_metadata
    from dryml.core.store.records import DefinitionRecord

    for reference in (references.numpy, references.parquet):
        record = store.read_state_ref_record(reference.digest())
        if record is None or record.state_ref != reference:
            raise FixtureManifestError("W3 fixture StateRef receipt is absent or inconsistent.")
        definition = store.read_definition_record(DefinitionRecord(reference.object.definition).digest)
        if definition is None:
            raise FixtureManifestError("W3 fixture definition receipt is absent.")
        for state_ref in reference.states.values():
            # State hashes are not independently addressable StateRefs; the root
            # record and snapshot metadata bind their complete exact closure.
            if not isinstance(state_ref, str) or len(state_ref) != 64:
                raise FixtureManifestError("W3 fixture StateRef closure is malformed.")
        try:
            metadata = read_snapshot_metadata(store.get_snapshot_directory(reference))
        except Exception as error:
            raise FixtureManifestError("W3 fixture snapshot metadata is absent or corrupt.") from error
        if metadata.state_ref != reference:
            raise FixtureManifestError("W3 fixture snapshot metadata names another receipt.")


def preflight_manifest(manifest: FixtureManifest) -> None:
    """Open selected Store authority and validate retained fixture completion.

    This execution-boundary check is required before real qualification. It does
    not construct TFDS datasets or frameworks, and it never writes the Store.
    """

    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    if not manifest.fixture_store.is_dir():
        raise QualificationUnrun("Selected fixture Store is unavailable.")
    validate_tfds_authority(manifest.tfds)
    validate_fixture_references(manifest.references, repo=Repo(DirStore(manifest.fixture_store)))


def verify_codec_equivalence(references: FixtureReferences, *, load: Callable[[StateRef], Sequence[object]]) -> None:
    """Verify retained cache values with an explicit caller loader.

    This test seam retains the same strict leaf comparison as the real Store
    preflight and never regenerates seed-derived samples.
    """

    numpy_values, parquet_values = tuple(load(references.numpy)), tuple(load(references.parquet))
    if len(numpy_values) != len(parquet_values):
        raise FixtureManifestError("W3 NumPy and Parquet caches have different cardinalities.")
    for left, right in zip(numpy_values, parquet_values):
        _assert_exact_values(left, right)


__all__ = [
    "BASELINE_PATH", "FixtureManifest", "FixtureManifestError", "FixtureReferences",
    "MANIFEST_FORMAT", "MANIFEST_VERSION", "QualificationUnrun", "REQUIRED_ENVIRONMENT_KEYS", "TFDSAuthority",
    "config_digest", "installed_environment", "load_baseline", "load_manifest", "preflight_manifest",
    "prepare_manifest", "prepare_tfds_authority", "require_supported_pyarrow", "validate_fixture_reference_metadata", "validate_fixture_references", "validate_tfds_authority", "validate_tfds_content_authority", "verify_codec_equivalence",
]
