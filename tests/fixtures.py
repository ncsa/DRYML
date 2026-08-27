import pytest
import tempfile
import uuid
import os
import io
from pathlib import Path
from typing import Literal, Sequence
from dataclasses import dataclass
from dryml.core.store.store import Store
from dryml.core.repo import make_store


StoreDef = Literal["directory", "zip_filepath", "buffer", "zip_buffer"]


# -------------------------
# Layer 1: resource creation
# -------------------------

@dataclass
class StoreResource:
    kind: StoreDef
    root: Path | None
    resource: object  # str path OR IOBase

    def rewind(self) -> None:
        # only meaningful for file-like objects
        if hasattr(self.resource, "seek"):
            try:
                self.resource.seek(0)
            except Exception:
                pass

    def close(self) -> None:
        # close only file-like resources we own
        if self.kind in ("buffer", "zip_buffer") and hasattr(self.resource, "close"):
            try:
                self.resource.close()
            except Exception:
                pass


def build_store_resource(tmp_path_factory, kind: StoreDef, *, prefix: str = "store") -> StoreResource:
    root: Path | None = None

    if kind == "directory":
        root = tmp_path_factory.mktemp(f"{prefix}_{kind}_{uuid.uuid4().hex[:8]}")
        d = root / "repo_dir"
        d.mkdir()
        resource = str(d)

    elif kind == "zip_filepath":
        root = tmp_path_factory.mktemp(f"{prefix}_{kind}_{uuid.uuid4().hex[:8]}")
        # path that does not need to exist yet
        resource = str(root / "repo.dry")

    elif kind == "buffer":
        # disk-backed file object
        resource = tempfile.TemporaryFile(mode="w+b")

    elif kind == "zip_buffer":
        # memory-backed file object
        resource = io.BytesIO()

    else:
        raise ValueError(f"Unknown StoreDef: {kind}")

    return StoreResource(kind=kind, root=root, resource=resource)


@pytest.fixture
def store_resource_factory(tmp_path_factory):
    created: list[StoreResource] = []

    def _make(kind: StoreDef, *, prefix: str = "store") -> StoreResource:
        res = build_store_resource(tmp_path_factory, kind, prefix=prefix)
        created.append(res)
        return res

    try:
        yield _make
    finally:
        for res in created:
            res.close()


@pytest.fixture(params=["directory", "zip_filepath", "buffer", "zip_buffer"], ids=lambda x: x)
def store_resource(request, store_resource_factory) -> StoreResource:
    return store_resource_factory(request.param)


# -------------------------
# Layer 2: store grouping
# -------------------------

@dataclass
class StoreHandle:
    res: StoreResource
    store: Store

    def rewind(self) -> None:
        self.res.rewind()

    def close(self) -> None:
        self.res.close()


@dataclass
class StoreSet:
    handles: list[StoreHandle]

    @property
    def stores(self) -> list[Store]:
        return [h.store for h in self.handles]

    def rewind_all(self) -> None:
        for h in self.handles:
            h.rewind()

    def close_all(self) -> None:
        for h in self.handles:
            h.close()

    def fresh_stores(self) -> list[Store]:
        """
        Rebuild Store objects from the same underlying resources.
        Useful for simulating "new Repo" reading the same backing data.
        """
        self.rewind_all()
        return [make_store(h.res.resource) for h in self.handles]


def _store_set_id(defs: Sequence[StoreDef]) -> str:
    return "+".join(defs)


def fixture_def(store_sets: list[list[StoreDef]], *, name: str):
    @pytest.fixture(name=name, params=store_sets, ids=_store_set_id)
    def _fx(request, store_resource_factory) -> StoreSet:
        defs: list[StoreDef] = request.param
        handles: list[StoreHandle] = []

        try:
            for i, kind in enumerate(defs):
                res = store_resource_factory(kind, prefix=f"{name}_{i}")
                store = make_store(res.resource)
                handles.append(StoreHandle(res=res, store=store))
            yield StoreSet(handles=handles)
        finally:
            # closes buffers/tempfiles (dirs are tmp_path-managed)
            for h in handles:
                h.close()

    return _fx


# Legacy wrappers (optional while migrating)
@pytest.fixture
def create_temp_dir(store_resource_factory):
    return str(store_resource_factory("directory", prefix="legacy_dir").resource)

@pytest.fixture
def create_temp_named_file(store_resource_factory):
    return str(store_resource_factory("zip_filepath", prefix="legacy_zip_path").resource)

@pytest.fixture
def create_temp_file(store_resource_factory):
    return store_resource_factory("buffer", prefix="legacy_tmpfile").resource

@pytest.fixture
def create_name(store_resource_factory):
    res = store_resource_factory("zip_filepath", prefix="legacy_name")
    p = Path(res.resource)
    return str(p.with_suffix(""))

primary_store_set = fixture_def(
    [
        ["directory"],
        ["zip_filepath"],
        ["zip_buffer"],
        ["buffer"],
    ],
    name="primary_store_set"
)


@pytest.fixture(scope="session")
def ray():
    ray = pytest.importorskip("ray")
    ray.init(num_cpus=1, num_gpus=0, ignore_reinit_error=True)
    try:
        yield ray
    finally:
        ray.shutdown()
