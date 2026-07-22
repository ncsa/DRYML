import os
import tempfile
from pathlib import Path

import dryml
import core2_objects as objects
from dryml.core2.repo import Repo, make_store
from dryml.core2.store.zip import ZipStore


def test_object_save_load_still_works_without_environment_records(store_resource_factory):
    res = store_resource_factory("directory", prefix="environment_guard")
    store = make_store(res.resource)
    obj = objects.HelloStr(msg="environment guard")
    repo = Repo([store])
    repo.add_objects(obj)
    repo.save(record_policy="none")

    assert not (Path(store.base_dir) / "records").exists()
    assert not (Path(store.base_dir) / "environment").exists()

    loaded_repo = dryml.core2.Repo([make_store(res.resource)])
    loaded = loaded_repo.get().one()
    assert loaded.definition == obj.definition
    assert loaded.get_message() == "Hello! environment guard"


def test_make_store_accepts_named_temporary_file_wrapper():
    with tempfile.NamedTemporaryFile(mode="w+b") as buffer:
        store = make_store(buffer)

        assert isinstance(store, ZipStore)
        store.commit()
        buffer.seek(0)
        reopened = make_store(buffer)
        assert isinstance(reopened, ZipStore)
        store.close()
        reopened.close()
