from pathlib import Path

from dryml.core import Object
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord


class DirectLayoutObject(Object):
    pass


def test_definition_records_use_direct_digest_paths_without_object_generations(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(DirectLayoutObject().definition)

    store.write_definition_record(record)

    path = Path(store.base_dir) / "definitions" / record.digest[:2] / f"{record.digest}.record"
    assert path.is_file()
    assert store.read_definition_record(record.digest) == record
    assert not list(Path(store.base_dir).rglob(".state-generations"))
    assert not list(Path(store.base_dir).rglob(".state-current.pkl"))
