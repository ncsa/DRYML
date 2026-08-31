#!/usr/bin/env python
"""Build and query a synthetic DirStore SQLite query index.

The benchmark creates lightweight `ConcreteDefinition` roots without object
state, rebuilds the Store-owned SQLite sidecar, and prints JSON metrics. It is
intended for local scale experiments rather than normal test execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.query.sqlite import SQLiteQueryIndexConfig
from dryml.core.store.dir import DirStore
from dryml.core.symbol import ImportRef
from dryml.core.store.records import DefinitionRecord


def synthetic_cdef(index: int) -> ConcreteDefinition:
    return ConcreteDefinition(ImportRef("builtins", "dict"), FrozenTuple((f"item-{index}",)), FrozenDict({}))


def write_root(store: DirStore, cdef: ConcreteDefinition) -> None:
    """Publish one synthetic definition through current immutable authority."""

    store.write_definition_record(DefinitionRecord(cdef))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path, help="Directory Store path to create or reuse.")
    parser.add_argument("--roots", type=int, default=10_000, help="Number of synthetic roots to create.")
    parser.add_argument("--reset", action="store_true", help="Delete the Store directory before creating roots.")
    parser.add_argument("--journal-mode", choices=("auto", "delete", "wal"), default="delete")
    args = parser.parse_args()

    if args.reset and args.store.exists():
        shutil.rmtree(args.store)

    store = DirStore(args.store, query_index=SQLiteQueryIndexConfig(journal_mode=args.journal_mode))

    start = time.perf_counter()
    for idx in range(args.roots):
        write_root(store, synthetic_cdef(idx))
    create_seconds = time.perf_counter() - start

    start = time.perf_counter()
    report = store.rebuild_query_index()
    rebuild_seconds = time.perf_counter() - start

    target = synthetic_cdef(args.roots // 2)
    index = store.open_query_index()
    start = time.perf_counter()
    with index.read_view() as view:
        exact_count = len(view.filter_stored_ids(view.exact_ids(target)))
    exact_seconds = time.perf_counter() - start

    status = store.query_index_status()
    db_path = Path(store.query_index_path)
    print(json.dumps({
        "roots": args.roots,
        "create_seconds": create_seconds,
        "rebuild_seconds": rebuild_seconds,
        "exact_seconds": exact_seconds,
        "exact_count": exact_count,
        "generation": status.generation,
        "row_counts": status.row_counts,
        "database_bytes": db_path.stat().st_size if db_path.exists() else 0,
        "reconcile_action": report.action,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
