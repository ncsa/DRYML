"""Repo-level federation facade for store-owned record sidecars."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .errors import RecordExportError, RecordNotFoundError
from .export import copy_record_closure
from .policy import RecordPolicy, RecordPolicyOptions, RecordPolicyReport
from .refs import LocatedRecordRef, LocatedSpecRef, RecordRef, SpecRef


class RepoRecordFederation:
    """Lightweight facade that delegates record/spec queries to repo stores."""

    def __init__(self, repo: Any):
        self.repo = repo

    def stores(self) -> tuple[Any, ...]:
        """Return the repo stores in deterministic repo order."""

        return tuple(self.repo.stores)

    def find_record(self, record_id: str) -> tuple[LocatedRecordRef, ...]:
        """Return located copies of *record_id* across repo stores."""

        refs = []
        for store in self.stores():
            record_io = store.records
            if record_io.has_record(record_id):
                refs.append(LocatedRecordRef(record_io._store_ref(), record_id))
        return tuple(sorted(refs, key=lambda ref: (ref.store_ref, ref.record_id)))

    def find_spec(self, spec_id: str, *, family: str | None = None) -> tuple[LocatedSpecRef, ...]:
        """Return located copies of *spec_id* across repo stores."""

        refs = []
        for store in self.stores():
            record_io = store.records
            if record_io.has_spec(spec_id, family=family):
                spec = record_io.read_spec(spec_id, family=family)
                refs.append(LocatedSpecRef(record_io._store_ref(), spec_id, family or _spec_family(spec)))
        return tuple(sorted(refs, key=lambda ref: (ref.store_ref, ref.kind or "", ref.spec_id)))

    def find_records(self, **filters) -> tuple[LocatedRecordRef, ...]:
        """Scan stores in repo order and return records matching payload filters."""

        refs = []
        for store in self.stores():
            refs.extend(store.records.find_records(**filters))
        return tuple(refs)

    def read_record(self, ref: LocatedRecordRef) -> dict[str, Any]:
        """Read a located record from its owning repo store."""

        return self._store_for_ref(ref.store_ref).records.read_record(ref.record_id)

    def read_spec(self, ref: LocatedSpecRef) -> dict[str, Any]:
        """Read a located spec from its owning repo store."""

        return self._store_for_ref(ref.store_ref).records.read_spec(ref.spec_id, family=ref.kind)

    def find_records_mentioning_cdef(self, cdef_id: str, **kwargs) -> tuple[LocatedRecordRef, ...]:
        """Return records across stores whose payloads mention *cdef_id*."""

        refs = []
        for store in self.stores():
            refs.extend(store.records.find_records_mentioning_cdef(cdef_id, **kwargs))
        return _dedupe_located_records(refs)

    def find_specs_mentioning_cdef(self, cdef_id: str, **kwargs) -> tuple[LocatedSpecRef, ...]:
        """Return specs across stores whose payloads mention *cdef_id*."""

        refs = []
        for store in self.stores():
            refs.extend(store.records.find_specs_mentioning_cdef(cdef_id, **kwargs))
        return _dedupe_located_specs(refs)

    def find_operation_specs_for_cdef(self, cdef_id: str, **kwargs) -> tuple[LocatedSpecRef, ...]:
        """Return operation specs across stores whose payloads mention *cdef_id*."""

        refs = []
        for store in self.stores():
            refs.extend(store.records.find_operation_specs_for_cdef(cdef_id, **kwargs))
        return _dedupe_located_specs(refs)

    def copy_closure(
        self,
        destination_store: Any,
        *,
        seed_records: Iterable[str | RecordRef | LocatedRecordRef] = (),
        seed_specs: Iterable[tuple[str, str] | str | SpecRef | LocatedSpecRef] = (),
        source_store: Any | None = None,
        policy: RecordPolicy | str = "closure",
        options: RecordPolicyOptions | None = None,
    ) -> RecordPolicyReport:
        """Copy one unambiguous store-local record closure to *destination_store*."""

        source = source_store or self._source_store_for_seeds(seed_records, seed_specs)
        return copy_record_closure(
            source,
            destination_store,
            seed_records=seed_records,
            seed_specs=seed_specs,
            policy=policy,
            options=options,
        )

    def _store_for_ref(self, store_ref: str) -> Any:
        for store in self.stores():
            if store.records._store_ref() == store_ref:
                return store
        raise RecordNotFoundError("store ref is not part of this repo", context={"store_ref": store_ref})

    def _source_store_for_seeds(self, seed_records, seed_specs) -> Any:
        explicit_store_refs = _located_seed_store_refs(seed_records, seed_specs)
        if len(explicit_store_refs) == 1:
            return self._store_for_ref(next(iter(explicit_store_refs)))
        if len(explicit_store_refs) > 1:
            raise RecordExportError("record closure seeds span multiple source stores", context={"stores": sorted(explicit_store_refs)})

        stores = set()
        for record_id in _record_seed_ids(seed_records):
            for ref in self.find_record(record_id):
                stores.add(ref.store_ref)
        for spec_id, family in _spec_seed_ids(seed_specs):
            for ref in self.find_spec(spec_id, family=family):
                stores.add(ref.store_ref)
        if not stores and len(self.repo.stores) == 1:
            return self.repo.stores[0]
        if len(stores) != 1:
            raise RecordExportError("record closure source store is ambiguous", context={"stores": sorted(stores)})
        return self._store_for_ref(next(iter(stores)))


def _record_seed_ids(seeds) -> tuple[str, ...]:
    ids = []
    for seed in seeds:
        ids.append(seed if isinstance(seed, str) else seed.record_id)
    return tuple(ids)


def _spec_seed_ids(seeds) -> tuple[tuple[str, str | None], ...]:
    ids = []
    for seed in seeds:
        if isinstance(seed, str):
            ids.append((seed, None))
        elif isinstance(seed, tuple):
            ids.append((seed[1], seed[0]))
        else:
            ids.append((seed.spec_id, seed.kind))
    return tuple(ids)


def _located_seed_store_refs(seed_records, seed_specs) -> set[str]:
    refs = set()
    for seed in seed_records:
        if isinstance(seed, LocatedRecordRef):
            refs.add(seed.store_ref)
    for seed in seed_specs:
        if isinstance(seed, LocatedSpecRef):
            refs.add(seed.store_ref)
    return refs


def _spec_family(spec: dict[str, Any]) -> str | None:
    from .specs import spec_family_for_id

    return spec_family_for_id(spec["id"])


def _dedupe_located_records(refs) -> tuple[LocatedRecordRef, ...]:
    return tuple(sorted(set(refs), key=lambda ref: (ref.store_ref, ref.record_id)))


def _dedupe_located_specs(refs) -> tuple[LocatedSpecRef, ...]:
    return tuple(sorted(set(refs), key=lambda ref: (ref.store_ref, ref.kind or "", ref.spec_id)))


__all__ = ["RepoRecordFederation"]
