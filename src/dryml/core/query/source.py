"""Private Query V3 producer inventory and coherent Store-cut capture.

This layer turns Repo, Store, and fixed members into detached identity evidence
without publishing a replacement query API. It snapshots Repo topology once per
capture and groups only identical authority-fence domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..object import Object
from ..reference_values import ObjectRef, StateRef
from ..store.store import Store, StoreCapabilityError
from .authority import CapturedStoreFacts, IdentityValue, _derive_identities
from .identity import IdentitySet, SourceEvidence


class InventoryCapabilityError(StoreCapabilityError):
    """Raised before a broad Query V3 capture lacks complete record coverage."""


@dataclass(frozen=True, slots=True)
class StoreSource:
    """Private Store producer retaining only its declared authority scope."""

    store: Store

    def __post_init__(self) -> None:
        if not isinstance(self.store, Store):
            raise TypeError("StoreSource requires a Store.")


@dataclass(frozen=True, slots=True)
class RepoSource:
    """Private Repo producer that adds captured live-cache identity knowledge."""

    repo: object
    weak: bool = True

    def __post_init__(self) -> None:
        if type(self.weak) is not bool:
            raise TypeError("RepoSource weak must be an exact bool.")


class SourceCapture:
    """One terminal-local registry of detached Store facts and cache inventory."""

    def __init__(self):
        self._facts: dict[str, CapturedStoreFacts] = {}

    def capture_store(self, source: StoreSource | Store) -> CapturedStoreFacts:
        """Capture one complete Store inventory exactly once per authority fence."""

        store = source.store if isinstance(source, StoreSource) else source
        if not isinstance(store, Store):
            raise TypeError("Store capture requires a Store or StoreSource.")
        key = store.authority_fence_key()
        cached = self._facts.get(key)
        if cached is not None:
            return cached
        self._require_complete_inventory(store)
        with store.authority_read_fence():
            main = store.read_main_ref()
            definitions = tuple(store.iter_definition_records())
            definition_by_digest = {record.digest: record.definition for record in definitions}
            if main is not None:
                main_definition = definition_by_digest.get(main.definition_digest)
                if main_definition is None:
                    raise InventoryCapabilityError("Main-reference authority targets a missing definition record.")
            else:
                main_definition = None
            facts = CapturedStoreFacts.build(
                definitions=definitions,
                stored_roots=store.iter_stored_root_records(),
                declarations=store.iter_declaration_records(),
                state_refs=store.iter_state_ref_records(),
                object_aliases=store.iter_object_alias_records(),
                state_aliases=store.iter_state_alias_records(),
                main_definition=main_definition,
                source=SourceEvidence.from_source(store.authority_fence_key()),
            )
        self._facts[key] = facts
        return facts

    def capture_stores(self, sources: Iterable[StoreSource | Store]) -> tuple[CapturedStoreFacts, ...]:
        """Capture each selected Store while preserving distinct transaction cuts."""

        return tuple(self.capture_store(source) for source in sources)

    def read_exact_state(self, source: StoreSource | Store, target: StateRef) -> StateRef | None:
        """Read one complete exact StateRef without requiring broad inventory.

        The digest-addressed record is independently authoritative for exact
        saved-state membership. This selective path intentionally does not use
        the capture registry or any record-family iterator.
        """

        store = source.store if isinstance(source, StoreSource) else source
        if not isinstance(store, Store):
            raise TypeError("Exact state lookup requires a Store or StoreSource.")
        if not isinstance(target, StateRef):
            raise TypeError("Exact state lookup requires a StateRef.")
        with store.authority_read_fence():
            record = store.read_state_ref_record(target.digest())
        if record is None:
            return None
        if record.state_ref != target:
            raise InventoryCapabilityError("Exact StateRef authority is incompatible with its digest.")
        return record.state_ref

    def capture_repo(self, source: RepoSource) -> IdentitySet:
        """Capture Repo Store topology and retained cache identities once."""

        if not isinstance(source, RepoSource):
            raise TypeError("Repo capture requires a RepoSource.")
        repo = source.repo
        retain_topology = getattr(repo, "retain_topology", None)
        if not callable(retain_topology):
            raise TypeError("RepoSource requires a Repo-like topology producer.")
        with retain_topology():
            stores = tuple(repo.stores)
            facts = self.capture_stores(stores)
            cache_values = _cache_values(repo, weak=source.weak)
        members = [
            (value, facts_item.source)
            for facts_item in facts
            for value in facts_item.knowledge()
        ]
        members.extend(
            (value, SourceEvidence.from_source(repo))
            for value in _derive_identities(cache_values)
        )
        return IdentitySet(members)

    @staticmethod
    def _require_complete_inventory(store: Store) -> None:
        capabilities = store._query_v3_inventory_capabilities()
        missing = [
            name for name in capabilities.__dataclass_fields__
            if getattr(capabilities, name) == "unsupported"
        ]
        if missing:
            raise InventoryCapabilityError(
                "Store cannot provide complete Query V3 inventory for required authority families."
            )


def _cache_values(repo, *, weak: bool) -> tuple[IdentityValue, ...]:
    """Read retained cache facts without constructing, saving, or capturing state."""

    caches = (repo.strong_obj_cache, repo.weak_obj_cache) if weak else (repo.strong_obj_cache,)
    values: list[IdentityValue] = []
    for cache in caches:
        for _, value in tuple(cache.items()):
            if not isinstance(value, Object):
                continue
            values.append(value.definition)
            reference = value.object_ref
            if isinstance(reference, ObjectRef):
                values.append(reference)
            receipt = value.last_state_ref
            if isinstance(receipt, StateRef) and receipt.object == reference:
                values.append(receipt)
    return tuple(values)
