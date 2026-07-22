"""Portable logical references to declared managed-method outputs."""

from __future__ import annotations

from dryml.core2.arg_roles import RefCDef
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.object import Object


class ManagedOutputRef(Object):
    """Identify one logical output by producer definition, method, and slot.

    The producer is stored as a non-materializing ``RefCDef`` edge. Store,
    realization, record, and representation identifiers intentionally do not
    participate in this Object's definition identity.

    Args:
        producer: Producing Object or exact definition.
        method: Managed method name on the producer.
        slot: Declared output slot.
    """

    def __init__(self, *, producer: RefCDef, method: str, slot: str):
        super().__init__()
        if not isinstance(producer, ConcreteDefinition):
            raise TypeError("ManagedOutputRef producer must resolve to a ConcreteDefinition.")
        if not isinstance(method, str) or not method:
            raise ValueError("ManagedOutputRef method must be a non-empty string.")
        if not isinstance(slot, str) or not slot:
            raise ValueError("ManagedOutputRef slot must be a non-empty string.")
        self.producer = producer
        self.method = method
        self.slot = slot


__all__ = ["ManagedOutputRef"]
