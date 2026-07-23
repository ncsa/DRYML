"""Importable lightweight targets for bounded dispatch-provenance tests."""

import dryml

from dryml.core.object import Pickleable


@dryml.world.req(cpus={"min": 1})
def provenance_add(left, right):
    return left + right


class ProvenanceBox(Pickleable):
    def __init__(self, value):
        super().__init__()
        self.value = value

    @dryml.world.req(cpus={"min": 1})
    def plus(self, amount):
        return self.value + amount
