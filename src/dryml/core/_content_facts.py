"""Internal bounds and caller-scoped memoization for immutable content facts.

This dependency-light helper owns no parser, domain policy or external resources.
Sizes overcount repeated immutable leaves and prospective tuple/container slots;
callers add source/context and facility bookkeeping before reserving storage.
"""

import asyncio
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import os
import sys
from threading import get_ident
import weakref
from types import MappingProxyType


_content_facts = ContextVar("dryml_immutable_content_facts", default=None)


def _caller():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), get_ident(), task


class _ContentFacts:
    """Bound pure interpretations by complete input bytes within one caller scope.

    At most 64 entries and 4 MiB of encoded source/output bytes are retained;
    each interpreted value remains bounded by its owning decoder's grammar.
    Kind keys distinguish reader implementations and interpretation parameters.
    No paths, authority cuts, pending flags, owners or availability are cached.
    """

    def __init__(self):
        self.caller = _caller()
        self.entries = OrderedDict()
        self.bytes = 0
        self.active = True

    def decode(self, kind, payload, build):
        key = kind, payload
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key][0]
        result = build()
        charge = len(payload) + (len(result) if type(result) is bytes else 0)
        if charge <= 4 * 1024 * 1024:
            while self.entries and (len(self.entries) >= 64 or self.bytes + charge > 4 * 1024 * 1024):
                _, (_, old_charge) = self.entries.popitem(last=False)
                self.bytes -= old_charge
            self.entries[key] = result, charge
            self.bytes += charge
        return result


def _current_content_facts():
    facts = _content_facts.get()
    return facts if facts is not None and facts.active and facts.caller == _caller() else None


@contextmanager
def _content_facts_scope():
    """Share immutable interpretations only; clear them on every outer scope exit."""
    if _current_content_facts() is not None:
        yield
        return
    facts = _ContentFacts()
    token = _content_facts.set(facts)
    try:
        yield
    finally:
        facts.active = False
        facts.entries.clear()
        facts.bytes = 0
        facts.caller = None
        _content_facts.reset(token)


def _primitive_size(value):
    """Bound prospective owned primitive containers, without allocating copies."""
    if type(value) in (dict, MappingProxyType):
        return 128 + 16 * len(value) + sum(
            96 + sys.getsizeof(key) + _primitive_size(item)
            for key, item in value.items()
        )
    if type(value) in (list, tuple):
        return 128 + 16 * len(value) + sum(_primitive_size(item) for item in value)
    if type(value) in (str, bytes, int, float, bool, type(None), weakref.ReferenceType):
        return sys.getsizeof(value)
    raise TypeError("Content fact context/value is not an exact primitive.")


def _context_matches(left, right):
    """Compare closed keys without invoking user equality or reusing dead IDs."""
    if type(left) is not type(right):
        return False
    if type(left) is tuple:
        return len(left) == len(right) and all(_context_matches(a, b) for a, b in zip(left, right))
    if type(left) is weakref.ReferenceType:
        target = left()
        return target is not None and target is right()
    return left == right


def _freeze_json_fact(value):
    if type(value) in (dict, MappingProxyType):
        return (0, tuple((key, _freeze_json_fact(item)) for key, item in value.items()))
    if type(value) in (list, tuple):
        return (1, tuple(_freeze_json_fact(item) for item in value))
    return value


def _thaw_json_fact(value):
    """Rebuild the existing canonical parser's detached mapping/tuple projection."""
    if type(value) is tuple:
        kind, items = value
        if kind == 0:
            return MappingProxyType({key: _thaw_json_fact(item) for key, item in items})
        return tuple(_thaw_json_fact(item) for item in items)
    return value


def _node_block_owner_size(content, blocks):
    """Conservatively charge an operation-local immutable decoded block owner.

    Encoded syntax containers and leaves bound the decoded Node/value graph;
    four prospective copies cover frozen carriers, table/edge bookkeeping and
    named-view preparation. Source bytes and owner/key bookkeeping are charged
    independently. Unsupported primitive syntax is deliberately not cacheable.
    """
    if type(content) is not bytes:
        raise TypeError("Node block ownership requires exact immutable bytes.")
    return 2048 + sys.getsizeof(content) + 4 * _primitive_size(blocks)
