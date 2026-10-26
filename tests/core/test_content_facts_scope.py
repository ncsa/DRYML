"""Pure memoization stays bounded, caller-local and non-authoritative."""

import asyncio
from contextvars import copy_context
from threading import Thread

import pytest

from dryml.core._content_facts import _content_facts_scope, _current_content_facts


def test_content_scope_rejects_copied_callers_and_releases_on_cancellation():
    inherited = copy_context()
    with pytest.raises(KeyboardInterrupt):
        with _content_facts_scope():
            facts = _current_content_facts()
            assert facts.decode("value", b"source", lambda: 1) == 1
            assert facts.decode("value", b"source", lambda: pytest.fail("duplicate interpretation")) == 1
            assert facts.decode("value", b"changed", lambda: 2) == 2
            inherited = copy_context()
            seen = []
            worker = Thread(target=lambda: inherited.run(lambda: seen.append(_current_content_facts())))
            worker.start()
            worker.join(5)
            assert not worker.is_alive() and seen == [None]

            async def child():
                assert _current_content_facts() is None
                with _content_facts_scope():
                    assert _current_content_facts() is not facts

            asyncio.run(child())
            raise KeyboardInterrupt()
    assert not facts.active and not facts.entries and facts.bytes == 0
    assert _current_content_facts() is None
    assert inherited.run(_current_content_facts) is None


def test_content_scope_evicts_without_changing_results_or_caching_errors():
    with _content_facts_scope():
        facts = _current_content_facts()
        for index in range(100):
            assert facts.decode("value", str(index).encode(), lambda: index) == index
        assert len(facts.entries) <= 64
        assert facts.decode("value", b"0", lambda: 0) == 0
        huge = b"x" * (4 * 1024 * 1024 + 1)
        assert facts.decode("large", huge, lambda: 1) == 1
        assert facts.decode("large", huge, lambda: 2) == 2
        with pytest.raises(ValueError):
            facts.decode("invalid", b"source", lambda: (_ for _ in ()).throw(ValueError()))
        assert facts.decode("invalid", b"source", lambda: "recovered") == "recovered"
        assert facts.bytes <= 4 * 1024 * 1024
