import pytest

from dryml.core import Object
from dryml.core.query.lowering import CollectSink, CountSink, ExistsSink, OneOrNoneSink, OneSink, PageSink, PagedResultCursor, ScanPolicy
from dryml.core.query.model import QueryCardinalityError


class LoweringLeaf(Object):
    def __init__(self, name="leaf"):
        super().__init__()
        self.name = name


def test_exists_sink_stops_after_first_match():
    sink = ExistsSink()
    cdef = LoweringLeaf("one").definition

    assert sink.accept(cdef) is False
    assert sink.done
    assert sink.result() is True
    assert sink.stop_reason == "first-match"


def test_one_sink_stops_after_second_match_and_reports_cardinality():
    sink = OneSink()
    first = LoweringLeaf("first").definition
    second = LoweringLeaf("second").definition

    assert sink.accept(first) is True
    assert sink.accept(second) is False
    assert sink.done
    assert sink.stop_reason == "second-match"
    with pytest.raises(QueryCardinalityError):
        sink.result()


def test_one_sink_zero_and_one_or_none_zero_semantics():
    with pytest.raises(QueryCardinalityError):
        OneSink().result()

    assert OneOrNoneSink().result() is None


def test_count_sink_retains_only_integer_and_collect_sink_retains_cdefs():
    first = LoweringLeaf("first").definition
    second = LoweringLeaf("second").definition
    count = CountSink(stop_after=2)
    collect = CollectSink(stop_after=2)

    assert count.accept(first) is True
    assert count.accept(second) is False
    assert count.result() == 2
    assert not hasattr(count, "_items")

    assert collect.accept(first) is True
    assert collect.accept(second) is False
    assert collect.result() == (first, second)


def test_page_sink_returns_cursor():
    cdef = LoweringLeaf("page").definition
    cursor = PagedResultCursor("store", 1, "00", 0, 1)
    sink = PageSink(1)

    assert sink.accept(cdef, cursor) is False
    assert sink.result() == ((cdef,), cursor)
    assert sink.stop_reason == "page-full"


def test_scan_policy_validates_mode_and_budget():
    assert ScanPolicy("allow").mode == "allow"
    assert ScanPolicy("warn", 10).max_verify == 10
    with pytest.raises(ValueError):
        ScanPolicy("bad")
    with pytest.raises(ValueError):
        ScanPolicy("allow", -1)
