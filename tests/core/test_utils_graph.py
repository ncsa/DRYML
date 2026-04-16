from __future__ import annotations

from collections import namedtuple

import pytest

from dryml.core2.utils.graph import (
    GraphCtx,
    GraphTransformError,
    GraphTransformer,
    GraphVisitor,
)


Point = namedtuple("Point", ["x", "y"])


class DoubleInts(GraphTransformer):
    def is_atomic(self, obj, ctx):
        return isinstance(obj, int)

    def transform_atomic(self, obj, ctx):
        return obj * 2


class DoubleIntsIncludingDictKeys(DoubleInts):
    def transform_dict_keys(self, obj, ctx):
        return True


class CollectInts(GraphVisitor):
    def __init__(self):
        self.items = []

    def is_atomic(self, obj, ctx):
        return isinstance(obj, int)

    def visit_atomic(self, obj, ctx):
        self.items.append((ctx.path, obj))


class CollectIntsIncludingDictKeys(CollectInts):
    def visit_dict_keys(self, obj, ctx):
        return True


class Box:
    def __init__(self, value):
        self.value = value


class BoxTransformer(GraphTransformer):
    def __init__(self):
        self.calls = 0

    def memo_key(self, obj, ctx):
        if isinstance(obj, Box):
            return id(obj)
        return None

    def transform_other(self, obj, ctx):
        if isinstance(obj, Box):
            self.calls += 1
            return Box(obj.value * 10)
        return super().transform_other(obj, ctx)


class BoxVisitor(GraphVisitor):
    def __init__(self):
        self.calls = 0

    def memo_key(self, obj, ctx):
        if isinstance(obj, Box):
            return id(obj)
        return None

    def visit_other(self, obj, ctx):
        if isinstance(obj, Box):
            self.calls += 1
            return
        return super().visit_other(obj, ctx)


class Node:
    def __init__(self, value, child=None):
        self.value = value
        self.child = child


class NodeTransformer(GraphTransformer):
    def is_atomic(self, obj, ctx):
        return isinstance(obj, int)

    def transform_atomic(self, obj, ctx):
        return obj + 1

    def memo_key(self, obj, ctx):
        if isinstance(obj, Node):
            return id(obj)
        return None

    def should_track_cycle(self, obj, ctx):
        return isinstance(obj, Node) or super().should_track_cycle(obj, ctx)

    def transform_other(self, obj, ctx):
        if isinstance(obj, Node):
            return Node(
                self.transform(obj.value, ctx.child("value")),
                self.transform(obj.child, ctx.child("child")) if obj.child is not None else None,
            )
        return super().transform_other(obj, ctx)


class NodeVisitor(GraphVisitor):
    def __init__(self):
        self.values = []

    def is_atomic(self, obj, ctx):
        return isinstance(obj, int)

    def visit_atomic(self, obj, ctx):
        self.values.append((ctx.path, obj))

    def memo_key(self, obj, ctx):
        if isinstance(obj, Node):
            return id(obj)
        return None

    def should_track_cycle(self, obj, ctx):
        return isinstance(obj, Node) or super().should_track_cycle(obj, ctx)

    def visit_other(self, obj, ctx):
        if isinstance(obj, Node):
            self.visit(obj.value, ctx.child("value"))
            if obj.child is not None:
                self.visit(obj.child, ctx.child("child"))
            return
        return super().visit_other(obj, ctx)


def test_graph_ctx_child_and_with_state():
    ctx = GraphCtx()
    ctx2 = ctx.child("a").child(3)
    ctx3 = ctx2.with_state(mode="x", enabled=True)

    assert ctx.path == ()
    assert ctx2.path == ("a", 3)
    assert ctx2.path_str() == "a/3"

    assert ctx.state == {}
    assert ctx2.state == {}
    assert ctx3.state == {"mode": "x", "enabled": True}

    assert ctx.memo is ctx2.memo
    assert ctx2.memo is ctx3.memo
    assert ctx.active_ids is ctx2.active_ids
    assert ctx2.active_ids is ctx3.active_ids


def test_transformer_maps_nested_plain_containers_and_namedtuple():
    x = {
        "a": [1, 2, (3, 4)],
        "b": Point(5, 6),
    }

    y = DoubleInts().transform(x)

    assert y == {
        "a": [2, 4, (6, 8)],
        "b": Point(10, 12),
    }
    assert isinstance(y["b"], Point)


def test_transformer_does_not_transform_dict_keys_by_default():
    x = {1: 2, 3: 4}
    y = DoubleInts().transform(x)

    assert y == {1: 4, 3: 8}


def test_transformer_can_opt_in_to_transform_dict_keys():
    x = {1: 2, 3: 4}
    y = DoubleIntsIncludingDictKeys().transform(x)

    assert y == {2: 4, 6: 8}


def test_transformer_raises_on_unsupported_leaf_by_default():
    class Unknown:
        pass

    with pytest.raises(TypeError):
        GraphTransformer().transform(Unknown())


def test_transformer_cycle_detection_on_recursive_list():
    x = []
    x.append(x)

    with pytest.raises(GraphTransformError, match="Cycle detected"):
        DoubleInts().transform(x)


def test_transformer_memoization_reuses_transformed_object():
    shared = Box(7)
    x = [shared, shared]

    tr = BoxTransformer()
    y = tr.transform(x)

    assert tr.calls == 1
    assert isinstance(y[0], Box)
    assert y[0].value == 70
    assert y[0] is y[1]


def test_transformer_supports_custom_object_graphs():
    x = Node(1, Node(2, None))
    y = NodeTransformer().transform(x)

    assert isinstance(y, Node)
    assert y.value == 2
    assert y.child.value == 3
    assert y.child.child is None


def test_transformer_detects_cycles_in_custom_object_graphs():
    x = Node(1)
    x.child = x

    with pytest.raises(GraphTransformError, match="Cycle detected"):
        NodeTransformer().transform(x)


def test_visitor_collects_atomic_values_with_paths():
    x = {
        "a": [1, 2],
        "b": Point(3, 4),
    }

    vis = CollectInts()
    vis.visit(x)

    assert vis.items == [
        (("a", 0), 1),
        (("a", 1), 2),
        (("b", 0), 3),
        (("b", 1), 4),
    ]


def test_visitor_does_not_visit_dict_keys_by_default():
    x = {1: 2, 3: 4}

    vis = CollectInts()
    vis.visit(x)

    assert vis.items == [
        ((1,), 2),
        ((3,), 4),
    ]


def test_visitor_can_opt_in_to_visit_dict_keys():
    x = {1: 2, 3: 4}

    vis = CollectIntsIncludingDictKeys()
    vis.visit(x)

    assert vis.items == [
        (("<key>",), 1),
        ((1,), 2),
        (("<key>",), 3),
        ((3,), 4),
    ]


def test_visitor_cycle_detection_on_recursive_list():
    x = []
    x.append(x)

    with pytest.raises(GraphTransformError, match="Cycle detected"):
        CollectInts().visit(x)


def test_visitor_memoization_visits_shared_object_once():
    shared = Box(5)
    x = [shared, shared]

    vis = BoxVisitor()
    vis.visit(x)

    assert vis.calls == 1


def test_visitor_supports_custom_object_graphs():
    x = Node(1, Node(2, None))

    vis = NodeVisitor()
    vis.visit(x)

    assert vis.values == [
        (("value",), 1),
        (("child", "value"), 2),
    ]


def test_visitor_detects_cycles_in_custom_object_graphs():
    x = Node(1)
    x.child = x

    with pytest.raises(GraphTransformError, match="Cycle detected"):
        NodeVisitor().visit(x)
