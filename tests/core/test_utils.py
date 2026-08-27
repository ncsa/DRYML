from dryml.core.utils.general import get_class_str, get_class_from_str, \
    get_unique_objects, get_unique_concrete_definitions
from tests.core import core_objects as objects
import pytest
from dryml.core.errors import CycleError
from dryml.core.utils.recurse import cycle_detect


def test_class_utils_1():
    cls = objects.HelloInt

    class_str = get_class_str(cls)

    cls_2 = get_class_from_str(class_str)

    assert cls is cls_2


def test_class_utils_2():
    obj = objects.HelloInt(msg=5)

    class_str = get_class_str(obj)

    cls_2 = get_class_from_str(class_str)

    assert type(obj) is cls_2


def test_list_unique_objs_1():
    obj_f1_1 = objects.TestClassF1() 
    obj_f1_2 = objects.TestClassF1() 
    obj_f1_3 = objects.TestClassF1() 
    obj_c_1 = objects.TestClassC(
        obj_f1_2,
        B=obj_f1_3) 
    obj_c_2 = objects.TestClassC(
        obj_f1_1,
        B=obj_c_1)

    unique_obj_definitions = set(get_unique_concrete_definitions(obj_c_2))

    assert len(unique_obj_definitions) == 5
    assert obj_f1_1.definition in unique_obj_definitions
    assert obj_f1_2.definition in unique_obj_definitions
    assert obj_f1_3.definition in unique_obj_definitions
    assert obj_c_1.definition in unique_obj_definitions
    assert obj_c_2.definition in unique_obj_definitions


class Node:
    def __init__(self, child=None):
        self.child = child


class Box:
    def __init__(self, child=None):
        self.child = child


def is_node(val):
    return isinstance(val, Node)


def is_box(val):
    return isinstance(val, Box)


@cycle_detect(arg_pos=0)
def walk_structure(val):
    """
    Generic recursive structure walker using the decorator's default tracking.
    This is useful for testing repeated atomic leaves and container cycles.
    """
    if isinstance(val, dict):
        total = 1
        for k, v in val.items():
            total += walk_structure(k)
            total += walk_structure(v)
        return total

    if isinstance(val, (list, tuple, set, frozenset)):
        total = 1
        for item in val:
            total += walk_structure(item)
        return total

    return 1


def test_cycle_detect_rejects_both_arg_pos_and_kwarg_name():
    with pytest.raises(ValueError, match="Specify only one"):
        cycle_detect(arg_pos=0, kwarg_name="x")


def test_cycle_detect_requires_one_of_arg_pos_or_kwarg_name():
    with pytest.raises(ValueError, match="Specify one"):
        cycle_detect(arg_pos=None, kwarg_name=None)


def test_cycle_detect_detects_simple_container_cycle():
    a = []
    a.append(a)

    with pytest.raises(CycleError, match="Cycle detected"):
        walk_structure(a)


def test_cycle_detect_allows_repeated_atomic_leaves():
    """
    Regression test for the original false positive on repeated small ints like 0.
    """
    structure = {
        "a": 0,
        "b": [0, 0, {"c": 0}],
        "d": (0, 0),
    }

    result = walk_structure(structure)
    assert isinstance(result, int)
    assert result > 0


def test_cycle_detect_allows_shared_subobject_when_not_on_active_path():
    """
    A DAG is not a cycle. The same container may appear in multiple sibling
    branches as long as it is not re-entered while already on the active stack.
    """
    shared = [1, 2]
    root = [shared, shared]

    result = walk_structure(root)
    assert isinstance(result, int)
    assert result > 0


def test_cycle_detect_cleans_up_path_after_exception():
    """
    If the wrapped function raises, the active-path state must still be cleaned
    up so a later independent call does not falsely detect a cycle.
    """
    @cycle_detect(arg_pos=0, should_track=is_node)
    def explode(node):
        if node.child is None:
            raise RuntimeError("boom")
        return explode(node.child)

    n = Node()

    with pytest.raises(RuntimeError, match="boom"):
        explode(n)

    # A second call on the same object should behave the same way, not trip a
    # stale cycle path from the previous failed call.
    with pytest.raises(RuntimeError, match="boom"):
        explode(n)


def test_cycle_detect_supports_kwarg_name_mode():
    @cycle_detect(arg_pos=None, kwarg_name="node", should_track=is_node)
    def walk_node(*, node):
        if node.child is None:
            return 1
        return 1 + walk_node(node=node.child)

    n = Node()
    n.child = n

    with pytest.raises(CycleError, match="Cycle detected"):
        walk_node(node=n)


def test_cycle_detect_state_is_isolated_per_decorated_function():
    """
    Regression test for the shared-ContextVar bug.

    If cycle state is global across all decorated functions, inner(box) will
    falsely think box is already on its own active path when called from
    outer(box).
    """
    @cycle_detect(arg_pos=0, should_track=is_box)
    def inner(box):
        if box.child is None:
            return 1
        return 1 + inner(box.child)

    @cycle_detect(arg_pos=0, should_track=is_box)
    def outer(box):
        return inner(box)

    b = Box()
    assert outer(b) == 1
