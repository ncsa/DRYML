from typing import Any

import numpy as np
from .freeze import FrozenList, FrozenTuple, FrozenDict, FrozenSet

# Officially supported DRYML types (and related methods)

_PY_POD = (type(None), bool, int, float, str, bytes)
_NP_SCALAR = (np.generic,)

def is_pod(x: Any) -> bool:
    return isinstance(x, _PY_POD) or isinstance(x, _NP_SCALAR) or isinstance(x, type)

compatible_containers = {
    'tuple': (tuple, FrozenList), # Tuples
    'list': (list, FrozenList),
    'dict': (dict, FrozenDict),
    'set': (set, FrozenSet)
}

container_types = (
    tuple, FrozenTuple,
    list, FrozenList,
    dict, FrozenDict,
    set, FrozenSet
)
