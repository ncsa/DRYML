"""Dataset recipes, independent local traversals, and detached continuation IO.

Importing this surface does not activate optional tensor frameworks. Prepare/open
retain execution selections without source acquisition; capture explicitly owns
snapshot costs. DatasetState directory IO publishes no Object or managed checkpoint.
See ``docs/data.md`` for ownership, recovery qualification, and failure contracts.
"""

from dryml.data.dataset import Dataset, DatasetCursor, DatasetExhaustedError, Map, StreamDataset
from dryml.data.native import BatchPreparation, NativeTrainingCursor, PreparedDataset
from dryml.data.source import ArrayDataset, GeneratorDataset, NpyFileDataset, TFDSAdapter, TorchDatasetAdapter
from dryml.data.combine import Chain, Zip
from dryml.data.accumulate import Accumulate
from dryml.data.methods import ArgMax, Cast, Flatten, Pipe, Project, Scale, Select
from dryml.data.padding import Padding
from dryml.data.reduction_methods import (
    Abs,
    ArrayMean,
    ArrayQuantile,
    Diff,
    Equal,
    MeanFinalize,
    MeanInitial,
    MeanUpdate,
    ReductionMode,
    ReservoirInitial,
    ReservoirQuantile,
    ReservoirUpdate,
    Squared,
)
from dryml.data.structural import Batch, Epoch, Repeat, Shuffle, Skip, Take, Unbatch
from dryml.data.errors import TraversalError, TraversalOwnershipError
from dryml.data.errors import UnsupportedContinuationError, ContinuationMismatchError, ContinuationFormatError
from dryml.data.recovery import RecoverableGenerator, ReplayRecoveryPolicy
from dryml.data.continuation import ContinuationValue
from dryml.data.state import DatasetState, StatePath
from dryml.data.traversal import AcceptanceMode, DatasetTraversal, EpochView, PendingElement, TraversalPosition
from dryml.data.util import (
    Collect,
    collect_xy,
    collate_xy,
    iter_xy,
    materialize_supervised,
    as_supervised,
)


__all__ = [
    "Dataset",
    "DatasetCursor",
    "DatasetExhaustedError",
    "AcceptanceMode",
    "DatasetTraversal",
    "EpochView",
    "PendingElement",
    "TraversalPosition",
    "TraversalError",
    "TraversalOwnershipError",
    "UnsupportedContinuationError",
    "ContinuationMismatchError",
    "ContinuationFormatError",
    "RecoverableGenerator",
    "ReplayRecoveryPolicy",
    "ContinuationValue",
    "DatasetState",
    "StatePath",
    "Epoch",
    "PreparedDataset",
    "BatchPreparation",
    "NativeTrainingCursor",
    "GeneratorDataset",
    "ArrayDataset",
    "NpyFileDataset",
    "TFDSAdapter",
    "TorchDatasetAdapter",
    "Map",
    "Accumulate",
    "StreamDataset",
    "Pipe",
    "Project",
    "Select",
    "ArgMax",
    "Cast",
    "Flatten",
    "Scale",
    "Padding",
    "Zip",
    "Chain",
    "Batch",
    "Unbatch",
    "Take",
    "Skip",
    "Shuffle",
    "Repeat",
    "iter_xy",
    "collect_xy",
    "collate_xy",
    "materialize_supervised",
    "Collect",
    "as_supervised",
    "Abs",
    "ArrayMean",
    "ArrayQuantile",
    "Diff",
    "Equal",
    "MeanFinalize",
    "MeanInitial",
    "MeanUpdate",
    "ReductionMode",
    "ReservoirInitial",
    "ReservoirQuantile",
    "ReservoirUpdate",
    "Squared",
]
