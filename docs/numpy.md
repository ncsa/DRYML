# NumPy Values And Array Storage

`dryml.numpy.RaggedTensor` is the owned packed-values/row-partitions reference
representation for ragged runtime tensors. It supports validated read-only buffers,
empty rows, nested partitions, dtype changes and row extraction without implicit
padding. See [Ragged Tensors](ragged.md) for construction and Dataset/Method usage.
It is an ordinary tensor value rather than a Serializable payload container.

## NumpyArray Storage

`dryml.numpy.NumpyArray` is a concrete `Serializable` container for one array.
Its constructor takes `shape` and `dtype` and initializes a zero array. Those two
small fields are searchable definition metadata; the array's values and bytes
live only in state. This is useful for a fixed set of probe points reused across
mesh/SDF evaluations and training runs.

## Create, Populate, Save And Reload

Using the existing active/default Repo in a materialization-enabled scope, such
as a dispatch worker:

```python
import numpy as np
from dryml.numpy import NumpyArray

points = NumpyArray(shape=(1024, 3), dtype="float32")
points.array = np.random.default_rng(123).uniform(-1, 1, (1024, 3)).astype("float32")
points_ref = points.save()

# Later, in a Repo connected to that Store:
restored = repo.load_state_ref(points_ref)
xyz = restored.array
assert xyz.shape == (1024, 3)
assert not xyz.flags.writeable
```

Repo selection is inherited: the standard Object machinery accepts its existing
`repo=` control keyword, and inherited `save(repo=...)` accepts an override.
Without overrides both use the existing session/default mechanisms. The class
itself performs no Repo lookup and retains no Repo constructor field. A caller
can establish the normal active Repo with `dryml.core.session.config(repo=repo)`.

Keep `points_ref`, the exact SavedNode returned by save. Loading it restores the
recorded point set without sampling again. In an orchestrator notebook, retain
this metadata and perform construction/loading inside a worker or an explicitly
permitted materialization scope, as for other stateful Objects.

Pass that SavedNode into later recipes through their normal `Ref`/`Mat` roles.
Passing the extracted ndarray as an ordinary constructor argument would instead
snapshot its bytes into that recipe's definition. Separate `NumpyArray` instances
have separate ObjectIds even when their shape/dtype definitions agree.

## Search By Shape And Dtype

```python
from dryml.core import Definition

matches = repo.query().saved().sel(
    Definition(NumpyArray, shape=(1024, 3), dtype="float32")
).collect()
```

Either field can be omitted from a selector to filter only the other. Search
reads metadata, without allocating zero arrays or loading `.npy` bytes. Results
are exact saved revisions: shape/dtype filtering does not implicitly select one
sampled point set or the newest state. Retain the chosen SavedNode for reuse.

`shape` accepts an integer or list/tuple of nonnegative integers and is retained
as a tuple. `()` creates a scalar; zero-sized dimensions are supported. Boolean
and nonintegral dimensions are rejected. `dtype` accepts NumPy dtype spellings,
dtype objects, scalar types, and supported canonical DRYML DTypes. Known aliases
are normalized during authoring without allocating arrays: e.g. `np.float32`,
`np.dtype("f4")`, and `"float32"` retain `"float32"`. Explicit byte order and string
widths use exact NumPy spellings where necessary. Use canonical strings in plain
`Definition` selectors, or `NumpyArray.defn(...)` to normalize accepted aliases.

The live `.shape` tuple and `.dtype` NumPy dtype are read-only configuration.
Changing those features requires a new container; replacement `.array` values
must match both exactly, without implicit resizing or casting. Restore checks the
same constraints before installation so payload bytes cannot contradict searches.

## Content Ownership And Saved Revisions

- A newly constructed container owns read-only zeros of the declared shape/dtype,
  has `has_content == True`, and can be saved immediately.
- Assigning `.array` requires an ndarray, copies it, and installs owned read-only
  content. Mutating or deleting the original input, including an input memmap's
  file, does not change the stored content. Array-subclass behavior is not kept.
  Shape/dtype mismatches leave the old array and saved receipt intact.
- Reading `.array` allocates a small read-only view, not a second array buffer.
  Replace the property to change content; for edits, modify a copy and assign it.
- Replacement clears the live last-state receipt. Saving again creates an exact
  new state for the same ObjectId; earlier saved references remain unchanged and
  can still be loaded independently.
- Validation or copy failure leaves existing content/receipts untouched. Hook
  decode failure leaves its prior array untouched; enclosing Core restoration
  still owns graph invalidation and requires fresh loading after a failed restore.
- Serialize assignment and state IO on a shared live instance, and keep input
  arrays stable during copying. Use independent loaded instances for independent
  editing. This class adds no managed computation or mutation coordination layer.

## Payload Contract

`NumpyArray` uses the `npy1` state codec and one `array.npy` file, with pickle
disabled. The codec name versions this container contract, not the NumPy header
version. Store format and Node identity protocols are unchanged. Normal Core
payload inventory, hashing, immutable publication and exact-source checks apply;
the container adds no alternate file authority or implicit external-file reuse.

Nonobject, unstructured ndarray dtypes that round-trip losslessly through NPY
without dtype metadata are supported, including scalars, empty arrays, non-native
byte order, and Fortran-contiguous arrays. Shape, dtype, and logical values survive
restoration. Object arrays, structured dtypes, dtype metadata, and custom dtypes
that NPY reduces to a different dtype are rejected before replacement. Ordinary
NumPy object arrays remain supported as runtime values; they are not this
non-pickle persistence format.

String dtypes require an explicit nonzero width, such as `"U8"` or `"S16"`, so the
declared dtype is fixed before zero initialization.

Save streams the owned array into Core's staging directory. Normal Core loading
constructs the zero array before eager restoration; it may temporarily hold that
buffer alongside the decoded payload. Restoration installs the decoded buffer
without another full-array copy or retained mmap.
Malformed, truncated, unsupported, and trailing payload bytes fail restoration.
IO failures propagate through the normal Repo save/load error boundary rather
than falling back to resampling or another saved state.
