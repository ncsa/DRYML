# Advisory Locking

`dryml.locking` is the dependency-light owner of DRYML's reusable native
advisory-lock mechanics. Store publication, SQLite query-index rebuild claims,
and future managed-operation ownership keep their own paths and lifecycle
policy while using this one implementation.

## APIs

`interprocess_lock(path)` preserves the Store writer's blocking, same-thread
recursive exclusive behavior. `interprocess_read_lock(path)` acquires a
blocking shared reader lease on POSIX. Windows uses its compatible exclusive
byte-range fallback, so readers remain safe but do not overlap.

`FileLock(path, shared=False)` owns a retained descriptor. Call
`acquire(blocking=True)` to wait for a lease or `acquire(blocking=False)` for a
single probe; only recognized nonblocking contention returns `False`.
Instances are non-reentrant, `release()` is idempotent, and context-manager
exit closes only the instance's descriptor. An instance must be acquired and
released by the same thread: a cross-thread `release()` raises `LockError` and
leaves the owning thread's lease intact. Lock files are durable coordination
names and are never unlinked by this module.

SQLite-style consumers that already own a descriptor use
`try_lock_file(fd, shared=False)` and `unlock_file(fd)`. These functions never
open, close, or unlink caller resources. Unexpected I/O and unavailable native
primitives raise `LockError` or `LockUnavailableError` rather than being
reported as contention.

## Support Boundary

The initial adapters are POSIX `flock` and Windows `msvcrt` byte-range locking.
`supports_advisory_locking(path)` checks only whether the direct parent has a
normal local-directory shape; it is not a filesystem qualification or a proof
of advisory-lock correctness. Current evidence covers the in-house host,
spawned-process contention and owner exit, POSIX fork cleanup, and Windows
adapter seams. Network and distributed filesystem behavior has not been
qualified. Future discovered platform or filesystem defects belong in this
module with a focused regression so all consumers receive the correction.
