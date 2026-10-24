# Advisory Locking

`dryml.locking` is the dependency-light owner of DRYML's reusable native
advisory-lock mechanics. Store publication, SQLite query-index rebuild claims,
and future managed-operation ownership keep their own paths and lifecycle
policy while using this one implementation.

## APIs

`interprocess_lock(path)` preserves the Store writer's blocking, same-thread
recursive exclusive behavior. `interprocess_read_lock(path)` acquires a
blocking shared reader lease on POSIX and Windows. Independent readers overlap;
exclusive writers remain incompatible with every reader.

`FileLock(path, shared=False)` owns a retained descriptor. Call
`acquire(blocking=True)` to wait for a lease or `acquire(blocking=False)` for a
single probe; only recognized nonblocking contention returns `False`.
Instances are non-reentrant, `release()` is idempotent, and context-manager
exit closes only the instance's descriptor. An instance must be acquired and
released by the same thread: a cross-thread `release()` raises `LockError` and
leaves the owning thread's lease intact. Lock files are durable coordination
names and are never unlinked by this module. The Windows adapter locks byte zero
even when it lies past end-of-file, so acquiring a lock does not initialize or
otherwise alter an empty lock file.

SQLite-style consumers that already own a descriptor use
`try_lock_file(fd, shared=False)` and `unlock_file(fd)`. These functions never
open, close, or unlink caller resources. Unexpected I/O and unavailable native
primitives raise `LockError` or `LockUnavailableError` rather than being
reported as contention.

## Support Boundary

The adapters are POSIX `flock` and Windows `LockFileEx`/`UnlockFileEx` byte-range
locking. Windows uses `msvcrt.get_osfhandle` only to borrow the descriptor's native
handle; the exclusive CRT locking fallback is not used for reader leases.
`supports_advisory_locking(path)` checks only whether the direct parent has a
normal local-directory shape; it is not a filesystem qualification or a proof
of advisory-lock correctness. Current evidence covers the in-house host,
spawned-process contention and owner exit, POSIX fork cleanup, and Windows
adapter seams. Network and distributed filesystem behavior has not been
qualified. Future discovered platform or filesystem defects belong in this
module with a focused regression so all consumers receive the correction.

## Store StateKey Leases And Retention Order

`dryml.core.store.retention` owns Store policy on top of these native mechanics.
Each full StateKey has a persistent `.state-leases/<shard>/<cdef>/<codec-hash>.lock`
coordination name. Readers, ordinary publication and grant acquisition retain
shared leases; deletion and withdrawn-data recovery try exclusive leases
nonblockingly. Retained/in-use deletion never queues a request. Lock files are
never deleted during payload cleanup, and equal codec hashes in different CDef
buckets or Store roots do not share protection.

The order is:

1. An admitted managed lifetime owner, when supplied by its later owning layer;
   Core's root-intent disposition fence serializes each protect/adopt/release
   handshake across sequential target phases.
2. Full-key leases sorted by Store UUID/physical authority domain, CDef digest and
   codec hash. `repo_plan.state_use_leases()` acquires the complete planned set.
3. Store writer fences in stable authority-domain order.
4. Short SQL transactions or short caller control CAS sections, with no earlier
   locks acquired inside them and no overlapping SQL/control transactions.

Store writers never wait for key leases or perform graph preparation, payload
authentication or application hooks. A read validates bytes while holding its
lease, then yields without a writer fence. Native key acquisition also rechecks
the Store owner's liveness after waiting. Same-operation deletion while a shared
or exclusive recovery lease is held returns in-use; cleanup uses an independent
phase after those leases end. It never upgrades a lease. A growing held key set
is refused and requires release/reacquire/revalidation. Nested use of already
protected keys reuses the calling execution owner's protection. Independent
threads/tasks and forked processes cannot inherit that protection's lifetime.

Independent shared leases overlap on POSIX and Windows. This lets grant retirement
inspect a key while another process's cache cursor protects its data; subsequent
exclusive deletion still returns in-use. No in-process-only fallback claims
process-shared protection on an unsupported platform. Native lock failures
propagate rather than becoming evidence that a key is free. Windows-host execution
remains a separate platform verification gate.

Staging allocations separately retain native `.staging/.leases/<allocation>.lock`
leases and exact owner markers. Recovery/removal probes the named allocation
nonblockingly and preserves live or mismatched work. Releasing a staging lease or
process death does not authorize grant release: only durable owner disposition
does. ZipStore leases protect one extraction view and defer extraction cleanup;
they are not cross-archive authority or durable retention grants.
