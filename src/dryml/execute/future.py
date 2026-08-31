from __future__ import annotations

from .transfer import restore_result


class OrchestratedFuture:
    def __init__(self, *, backend_future, prepared, update_targets, repo=None):
        self.backend_future = backend_future
        self.prepared = prepared
        self.update_targets = update_targets
        self.repo = repo
        self._result = None
        self._has_result = False

    def done(self) -> bool:
        return self.backend_future.done()

    def cancel(self) -> bool:
        return self.backend_future.cancel()

    def exception(self, timeout: float | None = None):
        try:
            self.result(timeout=timeout)
        except BaseException as exc:
            return exc
        return None

    def result(self, timeout: float | None = None):
        if self._has_result:
            return self._result

        response = self.backend_future.result(timeout=timeout)
        result = restore_result(
            response,
            repo=self.repo,
            result_store=self.prepared.result_store,
        )
        self._result = result
        self._has_result = True
        return result
