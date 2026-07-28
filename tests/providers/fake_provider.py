"""Import-light fake provider for provider/probe tests."""

from __future__ import annotations

import os
import sys
import time

import dryml.annotations as ann
from dryml.providers import DrymlProvider, OperationInspectionReport, ProviderIdentity, ProviderIssue


def target_fn():
    return None


class Provider(DrymlProvider):
    identity = ProviderIdentity("fake", "1", __name__, "Provider", ("operation_inspection",), {"fixture": True})

    def inspect_operation(self, request):
        options = request.provider_options.get("fake", {})
        if options.get("fail"):
            raise RuntimeError("intentional fake provider failure")
        if options.get("sleep"):
            time.sleep(float(options["sleep"]))
        if options.get("noisy"):
            print("hello from provider")
            print("warning from provider", file=sys.stderr)
        fragments = (
            ann.AnnotationFragment("environment", "requirement", {"requirements": ["numpy>=1"]}, ann.SourceTrace("provider")),
            ann.AnnotationFragment("world", "requirement", {"roles": {"main": {"resources": {"cpus": {"min": 1}}}}}, ann.SourceTrace("provider")),
            ann.AnnotationFragment("runtime", "default", {"frameworks": {"plain": {"provider": "fake"}}}, ann.SourceTrace("provider")),
        )
        return OperationInspectionReport(
            provider_identity=self.identity,
            status="ok",
            request_key=request.key,
            operation_id=request.operation_id,
            fragments=fragments,
            issues=(ProviderIssue("fake_ok", "info", "fake provider inspected operation", provider=self.identity.name),),
            metadata={
                "runtime_mode": _runtime_mode(),
                "runtime_enforcement": _runtime_enforcement(),
                "allocation": _allocation_repr(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
                "xla_visible_devices": os.environ.get("XLA_VISIBLE_DEVICES"),
                "heavy_imported": "DRYML_FAKE_HEAVY_IMPORTED" in os.environ,
            },
        )


def _runtime_mode():
    import dryml.runtime as runtime

    return runtime.active_runtime().mode.value


def _allocation_repr():
    import dryml.runtime as runtime

    return repr(runtime.active_runtime().allocation)


def _runtime_enforcement():
    import dryml.runtime as runtime

    return runtime.enforcement().value
