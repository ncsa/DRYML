"""Base class for optional DRYML provider methods."""

from __future__ import annotations

from .identity import ProviderIdentity
from .reports import AdapterPlanningReport, CompatibilityCheckReport, LoweringReport, OperationInspectionReport, RepresentationInspectionReport


class DrymlProvider:
    """Permissive provider base with structured unsupported defaults."""

    identity = ProviderIdentity("base", module=__name__, qualname="DrymlProvider")

    def inspect_operation(self, request):
        """Inspect an operation or return an unsupported report."""

        return OperationInspectionReport.unsupported(self.identity, request)

    def inspect_representations(self, request):
        """Inspect supported representations or return unsupported."""

        return RepresentationInspectionReport.unsupported(self.identity, request)

    def plan_adapters(self, request):
        """Plan representation adapters or return unsupported."""

        return AdapterPlanningReport.unsupported(self.identity, request)

    def check_compatibility(self, request):
        """Check compatibility or return unsupported."""

        return CompatibilityCheckReport.unsupported(self.identity, request)

    def lower_operation(self, request):
        """Plan operation lowering or return unsupported."""

        return LoweringReport.unsupported(self.identity, request)


__all__ = ["DrymlProvider"]
