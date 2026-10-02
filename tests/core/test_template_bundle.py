"""Regression coverage for the retired TemplateBundle public carrier."""

from __future__ import annotations

import pytest


def test_template_bundle_is_not_a_public_runtime_value():
    """Importing the retired carrier fails rather than preserving a compatibility alias."""

    with pytest.raises(ImportError):
        from dryml.core import TemplateBundle
