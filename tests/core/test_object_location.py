"""Reject retired mutable object-location APIs."""

from dryml.core import Object, Repo


def test_current_store_authority_exposes_no_mutable_object_location_api():
    """Definition and StateRef authority replace object-directory locations."""

    assert not hasattr(Repo, "location")
    assert not hasattr(Object, "location")
