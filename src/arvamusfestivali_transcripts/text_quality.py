"""Shared quality rules for transcript text."""

from __future__ import annotations


def has_meaningful_text(value: object) -> bool:
    """Return whether text contains at least one Unicode letter or digit."""
    return isinstance(value, str) and any(character.isalnum() for character in value)
