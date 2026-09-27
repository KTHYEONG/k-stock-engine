"""Deterministic identity digests shared by every published artifact.

Lives in ``core`` so that both the dataset layer and the evidence catalog can
compute the same ``bronze:`` identity without importing a heavyweight module.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

__all__ = ["dataset_digest"]

_HASH_SEPARATOR = chr(0)


def dataset_digest(values: Iterable[str]) -> str:
    """Return a stable Bronze-style digest for a set of source identities.

    The result depends only on the set of values, never on their order or
    duplicates, so two readers that saw the same evidence always agree on one
    dataset identity.
    """
    normalized = sorted(dict.fromkeys(str(value) for value in values))
    return f"bronze:{hashlib.sha256(_HASH_SEPARATOR.join(normalized).encode('utf-8')).hexdigest()}"
