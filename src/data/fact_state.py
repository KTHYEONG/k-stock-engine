"""Canonical per-scope state inputs of the financial-facts and financial-quality builds.

Both the single-dataset CLI commands and the scope refresh must feed these
builds the same evidence files; resolving them here keeps the two entry points
from silently diverging (for example a refresh that forgets the quarantine and
drops every withheld filing from the quality table).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from src.core.pit import PITDataError
from src.data.datasets import dataset_digest, load_manifest

SUPERSEDED_RECEIPTS_FILE: Final = "dart_fact_superseded_receipts.json"
UNRESOLVED_EVENTS_FILE: Final = "financial_quality_unresolved_events.json"


def load_superseded_receipts(state_root: Path) -> frozenset[str]:
    """Return the Bronze fact receipt hashes explicitly superseded by a later page.

    A missing file means no receipt is superseded.

    Raises:
        PITDataError: the file exists but is not a JSON list of 64-hex hashes.
    """
    path = Path(state_root) / SUPERSEDED_RECEIPTS_FILE
    if not path.exists():
        return frozenset()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PITDataError(f"superseded receipts file is unreadable: {path}") from exc
    if not isinstance(raw, list) or not all(
        isinstance(item, str) and len(item) == 64 and all(c in "0123456789abcdef" for c in item) for item in raw
    ):
        raise PITDataError(f"superseded receipts file must be a JSON list of sha256 hashes: {path}")
    return frozenset(raw)


def unresolved_events_file(state_root: Path) -> Path | None:
    """Return the manually evidenced quality-event file, or ``None`` when the scope has none."""
    path = Path(state_root) / UNRESOLVED_EVENTS_FILE
    return path if path.exists() else None


def fact_quarantine_file(state_root: Path, facts_dataset_dir: Path) -> Path:
    """Resolve and verify the quarantine list written by the build of one facts dataset.

    The facts manifest records the file name and its sha256; the file must be
    present in the scope state and byte-identical, otherwise quality events
    would be derived from a different quarantine than the facts they describe.

    Raises:
        PITDataError: the manifest does not name a quarantine file, the file is
            missing, or its hash differs from the recorded one.
    """
    details = load_manifest(Path(facts_dataset_dir)).details
    name = details.get("quarantine_file")
    expected = details.get("quarantine_sha256")
    if not isinstance(name, str) or not name or not isinstance(expected, str) or not expected:
        raise PITDataError(f"financial facts dataset records no quarantine file: {facts_dataset_dir}")
    path = Path(state_root) / Path(name).name
    if not path.is_file():
        raise PITDataError(f"financial facts quarantine file is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise PITDataError(f"financial facts quarantine file hash mismatch: {path}")
    return path


def file_digest(path: Path | None) -> str:
    """Identity digest of one evidence file's bytes; an absent file digests as an empty input set."""
    if path is None:
        return dataset_digest([])
    return dataset_digest([hashlib.sha256(Path(path).read_bytes()).hexdigest()])
