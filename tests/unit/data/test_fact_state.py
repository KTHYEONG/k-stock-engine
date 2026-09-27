"""Scope-state evidence resolution shared by the facts/quality commands and the refresh."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from src.core.pit import PITDataError
from src.data.datasets import DatasetIdentity, DatasetLayer, dataset_digest, publish_dataset
from src.data.fact_state import (
    SUPERSEDED_RECEIPTS_FILE,
    UNRESOLVED_EVENTS_FILE,
    fact_quarantine_file,
    file_digest,
    load_superseded_receipts,
    unresolved_events_file,
)


def _facts_dataset(silver: Path, details: dict[str, object]) -> Path:
    identity = DatasetIdentity(
        kind="financial_facts", layer=DatasetLayer.SILVER, policy_version="fixture-v1", inputs={}, params={}
    )
    return publish_dataset(
        layer_root=silver, identity=identity, partitions={"part.parquet": pl.DataFrame({"a": [1]})}, details=details
    ).path


def test_superseded_receipts_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_superseded_receipts(tmp_path) == frozenset()


def test_superseded_receipts_reads_hashes_and_rejects_malformed(tmp_path: Path) -> None:
    (tmp_path / SUPERSEDED_RECEIPTS_FILE).write_text(json.dumps(["a" * 64]), encoding="utf-8")
    assert load_superseded_receipts(tmp_path) == frozenset({"a" * 64})
    (tmp_path / SUPERSEDED_RECEIPTS_FILE).write_text(json.dumps(["not-a-hash"]), encoding="utf-8")
    with pytest.raises(PITDataError, match="sha256"):
        load_superseded_receipts(tmp_path)
    (tmp_path / SUPERSEDED_RECEIPTS_FILE).write_text("{", encoding="utf-8")
    with pytest.raises(PITDataError, match="unreadable"):
        load_superseded_receipts(tmp_path)


def test_unresolved_events_file_is_optional(tmp_path: Path) -> None:
    assert unresolved_events_file(tmp_path) is None
    (tmp_path / UNRESOLVED_EVENTS_FILE).write_text("[]", encoding="utf-8")
    assert unresolved_events_file(tmp_path) == tmp_path / UNRESOLVED_EVENTS_FILE


def test_fact_quarantine_file_is_verified_against_the_facts_manifest(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    quarantine = state / "dart_fact_quarantine_x.json"
    quarantine.write_text("[]", encoding="utf-8")
    sha = hashlib.sha256(quarantine.read_bytes()).hexdigest()
    facts = _facts_dataset(tmp_path / "silver", {"quarantine_file": quarantine.name, "quarantine_sha256": sha})

    assert fact_quarantine_file(state, facts) == quarantine

    quarantine.write_text("[{}]", encoding="utf-8")
    with pytest.raises(PITDataError, match="hash mismatch"):
        fact_quarantine_file(state, facts)
    quarantine.unlink()
    with pytest.raises(PITDataError, match="missing"):
        fact_quarantine_file(state, facts)


def test_fact_quarantine_file_requires_a_recorded_name(tmp_path: Path) -> None:
    facts = _facts_dataset(tmp_path / "silver", {})
    with pytest.raises(PITDataError, match="records no quarantine"):
        fact_quarantine_file(tmp_path, facts)


def test_file_digest_distinguishes_content_and_absence(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text("[1]", encoding="utf-8")
    assert file_digest(None) == dataset_digest([])
    assert file_digest(path) != file_digest(None)
    before = file_digest(path)
    path.write_text("[2]", encoding="utf-8")
    assert file_digest(path) != before
