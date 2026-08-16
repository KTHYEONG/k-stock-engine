from __future__ import annotations

import math
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.stocks.data.catalog import CatalogEntry, CatalogKind, EvidenceCompleteness, CatalogStore
from src.stocks.data.contracts import CoverageRange
from src.stocks.data.outcome_open_bars import (
    build_outcome_open_frame,
    load_outcome_open_bar_evidence,
    publish_outcome_open_bar_dataset,
)


def test_projection_has_only_outcome_columns() -> None:
    # projection_has_only_outcome_columns
    frame = build_outcome_open_frame(
        [{"instrument_id": "B", "price_date": "2024-01-03", "open": 101.0, "high": 110.0}]
    )
    assert frame.columns == ["instrument_id", "price_date", "open"]
    assert frame.row(0) == ("B", frame["price_date"][0], 101.0)


@pytest.mark.parametrize("open_price", [0.0, -1.0, math.nan])
def test_projection_rejects_invalid_open_or_duplicate_key(open_price: float) -> None:
    # projection_rejects_invalid_open_or_duplicate_key
    with pytest.raises(ValueError, match="strictly positive"):
        build_outcome_open_frame(
            [{"instrument_id": "A", "price_date": "2024-01-02", "open": open_price}]
        )
    with pytest.raises(ValueError, match="duplicate"):
        build_outcome_open_frame(
            [
                {"instrument_id": "A", "price_date": "2024-01-02", "open": 100.0},
                {"instrument_id": "A", "price_date": "2024-01-02", "open": 101.0},
            ]
        )


def test_projection_publishes_verified_parquet_and_loads(tmp_path: Path) -> None:
    payload = {
        "record_count": 1,
        "records": [{"instrument_id": "A", "price_date": "2024-01-02", "open": 100.0}],
    }
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    raw_entry = CatalogEntry(
        kind=CatalogKind.RAW_BARS, name="raw", content_hash=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        schema_hash="schema", registered_at=datetime(2024, 1, 3, tzinfo=UTC),
        coverage=CoverageRange(start=date(2024, 1, 2), end=date(2024, 1, 2)),
        completeness=EvidenceCompleteness.COMPLETE, path=str(raw_path), row_count=1,
    )
    catalog_root = tmp_path / "catalog"
    CatalogStore(catalog_root).register(raw_entry)
    entry = publish_outcome_open_bar_dataset(
        raw_entry, tmp_path / "outcome_open", catalog_root, "open-v1", datetime(2024, 1, 3, tzinfo=UTC)
    )
    loaded_entry, frame = load_outcome_open_bar_evidence(
        CatalogStore(catalog_root), entry.name, datetime(2024, 1, 3, tzinfo=UTC)
    )
    assert loaded_entry == entry
    assert frame is not None
    assert frame.to_dicts() == [{"instrument_id": "A", "price_date": date(2024, 1, 2), "open": 100.0}]
