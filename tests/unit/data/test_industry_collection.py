"""Weekly KIS industry snapshot planning: current universe and per-week coverage."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.core.pit import EvidenceKind, PITDataError
from src.data.industry_collection import current_universe_tickers, tickers_missing_weekly_snapshot
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
from src.data.runtime import load_data_runtime
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
THURSDAY = date(2026, 10, 22)


def _partition(root: Path, session: str, rows: list[dict[str, object]]) -> None:
    part = root / f"session={session}"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema={"ticker": pl.String, "eligible": pl.Boolean}).write_parquet(part / "part.parquet")


def test_current_universe_uses_only_eligible_tickers_of_the_latest_session(tmp_path: Path) -> None:
    _partition(tmp_path, "2026-10-21", [{"ticker": "000001", "eligible": True}])
    _partition(
        tmp_path, "2026-10-22",
        [{"ticker": "000660", "eligible": True}, {"ticker": "005930", "eligible": True},
         {"ticker": "005935", "eligible": False}, {"ticker": "000660", "eligible": True}],
    )

    assert current_universe_tickers(tmp_path) == ("000660", "005930")


def test_current_universe_rejects_missing_or_empty_partitions(tmp_path: Path) -> None:
    with pytest.raises(PITDataError, match="no published partitions"):
        current_universe_tickers(tmp_path)
    (tmp_path / "session=2026-10-22").mkdir()
    with pytest.raises(PITDataError, match="latest partition is empty"):
        current_universe_tickers(tmp_path)
    _partition(tmp_path, "2026-10-23", [{"ticker": "005935", "eligible": False}])
    with pytest.raises(PITDataError, match="no eligible tickers on session=2026-10-23"):
        current_universe_tickers(tmp_path)


def _catalog_with(tmp_path: Path, receipts: list[tuple[str, date, EvidenceStatus]]) -> ReceiptCatalog:
    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    writer = ScopedBronzeWriter(runtime=runtime, catalog=catalog)
    payloads = []
    for ticker, day, status in receipts:
        endpoint = "inquire-price"
        payloads.append(
            ScopedRawPayload(
                kind=EvidenceKind.INDUSTRY,
                source="kis_industry",
                natural_key=f"{endpoint}:{ticker}:{day.isoformat()}",
                as_of=day,
                fiscal_period=None,
                status=status,
                payload=json.dumps({"endpoint": endpoint, "symbol": ticker, "day": day.isoformat()}).encode(),
                retrieved_at=datetime(day.year, day.month, day.day, 10, tzinfo=UTC),
                source_label=f"KIS:{endpoint}:{ticker}:{day.isoformat()}",
            )
        )
    writer.persist_many(tuple(payloads))
    return catalog


def test_weekly_snapshot_counts_answered_receipts_from_this_iso_week_up_to_today(tmp_path: Path) -> None:
    monday = THURSDAY - timedelta(days=THURSDAY.weekday())
    catalog = _catalog_with(
        tmp_path,
        [
            ("000010", monday, EvidenceStatus.SUCCESS),
            ("000020", THURSDAY, EvidenceStatus.SUCCESS),
            ("000030", monday - timedelta(days=1), EvidenceStatus.SUCCESS),
            ("000040", THURSDAY + timedelta(days=1), EvidenceStatus.SUCCESS),
            ("000050", monday, EvidenceStatus.EMPTY),
            ("000070", monday, EvidenceStatus.PROVIDER_ERROR),
        ],
    )
    tickers = ("000010", "000020", "000030", "000040", "000050", "000060", "000070")

    assert tickers_missing_weekly_snapshot(catalog, tickers, today=THURSDAY) == (
        "000030", "000040", "000060", "000070",
    )
    assert tickers_missing_weekly_snapshot(catalog, ("000030",), today=monday - timedelta(days=1)) == ()
    assert tickers_missing_weekly_snapshot(catalog, (), today=THURSDAY) == ()


def test_run_of_only_unassigned_industries_completes_and_covers_the_week(tmp_path: Path) -> None:
    from src.data.industry_collection import collect_classification_with_isolation
    from src.integrations.kis.industry import KisIndustryCollector

    class _Client:
        def inquire_price(self, symbol: str) -> dict[str, object]:
            return {"rprs_mrkt_kor_name": "KOSDAQ"}

    runtime = load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    summary = collect_classification_with_isolation(
        stage="kis_industry",
        collector_cls=lambda symbols: KisIndustryCollector(symbols, client=_Client()),
        fetch_attr="fetch_industry_classification",
        bronze_root=runtime.workspace.bronze_root,
        writer=ScopedBronzeWriter(runtime=runtime, catalog=catalog),
        symbols=("0001A0", "001000"),
        pace_seconds=0.0,
    )

    assert summary["skipped_count"] == 0
    assert summary["empty_count"] == 2
    today = datetime.now(UTC).date()
    assert tickers_missing_weekly_snapshot(catalog, ("0001A0", "001000"), today=today) == ()
