"""KIS stock-master Bronze capture: recapture window, payload round-trip, fail-closed collector."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from src.config import load_provider_policy, load_runtime_config
from src.config.providers import ProviderPolicy
from src.core.pit import EvidenceKind, PITDataError
from src.data.evidence_sources import KIS_STOCK_MASTER_SOURCE
from src.data.jobs.kis_master import KisStockMasterJob, build_kis_master_job_context
from src.data.jobs.runner import JobContext, run_job
from src.data.payload_store import read_payload
from src.data.receipt_catalog import EvidenceStatus
from src.data.runtime import load_data_runtime
from src.data.scoped_ingestion import ScopedRawPayload
from src.integrations.kis.master import KisMasterFile

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
NOW = datetime(2026, 11, 26, 2, 0, tzinfo=UTC)  # 11:00 KST
ZIP_BYTES = b"PK-fixture-bytes"


class _FakeCollector:
    def __init__(self) -> None:
        self.markets: list[str] = []

    def fetch(self, market: str) -> KisMasterFile:
        self.markets.append(market)
        return KisMasterFile(
            market=market,
            file_name=f"{market}_code.mst.zip",
            content=ZIP_BYTES,
            last_modified="Wed, 26 Nov 2026 01:55:08 GMT",
        )


def _runtime(tmp_path: Path):  # type: ignore[no-untyped-def]
    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def _provider() -> ProviderPolicy:
    return load_provider_policy(load_runtime_config())


def _ctx(tmp_path: Path, *, collector: object | None = None, now: datetime = NOW) -> JobContext:
    context = build_kis_master_job_context(runtime=_runtime(tmp_path), provider=_provider(), collector=collector)  # type: ignore[arg-type]
    from dataclasses import replace

    return replace(context, now=lambda: now)


def _record_capture(ctx: JobContext, market: str, retrieved_at: datetime) -> None:
    ctx.writer.persist(
        ScopedRawPayload(
            kind=EvidenceKind.SECURITY_MASTER,
            source=KIS_STOCK_MASTER_SOURCE,
            natural_key=f"{market}:{retrieved_at.astimezone(UTC) + timedelta(hours=9):%Y-%m-%dT%H%M}",
            as_of=date(2026, 11, 26),
            fiscal_period=None,
            status=EvidenceStatus.SUCCESS,
            payload=f'{{"market": "{market}", "at": "{retrieved_at.isoformat()}"}}'.encode(),
            retrieved_at=retrieved_at,
            source_label=f"test:{market}",
        )
    )


def test_pending_two_markets_when_never_captured(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    units = KisStockMasterJob().pending(ctx)

    assert [unit.payload["market"] for unit in units] == ["kosdaq", "kospi"]


def test_pending_skips_recent_capture(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _record_capture(ctx, "kospi", NOW - timedelta(minutes=30))
    _record_capture(ctx, "kosdaq", NOW - timedelta(hours=3))

    units = KisStockMasterJob().pending(ctx)

    assert [unit.payload["market"] for unit in units] == ["kosdaq"]


def test_fetch_payload_round_trips(tmp_path: Path) -> None:
    collector = _FakeCollector()
    ctx = _ctx(tmp_path, collector=collector)

    run_job(
        KisStockMasterJob(), ctx, chunk_size=2, max_chunks=None, dry_run=False, emit=lambda _payload: None
    )

    assert collector.markets == ["kosdaq", "kospi"]
    entries = {entry.natural_key: entry for entry in ctx.catalog.entries(source=KIS_STOCK_MASTER_SOURCE)}
    assert set(entries) == {"kospi:2026-11-26T1100", "kosdaq:2026-11-26T1100"}
    entry = entries["kospi:2026-11-26T1100"]
    assert entry.as_of == date(2026, 11, 26)
    document = json.loads(read_payload(Path(entry.payload_path)))
    assert document["schema"] == "kis-stock-master-v1"
    assert document["market"] == "kospi"
    assert document["file"] == "kospi_code.mst.zip"
    assert document["last_modified"] == "Wed, 26 Nov 2026 01:55:08 GMT"
    assert document["zip_sha256"] == hashlib.sha256(ZIP_BYTES).hexdigest()
    assert base64.b64decode(document["zip_b64"]) == ZIP_BYTES
    assert document["observed_at"] == NOW.isoformat()


def test_unchanged_zip_recaptured_keeps_new_observation_time(tmp_path: Path) -> None:
    collector = _FakeCollector()
    later = NOW + timedelta(hours=3)
    for moment in (NOW, later):
        run_job(
            KisStockMasterJob(), _ctx(tmp_path, collector=collector, now=moment),
            chunk_size=2, max_chunks=None, dry_run=False, emit=lambda _payload: None,
        )

    ctx = _ctx(tmp_path)
    entries = {entry.natural_key: entry for entry in ctx.catalog.entries(source=KIS_STOCK_MASTER_SOURCE)}
    second = entries["kospi:2026-11-26T1400"]
    assert second.retrieved_at == later
    assert json.loads(read_payload(Path(second.payload_path)))["observed_at"] == later.isoformat()


def test_missing_collector_fails_closed(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, collector=None)
    units = KisStockMasterJob().pending(ctx)

    with pytest.raises(PITDataError):
        KisStockMasterJob().fetch(ctx, units[:1])


def test_health_check_requires_collector(tmp_path: Path) -> None:
    with pytest.raises(PITDataError):
        KisStockMasterJob().health_check(_ctx(tmp_path, collector=None))
    KisStockMasterJob().health_check(_ctx(tmp_path, collector=_FakeCollector()))
