"""KIS whole-market stock-master capture: raw zips into Bronze, no parsing."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from src.config.providers import ProviderPolicy
from src.core.pit import EvidenceKind, PITDataError
from src.data.evidence_sources import KIS_STOCK_MASTER_SOURCE
from src.data.jobs.runner import JobContext, JobUnit
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
from src.data.runtime import DataRuntime
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload
from src.integrations.kis.master import KisMasterClient
from src.integrations.quota import ProviderQuotaStateStore

__all__ = [
    "KisStockMasterJob",
    "build_kis_master_job_context",
]

_KST = timedelta(hours=9)
_SCHEMA = "kis-stock-master-v1"


def _kst(moment: datetime) -> datetime:
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return aware.astimezone(UTC) + _KST


class KisStockMasterJob:
    """Whole-market stock-master zips, one unit per market per capture window.

    The payload carries ``observed_at`` so an unchanged zip captured again is a distinct Bronze blob:
    Bronze is content-addressed, and part 2 needs every observation time, not only the first.
    """

    name = "kis_stock_master"
    source = KIS_STOCK_MASTER_SOURCE
    kind = EvidenceKind.SECURITY_MASTER

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        policy = ctx.provider.kis_master
        now = ctx.now()
        newest = _newest_retrieval_by_market(ctx.catalog)
        units: list[JobUnit] = []
        for market in sorted(policy.files):
            last = newest.get(market)
            if last is not None and now - last < timedelta(minutes=policy.min_recapture_minutes):
                continue
            units.append(JobUnit(source=self.source, natural_key=market, payload={"market": market}, max_requests=1))
        return tuple(units)

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        out: list[ScopedRawPayload] = []
        for unit in units:
            collector = ctx.collector
            if collector is None:
                raise PITDataError("KIS master endpoint is not configured")
            market = unit.payload["market"]
            master = collector.fetch(market)
            retrieved_at = ctx.now()
            retrieved_kst = _kst(retrieved_at)
            body = json.dumps(
                {
                    "schema": _SCHEMA,
                    "market": master.market,
                    "observed_at": retrieved_at.astimezone(UTC).isoformat(),
                    "file": master.file_name,
                    "last_modified": master.last_modified,
                    "zip_sha256": hashlib.sha256(master.content).hexdigest(),
                    "zip_b64": base64.b64encode(master.content).decode("ascii"),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
            out.append(
                ScopedRawPayload(
                    kind=self.kind,
                    source=self.source,
                    natural_key=f"{market}:{retrieved_kst:%Y-%m-%dT%H%M}",
                    as_of=retrieved_kst.date(),
                    fiscal_period=None,
                    status=EvidenceStatus.SUCCESS,
                    payload=body,
                    retrieved_at=retrieved_at,
                    source_label=f"kis:master:{market}",
                )
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        if ctx.collector is None:
            raise PITDataError("KIS master endpoint is not configured")


def _newest_retrieval_by_market(catalog: ReceiptCatalog) -> dict[str, datetime]:
    newest: dict[str, datetime] = {}
    for entry in catalog.entries(source=KIS_STOCK_MASTER_SOURCE):
        market, _, _ = entry.natural_key.partition(":")
        retrieved = entry.retrieved_at
        if retrieved.tzinfo is None:  # pragma: no cover - writer requires tz-aware time
            retrieved = retrieved.replace(tzinfo=UTC)
        if market not in newest or retrieved > newest[market]:
            newest[market] = retrieved
    return newest


def build_kis_master_job_context(
    *,
    runtime: DataRuntime,
    provider: ProviderPolicy,
    collector: KisMasterClient | None = None,
) -> JobContext:
    """Build the execution context for the public KIS master capture (no credential)."""
    quota_store = ProviderQuotaStateStore(runtime.workspace.state_root / "quota")
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    return JobContext(
        runtime=runtime,
        catalog=catalog,
        writer=ScopedBronzeWriter(runtime=runtime, catalog=catalog),
        provider=provider,
        quota_store=quota_store,
        runner=provider.runner("kis_master"),
        key_env="",
        collector=collector,
        now=lambda: datetime.now(UTC),
        sleep=time.sleep,
    )
