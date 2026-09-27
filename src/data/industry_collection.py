"""KIS industry classification collection into Bronze with per-ticker isolation."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.pit import EvidenceKind, PITDataError
from src.data.evidence_sources import KIS_INDUSTRY_SOURCE
from src.data.receipt_catalog import EvidenceStatus
from src.data.runtime import DataRuntime
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload

__all__ = [
    "collect_classification_with_isolation",
    "resolve_industry_symbols",
]

_LOG = logging.getLogger(__name__)


def eligible_universe_tickers(silver_root: Path) -> tuple[str, ...]:
    """Eligible tickers from the scope's certified ordinary universe.

    The universe's eligibility classification already excludes preferred
    shares at collection time, so industry collection starting from this set
    never requests a preferred share in the first place.
    """
    import polars as pl

    datasets = sorted(
        path
        for path in Path(silver_root).glob("ordinary_universe_*")
        if path.is_dir() and not path.name.startswith(".")
    )
    if len(datasets) != 1:
        raise PITDataError("ordinary universe requires exactly one published dataset")
    files = sorted(datasets[0].rglob("*.parquet"))
    if not files:
        raise PITDataError("ordinary universe has no published partitions")
    tickers = (
        pl.scan_parquet([str(path) for path in files])
        .filter(pl.col("eligible"))
        .select(pl.col("ticker").cast(pl.String))
        .unique()
        .collect()["ticker"]
        .sort()
        .to_list()
    )
    if not tickers:
        raise PITDataError("ordinary universe has no eligible tickers")
    return tuple(str(ticker) for ticker in tickers)


def resolve_industry_symbols(runtime: DataRuntime, symbols_from: Path | str | None) -> tuple[str, ...]:
    """Ticker set for one classification run, from a file or the universe."""
    if symbols_from is not None:
        text = Path(symbols_from).read_text(encoding="utf-8")
        cleaned = tuple(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))
        if not cleaned:
            raise PITDataError(f"industry symbols file has no tickers: {symbols_from}")
        return cleaned
    return eligible_universe_tickers(runtime.workspace.silver_root)


def _collected_moment(page: dict[str, Any]) -> datetime:
    collected_at = str(page.get("collected_at") or "")
    try:
        moment = datetime.fromisoformat(collected_at)
    except ValueError:
        moment = datetime.now(UTC)
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _industry_scoped_payload(
    *, symbol: str, page: dict[str, Any], moment: datetime
) -> ScopedRawPayload:
    """One KIS classification page as a scoped payload keyed by endpoint, symbol and date."""
    records = page.get("records")
    output = page.get("output")
    endpoint = str(page.get("endpoint") or "").strip()
    if not endpoint:
        raise PITDataError(f"KIS industry page for {symbol!r} is missing its endpoint")
    body = {
        "provider": "KIS",
        "endpoint": endpoint,
        "symbol": symbol,
        "collected_at": moment.isoformat(),
        "output": dict(output) if isinstance(output, dict) else {},
        "records": [dict(record) for record in records] if isinstance(records, list) else [],
    }
    collected_date = moment.date()
    return ScopedRawPayload(
        kind=EvidenceKind.INDUSTRY,
        source=KIS_INDUSTRY_SOURCE,
        natural_key=f"{endpoint}:{symbol}:{collected_date.isoformat()}",
        as_of=collected_date,
        fiscal_period=None,
        status=EvidenceStatus.SUCCESS if body["records"] else EvidenceStatus.EMPTY,
        payload=json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8"),
        retrieved_at=moment,
        source_label=f"KIS:{endpoint}:{symbol}:{collected_date.isoformat()}",
    )


def collect_classification_with_isolation(
    *,
    stage: str,
    collector_cls: Any,
    fetch_attr: str,
    bronze_root: Path,
    writer: ScopedBronzeWriter,
    symbols: tuple[str, ...],
    pace_seconds: float,
) -> dict[str, object]:
    """Collect one symbol at a time, isolating per-ticker PIT failures.

    The collector returns transport-only pages; this data-layer caller owns
    every Bronze write, and every write goes through the scoped writer so the
    page lands in the catalog as a ``kis_industry`` receipt with its blob.
    """
    from src.integrations.errors import ProviderError

    total = len(symbols)
    pages = 0
    skipped: dict[str, str] = {}
    for index, symbol in enumerate(symbols, start=1):
        collector = collector_cls((symbol,))
        try:
            fetch = getattr(collector, fetch_attr)
            payloads: list[ScopedRawPayload] = []
            for page in fetch(bronze_root=bronze_root):
                assert isinstance(page, dict)
                payloads.append(
                    _industry_scoped_payload(symbol=symbol, page=page, moment=_collected_moment(page))
                )
            writer.persist_many(tuple(payloads))
            pages += len(payloads)
        except (PITDataError, ProviderError) as exc:
            skipped[symbol] = str(exc)
        if index % 100 == 0:
            _LOG.info(
                "[DATA] stage=%s done=%d/%d ok=%d skipped=%d",
                stage,
                index,
                total,
                pages,
                len(skipped),
            )
        time.sleep(pace_seconds)
    if pages == 0:
        raise PITDataError(f"{stage} collected nothing ({total} requested, {len(skipped)} skipped)")
    return {
        "symbols_requested": total,
        "pages_collected": pages,
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
