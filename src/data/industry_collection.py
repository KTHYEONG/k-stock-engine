"""KIS industry classification collection into Bronze with per-ticker isolation."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from src.core.pit import EvidenceKind, PITDataError
from src.data.evidence_sources import KIS_INDUSTRY_SOURCE
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog
from src.data.runtime import DataRuntime
from src.data.scoped_ingestion import ScopedBronzeWriter, ScopedRawPayload

__all__ = [
    "INDUSTRY_SNAPSHOT_ENDPOINT",
    "collect_classification_with_isolation",
    "current_universe_tickers",
    "resolve_industry_symbols",
    "tickers_missing_weekly_snapshot",
]

INDUSTRY_SNAPSHOT_ENDPOINT: Final = "inquire-price"

_LOG = logging.getLogger(__name__)

# An empty snapshot is KIS answering "no industry" for the ticker; re-asking within the week returns the same.
_ANSWERED: Final = frozenset({EvidenceStatus.SUCCESS, EvidenceStatus.EMPTY})


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


def current_universe_tickers(universe_path: Path) -> tuple[str, ...]:
    """Sorted eligible tickers on the latest session of one published ordinary-universe dataset.

    Raises:
        PITDataError: the dataset has no session partitions or no eligible ticker on its latest session.
    """
    import polars as pl

    sessions = sorted(
        path for path in Path(universe_path).glob("session=*") if path.is_dir() and not path.is_symlink()
    )
    if not sessions:
        raise PITDataError(f"ordinary universe has no published partitions: {universe_path}")
    files = sorted(sessions[-1].glob("*.parquet"))
    if not files:
        raise PITDataError(f"ordinary universe latest partition is empty: {sessions[-1]}")
    tickers = (
        pl.scan_parquet([str(path) for path in files], hive_partitioning=False)
        .filter(pl.col("eligible"))
        .select(pl.col("ticker").cast(pl.String))
        .unique()
        .collect()["ticker"]
        .sort()
        .to_list()
    )
    if not tickers:
        raise PITDataError(f"ordinary universe has no eligible tickers on {sessions[-1].name}")
    return tuple(str(ticker) for ticker in tickers)


def tickers_missing_weekly_snapshot(
    catalog: ReceiptCatalog, tickers: tuple[str, ...], *, today: date
) -> tuple[str, ...]:
    """Tickers lacking an answered (success or empty) ``inquire-price`` receipt dated within ``today``'s ISO week up to ``today``.

    ``today`` must use the same calendar basis as the receipt natural key (the collector's UTC date).
    """
    week_start = today - timedelta(days=today.weekday())
    days = tuple(week_start + timedelta(days=offset) for offset in range(today.weekday() + 1))
    keys = {f"{INDUSTRY_SNAPSHOT_ENDPOINT}:{ticker}:{day.isoformat()}": ticker for ticker in tickers for day in days}
    found = catalog.latest(source=KIS_INDUSTRY_SOURCE, natural_keys=keys)
    covered = {keys[key] for key, entry in found.items() if entry.status in _ANSWERED}
    return tuple(ticker for ticker in tickers if ticker not in covered)


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


_SKIP_REASON_CHARS = 240
_SKIP_REASON_LINES = 20


def _skip_reason(exc: BaseException) -> str:
    """One-line reason including the wrapped cause, which the adapter message alone hides."""
    cause = exc.__cause__
    text = str(exc) if cause is None else f"{exc} cause={type(cause).__name__}: {cause}"
    return " ".join(text.split())[:_SKIP_REASON_CHARS]


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
    empty: list[str] = []
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
            if any(payload.status is EvidenceStatus.EMPTY for payload in payloads):
                empty.append(symbol)
        except (PITDataError, ProviderError) as exc:
            skipped[symbol] = _skip_reason(exc)
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
    # Logged before the all-skipped raise: the caller only sees the exception type, not the reasons.
    for symbol in sorted(skipped)[:_SKIP_REASON_LINES]:
        _LOG.warning("[DATA] stage=%s_skip ticker=%s reason=%s", stage, symbol, skipped[symbol])
    if empty:
        _LOG.info("[DATA] stage=%s_empty count=%d head=%s", stage, len(empty), ",".join(empty[:_SKIP_REASON_LINES]))
    if pages == 0:
        raise PITDataError(f"{stage} collected nothing ({total} requested, {len(skipped)} skipped)")
    return {
        "symbols_requested": total,
        "pages_collected": pages,
        "empty_count": len(empty),
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
