"""Single source for the eligible universe and the DART corp-code bridge.

Every collector derives the same eligible ticker set from the registry's
``ordinary_universe`` and the same corp-code mapping from the frozen Bronze
``dart_corp_codes`` receipt, so planning never diverges between jobs.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from src.core.pit import PITDataError
from src.data.dataset_registry import DatasetRegistry
from src.data.jobs.runner import JobContext
from src.data.receipt_catalog import ReceiptCatalog
from src.integrations.dart.client import DartCorpCodeRecord

__all__ = [
    "corp_code_bridge",
    "eligible_tickers",
    "index_corp_codes",
    "read_corp_code_bridge",
    "read_corp_code_bridge_rows",
]

_TICKER_PATTERN = r"^\d{6}$"


def eligible_tickers(ctx: JobContext) -> frozenset[str]:
    """Eligible tickers from the registry's current ordinary universe."""
    import re

    import polars as pl

    registry = DatasetRegistry(ctx.runtime.workspace.state_root)
    dataset_id = registry.require("ordinary_universe")
    dataset_dir = ctx.runtime.workspace.silver_root / dataset_id
    if not dataset_dir.is_dir():
        raise PITDataError(f"ordinary universe dataset is missing: {dataset_id}")
    files = sorted(dataset_dir.rglob("*.parquet"))
    if not files:
        raise PITDataError("ordinary universe has no published partitions")
    try:
        tickers = (
            pl.scan_parquet([str(path) for path in files])
            .filter(pl.col("eligible"))
            .select(pl.col("ticker").cast(pl.String))
            .unique()
            .collect()["ticker"]
            .to_list()
        )
    except Exception as exc:
        raise PITDataError(f"ordinary universe is unreadable: {dataset_id}") from exc
    cleaned = frozenset(
        str(ticker).strip() for ticker in tickers if str(ticker).strip() and re.match(_TICKER_PATTERN, str(ticker).strip())
    )
    if not cleaned:
        raise PITDataError("ordinary universe has no eligible tickers")
    return cleaned


def read_corp_code_bridge(catalog: ReceiptCatalog) -> tuple[dict[str, str], str]:
    """Corp-code to ticker mapping from the latest ``dart_corp_codes`` catalog receipt.

    Returns:
        The mapping and the content hash of the receipt it came from.

    Raises:
        PITDataError: no receipt, a hash mismatch, a malformed payload, or one
            corp code mapped to several tickers.
    """
    _, mapping, receipt_hash = read_corp_code_bridge_rows(catalog)
    return mapping, receipt_hash


def read_corp_code_bridge_rows(
    catalog: ReceiptCatalog,
) -> tuple[list[dict[str, str]], dict[str, str], str]:
    """Bridge rows, their corp-code mapping and the receipt hash, from one catalog read.

    Raises:
        PITDataError: as for :func:`read_corp_code_bridge`.
    """
    from src.data.evidence_sources import DART_CORP_CODES_SOURCE

    entry = catalog.latest(source=DART_CORP_CODES_SOURCE, natural_keys={"dart_corp_codes"}).get("dart_corp_codes")
    if entry is None:
        raise PITDataError("ticker bridge missing: no retained dart_corp_codes receipt")
    try:
        raw = Path(entry.payload_path).read_bytes()
    except OSError as exc:
        raise PITDataError("ticker bridge payload is unreadable") from exc
    if hashlib.sha256(raw).hexdigest() != entry.content_hash:
        raise PITDataError("ticker bridge payload hash mismatch")
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise PITDataError("ticker bridge payload is invalid") from exc
    if not isinstance(decoded, list):
        raise PITDataError("ticker bridge payload must be a list")
    rows: list[dict[str, str]] = []
    mapping: dict[str, str] = {}
    for row in decoded:
        if not isinstance(row, dict):
            raise PITDataError("ticker bridge row must be an object")
        corp_code = str(row.get("corp_code") or "").strip()
        ticker = str(row.get("ticker") or "").strip()
        if not corp_code or not ticker:
            raise PITDataError("ticker bridge row lacks corp_code or ticker")
        previous = mapping.get(corp_code)
        if previous is not None and previous != ticker:
            raise PITDataError(f"ticker bridge corp_code maps to multiple tickers: {corp_code}")
        mapping[corp_code] = ticker
        rows.append({"corp_code": corp_code, "corp_name": str(row.get("corp_name") or "").strip(), "ticker": ticker})
    if not mapping:
        raise PITDataError("ticker bridge payload is empty")
    return rows, mapping, entry.content_hash


def corp_code_bridge(ctx: JobContext) -> Mapping[str, str]:
    """Corp-code to ticker mapping from the frozen Bronze receipt."""
    mapping, _ = read_corp_code_bridge(ctx.catalog)
    return mapping


def index_corp_codes(records: Iterable[DartCorpCodeRecord]) -> dict[str, str]:
    """Ticker to corp-code index over listed tickers, failing on conflicts."""
    import re

    ticker_re = re.compile(_TICKER_PATTERN)
    code_by_ticker: dict[str, str] = {}
    for record in records:
        ticker = str(record.ticker).strip()
        corp_code = str(record.corp_code).strip()
        if not ticker or not ticker_re.match(ticker):
            continue
        if not corp_code:
            continue
        previous = code_by_ticker.get(ticker)
        if previous is not None and previous != corp_code:
            raise PITDataError(f"ticker {ticker} maps to multiple corp codes")
        code_by_ticker[ticker] = corp_code
    return code_by_ticker
