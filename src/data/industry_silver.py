"""Certified current-state industry-classification Silver snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.data.bronze_aggregation import discover_verified_bronze_receipts
from src.data.industry_ksic_map import learn_ksic_industry_mapping
from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError

POLICY_VERSION = "kis-industry-classification-v2"

_SCHEMA: dict[str, Any] = {
    "ticker": pl.String,
    "instrument_id": pl.String,
    "industry_name": pl.String,
    "industry_basis": pl.String,
    "market_name": pl.String,
    "ksic_code": pl.String,
    "ksic_name": pl.String,
    "ksic_support": pl.Int64,
    "delisted_on": pl.Date,
    "available_at": pl.Datetime("us", "UTC"),
    "source_hash": pl.String,
    "ksic_source_hash": pl.String,
    "policy_version": pl.String,
}


@dataclass(frozen=True, slots=True)
class IndustryClassificationResult:
    dataset_path: Path
    dataset_id: str
    rows: int
    observed_rows: int
    inferred_rows: int
    unmapped_rows: int
    mapping_disagreements: int


def _load_payload(payload_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PITDataError(f"invalid verified KIS industry payload {payload_path}") from exc
    if not isinstance(payload, dict):
        raise PITDataError(f"invalid verified KIS industry payload {payload_path}")
    return payload


def _parse_symbol_and_time(payload: dict[str, Any], payload_path: Path) -> tuple[str, datetime]:
    symbol = payload.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise PITDataError(f"invalid verified KIS industry payload {payload_path}")
    try:
        collected_at = datetime.fromisoformat(str(payload.get("collected_at")))
    except (TypeError, ValueError) as exc:
        raise PITDataError(f"invalid verified KIS industry payload {payload_path}") from exc
    if collected_at.tzinfo is None:
        collected_at = collected_at.replace(tzinfo=UTC)
    return symbol, collected_at.astimezone(UTC)


def _parse_records(payload: dict[str, Any], payload_path: Path) -> dict[str, Any]:
    records = payload.get("records")
    if not isinstance(records, list) or not records or not isinstance(records[0], dict):
        raise PITDataError(f"invalid verified KIS industry payload {payload_path}")
    return records[0]


def _parse_quote_record(record: dict[str, Any], payload_path: Path) -> tuple[str, str]:
    industry = record.get("industry_name")
    if not isinstance(industry, str) or not industry.strip():
        raise PITDataError(f"invalid verified KIS industry payload {payload_path}")
    return industry, str(record.get("market_name") or "")


def _parse_stock_record(record: dict[str, Any], payload_path: Path) -> tuple[str, str, date | None]:
    code = str(record.get("ksic_code") or "").strip()
    if len(code) != 6 or not code.isascii() or not code.isdigit():
        raise PITDataError(f"invalid verified KIS stock-info payload {payload_path}")
    name = str(record.get("ksic_name") or "")
    raw_delisted = str(record.get("delisted_on") or "").strip()
    if not raw_delisted:
        return code, name, None
    try:
        return code, name, date.fromisoformat(raw_delisted)
    except ValueError as exc:
        raise PITDataError(f"invalid verified KIS stock-info payload {payload_path}") from exc


def materialize_industry_classification_silver(
    *, bronze_root: Path, silver_root: Path, symbols: frozenset[str] | None = None
) -> IndustryClassificationResult:
    """Build one row per ticker from two current-state KIS evidence streams.

    Both streams live under ``EvidenceKind.INDUSTRY`` and are told apart by the
    payload ``endpoint``: ``inquire-price`` (an observed industry name) and
    ``search-stock-info`` (KSIC code, KSIC name, listing-abolition date). A
    ticker's ``industry_name`` is the observed value when present; otherwise it
    is inferred from its KSIC code through a mapping learned only from tickers
    that have both streams; otherwise it is null. Inferred values are never
    presented as observed. The table is a current-state snapshot and must never
    be read as history for sessions before a row's ``available_at``.

    ``mapping_disagreements`` is a leave-one-out diagnostic: an observed ticker
    counts when a mapping learned from all OTHER paired tickers gives a
    different industry for its KSIC code. The observed value always wins.

    Args:
        bronze_root: Scope Bronze root holding certified ``INDUSTRY`` receipts.
        silver_root: Scope Silver root receiving ``industry_<hash16>/``.
        symbols: Optionally restricts which tickers' Bronze evidence is
            considered (for incremental/targeted rebuilds); ``None`` means
            "use every certified ``INDUSTRY`` receipt found under
            ``bronze_root``."

    Returns:
        The snapshot result; ``rows`` equals the number of tickers covered,
        split into ``observed_rows``, ``inferred_rows``, and
        ``unmapped_rows``, with ``mapping_disagreements`` counting observed
        tickers whose mapped prediction differs.

    Raises:
        PITDataError: hash-verification failure of any considered Bronze
            receipt, malformed certified payload, an unknown industry
            endpoint, no considered evidence, or an existing dataset with
            different content. ``symbols`` provided but empty is also
            rejected.
    """
    if symbols is not None and not symbols:
        raise PITDataError("industry classification requires a non-empty symbol filter")
    grouped = discover_verified_bronze_receipts(
        bronze_root=Path(bronze_root), kinds=frozenset({EvidenceKind.INDUSTRY})
    )
    receipts = grouped.get(EvidenceKind.INDUSTRY, ())
    quote_parsed: list[tuple[BronzeReceipt, str, datetime, str, str]] = []
    stock_parsed: list[tuple[BronzeReceipt, str, datetime, str, str, date | None]] = []
    for receipt in receipts:
        payload = _load_payload(receipt.payload_path)
        endpoint = payload.get("endpoint")
        if endpoint == "inquire-price":
            symbol, collected_at = _parse_symbol_and_time(payload, receipt.payload_path)
            industry, market = _parse_quote_record(
                _parse_records(payload, receipt.payload_path), receipt.payload_path
            )
            quote_parsed.append((receipt, symbol, collected_at, industry, market))
        elif endpoint == "search-stock-info":
            symbol, collected_at = _parse_symbol_and_time(payload, receipt.payload_path)
            ksic_code, ksic_name, delisted_on = _parse_stock_record(
                _parse_records(payload, receipt.payload_path), receipt.payload_path
            )
            stock_parsed.append((receipt, symbol, collected_at, ksic_code, ksic_name, delisted_on))
        else:
            raise PITDataError(f"unknown KIS industry endpoint in {receipt.payload_path}")
    if symbols is not None:
        selected = set(symbols)
        quote_filtered = [entry for entry in quote_parsed if entry[1] in selected]
        stock_filtered = [entry for entry in stock_parsed if entry[1] in selected]
        if not quote_filtered and not stock_filtered:
            if quote_parsed or stock_parsed:
                raise PITDataError("no certified INDUSTRY Bronze evidence matches the symbol filter")
            raise PITDataError("no certified INDUSTRY Bronze evidence found")
        quote_parsed = quote_filtered
        stock_parsed = stock_filtered
    if not quote_parsed and not stock_parsed:
        raise PITDataError("no certified INDUSTRY Bronze evidence found")
    considered = sorted(
        [receipt.content_hash for receipt, *_ in quote_parsed]
        + [receipt.content_hash for receipt, *_ in stock_parsed]
    )
    quote_best: dict[str, tuple[datetime, str, str, str]] = {}
    for receipt, symbol, collected_at, industry, market in quote_parsed:
        known = quote_best.get(symbol)
        if known is None or (collected_at, receipt.content_hash) > (known[0], known[1]):
            quote_best[symbol] = (collected_at, receipt.content_hash, industry, market)
    stock_best: dict[str, tuple[datetime, str, str, str, date | None]] = {}
    for receipt, symbol, collected_at, ksic_code, ksic_name, delisted_on in stock_parsed:
        known_stock = stock_best.get(symbol)
        if known_stock is None or (collected_at, receipt.content_hash) > (known_stock[0], known_stock[1]):
            stock_best[symbol] = (collected_at, receipt.content_hash, ksic_code, ksic_name, delisted_on)
    both_observations = [
        (ticker, stock_best[ticker][2], quote_best[ticker][2])
        for ticker in sorted(quote_best)
        if ticker in stock_best
    ]
    mapping = learn_ksic_industry_mapping(both_observations)
    industries_by_code: dict[str, set[str]] = {}
    for _ticker, code, industry in both_observations:
        industries_by_code.setdefault(code, set()).add(industry)
    mapping_disagreements = 0
    for ticker, code, industry in both_observations:
        if len(industries_by_code[code]) <= 1:
            continue
        others = [(t, c, i) for (t, c, i) in both_observations if t != ticker]
        prediction = learn_ksic_industry_mapping(others).lookup(code)
        if prediction is not None and prediction[0] != industry:
            mapping_disagreements += 1
    ticker_list: list[str] = []
    instrument_list: list[str] = []
    industry_list: list[str | None] = []
    basis_list: list[str] = []
    market_list: list[str | None] = []
    ksic_code_list: list[str | None] = []
    ksic_name_list: list[str | None] = []
    ksic_support_list: list[int | None] = []
    delisted_list: list[date | None] = []
    available_list: list[datetime] = []
    source_list: list[str] = []
    ksic_source_list: list[str | None] = []
    observed_rows = 0
    inferred_rows = 0
    unmapped_rows = 0
    for ticker in sorted(set(quote_best) | set(stock_best)):
        quote = quote_best.get(ticker)
        if quote is not None:
            collected_at, content_hash, industry, market = quote
            stock = stock_best.get(ticker)
            observed_rows += 1
            ticker_list.append(ticker)
            instrument_list.append(f"KRX:{ticker}")
            industry_list.append(industry)
            basis_list.append("observed_quote")
            market_list.append(market)
            source_list.append(content_hash)
            if stock is not None:
                stock_at, stock_hash, ksic_code, ksic_name, delisted_on = stock
                available_list.append(stock_at if stock_at > collected_at else collected_at)
                ksic_code_list.append(ksic_code)
                ksic_name_list.append(ksic_name)
                delisted_list.append(delisted_on)
                ksic_source_list.append(stock_hash)
            else:
                available_list.append(collected_at)
                ksic_code_list.append(None)
                ksic_name_list.append(None)
                delisted_list.append(None)
                ksic_source_list.append(None)
            ksic_support_list.append(None)
        else:
            stock_at, stock_hash, ksic_code, ksic_name, delisted_on = stock_best[ticker]
            prediction = mapping.lookup(ksic_code)
            ticker_list.append(ticker)
            instrument_list.append(f"KRX:{ticker}")
            market_list.append(None)
            available_list.append(stock_at)
            ksic_code_list.append(ksic_code)
            ksic_name_list.append(ksic_name)
            delisted_list.append(delisted_on)
            source_list.append(stock_hash)
            ksic_source_list.append(stock_hash)
            if prediction is not None:
                inferred_rows += 1
                industry_list.append(prediction[0])
                basis_list.append("ksic_inferred")
                ksic_support_list.append(prediction[1])
            else:
                unmapped_rows += 1
                industry_list.append(None)
                basis_list.append("unmapped")
                ksic_support_list.append(None)
    frame = (
        pl.DataFrame(
            {
                "ticker": ticker_list,
                "instrument_id": instrument_list,
                "industry_name": industry_list,
                "industry_basis": basis_list,
                "market_name": market_list,
                "ksic_code": ksic_code_list,
                "ksic_name": ksic_name_list,
                "ksic_support": ksic_support_list,
                "delisted_on": delisted_list,
                "available_at": available_list,
                "source_hash": source_list,
                "ksic_source_hash": ksic_source_list,
                "policy_version": [POLICY_VERSION for _ in ticker_list],
            },
            schema=_SCHEMA,
        ).sort("ticker")
    )
    silver_root = Path(silver_root)
    silver_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".industry-silver-", dir=silver_root))
    try:
        rel = Path("part.parquet")
        out_path = staging / rel
        frame.write_parquet(out_path)
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        dataset_id = "industry_" + hashlib.sha256(
            "\n".join((POLICY_VERSION, *considered)).encode("utf-8")
        ).hexdigest()[:16]
        target_path = silver_root / dataset_id
        manifest = {
            "dataset_id": dataset_id,
            "policy_version": POLICY_VERSION,
            "rows": frame.height,
            "observed_rows": observed_rows,
            "inferred_rows": inferred_rows,
            "unmapped_rows": unmapped_rows,
            "mapping_disagreements": mapping_disagreements,
            "conflicting_ksic": sorted(mapping.conflicting_ksic),
            "partitions": [{"path": str(rel), "row_count": frame.height, "parquet_sha256": digest}],
        }
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (staging / "manifest.json").write_text(encoded, encoding="utf-8")
        if target_path.exists():
            try:
                current = (target_path / "manifest.json").read_text(encoding="utf-8")
            except OSError as exc:
                raise PITDataError(
                    f"existing industry dataset is unreadable: {target_path}"
                ) from exc
            if current != encoded:
                raise PITDataError(f"existing industry dataset differs: {target_path}")
            shutil.rmtree(staging, ignore_errors=True)
        else:
            os.rename(staging, target_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return IndustryClassificationResult(
        dataset_path=target_path,
        dataset_id=dataset_id,
        rows=frame.height,
        observed_rows=observed_rows,
        inferred_rows=inferred_rows,
        unmapped_rows=unmapped_rows,
        mapping_disagreements=mapping_disagreements,
    )
