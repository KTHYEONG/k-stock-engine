"""Shadow reconciliation of DB Securities EOD evidence against KRX daily-market Silver."""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, cast

import polars as pl

from src.core.pit import PITDataError
from src.data.evidence_sources import DBSEC_CLOSING_BAR_SOURCE, DBSEC_DAILY_CHART_SOURCE
from src.data.payload_store import read_payload
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry

__all__ = [
    "FieldMismatch",
    "ReconcileVerdict",
    "SessionReconciliation",
    "fallback_readiness",
    "load_dbsec_session_rows",
    "read_reconciliation_history",
    "reconcile_session",
    "write_reconciliation",
]


class ReconcileVerdict(StrEnum):
    PASS = "pass"  # noqa: S105 - reconciliation verdict, not a credential
    FAIL = "fail"
    NO_DATA = "no_data"


@dataclass(frozen=True, slots=True)
class FieldMismatch:
    ticker: str
    field: str
    krx: int | None
    db: int | None


@dataclass(frozen=True, slots=True)
class SessionReconciliation:
    session: date
    verdict: ReconcileVerdict
    krx_tradable: int
    db_daily_rows: int
    db_closing_bars: int
    coverage: float
    close_match_rate: float
    volume_match_rate: float
    open_match_rate: float
    high_match_rate: float
    low_match_rate: float
    missing_closing_bar: int
    samples: tuple[FieldMismatch, ...]


_SCHEMA: Final = {
    "ticker": pl.String,
    "db_open": pl.Int64,
    "db_high": pl.Int64,
    "db_low": pl.Int64,
    "db_last": pl.Int64,
    "db_volume": pl.Int64,
    "db_close": pl.Int64,
    "db_closing_bar_volume": pl.Int64,
}

_DAILY_FIELDS: Final = (
    ("Oprc", "db_open"),
    ("Hprc", "db_high"),
    ("Lprc", "db_low"),
    ("Prpr", "db_last"),
    ("AcmlVol", "db_volume"),
)
_CLOSING_FIELDS: Final = (
    ("Prpr", "db_close"),
    ("CntgVol", "db_closing_bar_volume"),
)

_TRADABLE_MARKETS: Final = frozenset({"KOSPI", "KOSDAQ"})


_INT64_MAX: Final = 2**63 - 1


def _strict_int(value: object, *, source: str, session: date, ticker: str, field: str) -> int | None:
    parsed = _parse_strict_int(value, source=source, session=session, ticker=ticker, field=field)
    if parsed is not None and not -_INT64_MAX - 1 <= parsed <= _INT64_MAX:
        raise PITDataError(f"DB {source} row has out-of-range {field} for {session.isoformat()}:{ticker}")
    return parsed


def _parse_strict_int(value: object, *, source: str, session: date, ticker: str, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise PITDataError(f"DB {source} row has non-integer {field} for {session.isoformat()}:{ticker}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise PITDataError(f"DB {source} row has non-integer {field} for {session.isoformat()}:{ticker}")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = Decimal(text.replace(",", ""))
        except InvalidOperation:
            raise PITDataError(
                f"DB {source} row has non-integer {field} for {session.isoformat()}:{ticker}"
            ) from None
        if parsed.is_finite() and parsed == parsed.to_integral_value():
            return int(parsed)
        raise PITDataError(f"DB {source} row has non-integer {field} for {session.isoformat()}:{ticker}")
    raise PITDataError(f"DB {source} row has non-integer {field} for {session.isoformat()}:{ticker}")


def _decode_rows(entry: ReceiptIndexEntry | None, *, source: str, session: date, ticker: str) -> Mapping[str, object] | None:
    if entry is None or entry.status is not EvidenceStatus.SUCCESS:
        return None
    raw = read_payload(Path(entry.payload_path))
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise PITDataError(f"DB {source} envelope is not valid JSON for {session.isoformat()}:{ticker}") from exc
    rows = document.get("rows") if isinstance(document, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise PITDataError(f"DB {source} envelope carries no rows for {session.isoformat()}:{ticker}")
    row = rows[0]
    if not isinstance(row, Mapping):
        raise PITDataError(f"DB {source} envelope carries no rows for {session.isoformat()}:{ticker}")
    return row


def load_dbsec_session_rows(catalog: ReceiptCatalog, session: date, tickers: Collection[str]) -> pl.DataFrame:
    """Decode stored DB daily-chart and closing-bar rows for ``session`` into one frame keyed by ticker; raises PITDataError on a malformed envelope."""
    ordered = sorted(set(tickers))
    columns: dict[str, list[Any]] = {name: [] for name in _SCHEMA}
    keys = {ticker: f"{session.isoformat()}:{ticker}" for ticker in ordered}
    daily_entries = catalog.latest(source=DBSEC_DAILY_CHART_SOURCE, natural_keys=set(keys.values()))
    closing_entries = catalog.latest(source=DBSEC_CLOSING_BAR_SOURCE, natural_keys=set(keys.values()))
    for ticker in ordered:
        key = keys[ticker]
        daily = _decode_rows(daily_entries.get(key), source=DBSEC_DAILY_CHART_SOURCE, session=session, ticker=ticker)
        closing = _decode_rows(
            closing_entries.get(key), source=DBSEC_CLOSING_BAR_SOURCE, session=session, ticker=ticker
        )
        values: dict[str, int | None] = {}
        if daily is not None:
            for raw_field, column in _DAILY_FIELDS:
                if raw_field not in daily:
                    raise PITDataError(
                        f"DB {DBSEC_DAILY_CHART_SOURCE} row is missing {raw_field} "
                        f"for {session.isoformat()}:{ticker}"
                    )
                values[column] = _strict_int(
                    daily[raw_field],
                    source=DBSEC_DAILY_CHART_SOURCE,
                    session=session,
                    ticker=ticker,
                    field=raw_field,
                )
        if closing is not None:
            for raw_field, column in _CLOSING_FIELDS:
                if raw_field not in closing:
                    raise PITDataError(
                        f"DB {DBSEC_CLOSING_BAR_SOURCE} row is missing {raw_field} "
                        f"for {session.isoformat()}:{ticker}"
                    )
                values[column] = _strict_int(
                    closing[raw_field],
                    source=DBSEC_CLOSING_BAR_SOURCE,
                    session=session,
                    ticker=ticker,
                    field=raw_field,
                )
        columns["ticker"].append(ticker)
        for name in _SCHEMA:
            if name != "ticker":
                columns[name].append(values.get(name))
    return pl.DataFrame({name: pl.Series(name, columns[name], dtype=dtype) for name, dtype in _SCHEMA.items()})


def _as_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _match_rate(pairs: Sequence[tuple[int | None, int | None]]) -> float:
    usable = [(left, right) for left, right in pairs if left is not None and right is not None]
    if not usable:
        return 0.0
    return sum(1 for left, right in usable if left == right) / len(usable)


def reconcile_session(
    *,
    session: date,
    krx: pl.DataFrame,
    db: pl.DataFrame,
    min_match_rate: float,
    min_coverage: float,
    sample_limit: int,
) -> SessionReconciliation:
    """Compare one session's DB frame with KRX daily-market Silver rows for the same session."""
    frame = krx
    if "price_state" in frame.columns:
        frame = frame.filter(pl.col("price_state") == "tradable")
    if "market" in frame.columns:
        frame = frame.filter(pl.col("market").is_in(_TRADABLE_MARKETS))
    db_daily_rows = (
        int(db.filter(pl.col("db_volume").is_not_null()).height) if "db_volume" in db.columns else 0
    )
    db_closing_bars = (
        int(db.filter(pl.col("db_close").is_not_null()).height) if "db_close" in db.columns else 0
    )
    if frame.height == 0:
        return SessionReconciliation(
            session=session,
            verdict=ReconcileVerdict.NO_DATA,
            krx_tradable=0,
            db_daily_rows=db_daily_rows,
            db_closing_bars=db_closing_bars,
            coverage=0.0,
            close_match_rate=0.0,
            volume_match_rate=0.0,
            open_match_rate=0.0,
            high_match_rate=0.0,
            low_match_rate=0.0,
            missing_closing_bar=0,
            samples=(),
        )
    krx_rows = frame.to_dicts()
    db_by_ticker: dict[str, dict[str, int | None]] = {}
    if db.height:
        for row in db.to_dicts():
            ticker = row.get("ticker")
            if ticker is None:
                continue
            db_by_ticker[str(ticker)] = {
                name: _as_int_or_none(row.get(name)) for name in _SCHEMA if name != "ticker"
            }
    krx_tradable = len(krx_rows)
    covered = 0
    close_pairs: list[tuple[int | None, int | None]] = []
    volume_pairs: list[tuple[int | None, int | None]] = []
    open_pairs: list[tuple[int | None, int | None]] = []
    high_pairs: list[tuple[int | None, int | None]] = []
    low_pairs: list[tuple[int | None, int | None]] = []
    missing_closing_bar = 0
    close_samples: list[FieldMismatch] = []
    volume_samples: list[FieldMismatch] = []
    open_samples: list[FieldMismatch] = []
    for record in krx_rows:
        ticker = str(record.get("ticker"))
        entry = db_by_ticker.get(ticker, {})
        krx_close = _as_int_or_none(record.get("close"))
        krx_volume = _as_int_or_none(record.get("volume"))
        krx_open = _as_int_or_none(record.get("open"))
        krx_high = _as_int_or_none(record.get("high"))
        krx_low = _as_int_or_none(record.get("low"))
        db_volume = entry.get("db_volume")
        db_close = entry.get("db_close")
        if db_volume is not None:
            covered += 1
        if db_close is None:
            missing_closing_bar += 1
            if krx_close is not None:
                close_samples.append(FieldMismatch(ticker=ticker, field="close", krx=krx_close, db=None))
        else:
            close_pairs.append((krx_close, db_close))
            if krx_close is not None and krx_close != db_close:
                close_samples.append(FieldMismatch(ticker=ticker, field="close", krx=krx_close, db=db_close))
        volume_pairs.append((krx_volume, db_volume))
        if krx_volume is not None and db_volume is not None and krx_volume != db_volume:
            volume_samples.append(FieldMismatch(ticker=ticker, field="volume", krx=krx_volume, db=db_volume))
        if krx_open is not None and krx_open > 0:
            db_open = entry.get("db_open")
            open_pairs.append((krx_open, db_open))
            if db_open is not None and krx_open != db_open:
                open_samples.append(FieldMismatch(ticker=ticker, field="open", krx=krx_open, db=db_open))
        high_pairs.append((krx_high, entry.get("db_high")))
        low_pairs.append((krx_low, entry.get("db_low")))
    coverage = covered / krx_tradable if krx_tradable else 0.0
    close_match_rate = _match_rate(close_pairs)
    volume_match_rate = _match_rate(volume_pairs)
    open_match_rate = _match_rate(open_pairs)
    high_match_rate = _match_rate(high_pairs)
    low_match_rate = _match_rate(low_pairs)
    verdict = (
        ReconcileVerdict.PASS
        if coverage >= min_coverage
        and close_match_rate >= min_match_rate
        and volume_match_rate >= min_match_rate
        else ReconcileVerdict.FAIL
    )
    close_samples.sort(key=lambda item: item.ticker)
    volume_samples.sort(key=lambda item: item.ticker)
    open_samples.sort(key=lambda item: item.ticker)
    limit = max(0, int(sample_limit))
    samples = tuple((close_samples + volume_samples + open_samples)[:limit])
    return SessionReconciliation(
        session=session,
        verdict=verdict,
        krx_tradable=krx_tradable,
        db_daily_rows=db_daily_rows,
        db_closing_bars=db_closing_bars,
        coverage=coverage,
        close_match_rate=close_match_rate,
        volume_match_rate=volume_match_rate,
        open_match_rate=open_match_rate,
        high_match_rate=high_match_rate,
        low_match_rate=low_match_rate,
        missing_closing_bar=missing_closing_bar,
        samples=samples,
    )


def fallback_readiness(history: Sequence[SessionReconciliation], *, ready_sessions: int) -> bool:
    """True iff the newest ``ready_sessions`` reconciled sessions are consecutive XKRX sessions and all PASS."""
    if isinstance(ready_sessions, bool) or int(ready_sessions) < 1:
        raise ValueError(f"invalid ready_sessions {ready_sessions!r}: must be a positive integer")
    counted = sorted(
        (item for item in history if item.verdict in (ReconcileVerdict.PASS, ReconcileVerdict.FAIL)),
        key=lambda item: item.session,
    )
    if len(counted) < int(ready_sessions):
        return False
    window = counted[-int(ready_sessions) :]
    if any(item.verdict is not ReconcileVerdict.PASS for item in window):
        return False
    reported = {item.session for item in history}
    from src.core.krx_calendar import xkrx_session_calendar

    for first, second in pairwise(window):
        try:
            calendar = xkrx_session_calendar(start=first.session, end=second.session)
            sessions = {instant.date() for instant in calendar.sessions}
        except ValueError:
            return False
        between = {day for day in sessions if first.session < day < second.session}
        if any(day not in reported for day in between):
            return False
    return True


def _report_payload(report: SessionReconciliation) -> dict[str, Any]:
    return {
        "session": report.session.isoformat(),
        "verdict": report.verdict.value,
        "krx_tradable": int(report.krx_tradable),
        "db_daily_rows": int(report.db_daily_rows),
        "db_closing_bars": int(report.db_closing_bars),
        "coverage": float(report.coverage),
        "close_match_rate": float(report.close_match_rate),
        "volume_match_rate": float(report.volume_match_rate),
        "open_match_rate": float(report.open_match_rate),
        "high_match_rate": float(report.high_match_rate),
        "low_match_rate": float(report.low_match_rate),
        "missing_closing_bar": int(report.missing_closing_bar),
        "samples": [
            {"ticker": item.ticker, "field": item.field, "krx": item.krx, "db": item.db}
            for item in report.samples
        ],
    }


def _parse_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"invalid integer {value!r}")
    return int(cast(int, value))


def _parse_report(document: object, *, origin: Path) -> SessionReconciliation:
    if not isinstance(document, Mapping):
        raise PITDataError(f"unreadable reconciliation report: {origin.name}")
    try:
        raw = dict(document)
        session = date.fromisoformat(str(raw["session"]))
        verdict = ReconcileVerdict(str(raw["verdict"]))
        raw_samples = raw["samples"]
        if not isinstance(raw_samples, list):
            raise ValueError("samples must be a list")
        samples = tuple(
            FieldMismatch(
                ticker=str(cast(Mapping[str, object], item)["ticker"]),
                field=str(cast(Mapping[str, object], item)["field"]),
                krx=_parse_int_or_none(cast(Mapping[str, object], item)["krx"]),
                db=_parse_int_or_none(cast(Mapping[str, object], item)["db"]),
            )
            for item in raw_samples
        )
        return SessionReconciliation(
            session=session,
            verdict=verdict,
            krx_tradable=int(cast(Any, raw["krx_tradable"])),
            db_daily_rows=int(cast(Any, raw["db_daily_rows"])),
            db_closing_bars=int(cast(Any, raw["db_closing_bars"])),
            coverage=float(cast(Any, raw["coverage"])),
            close_match_rate=float(cast(Any, raw["close_match_rate"])),
            volume_match_rate=float(cast(Any, raw["volume_match_rate"])),
            open_match_rate=float(cast(Any, raw["open_match_rate"])),
            high_match_rate=float(cast(Any, raw["high_match_rate"])),
            low_match_rate=float(cast(Any, raw["low_match_rate"])),
            missing_closing_bar=int(cast(Any, raw["missing_closing_bar"])),
            samples=samples,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PITDataError(f"unreadable reconciliation report: {origin.name}") from exc


def write_reconciliation(report: SessionReconciliation, *, root: Path) -> Path:
    """Write ``<root>/<session>.json`` atomically and return its path."""
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{report.session.isoformat()}.json"
    encoded = json.dumps(_report_payload(report), ensure_ascii=False, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_reconciliation_history(root: Path) -> tuple[SessionReconciliation, ...]:
    """Load every stored session report, ascending by session; unreadable files raise PITDataError."""
    directory = Path(root)
    if not directory.exists():
        return ()
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError as exc:
        raise PITDataError(f"unreadable reconciliation history: {directory}") from exc
    reports: list[SessionReconciliation] = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise PITDataError(f"unreadable reconciliation report: {path.name}") from exc
        reports.append(_parse_report(document, origin=path))
    return tuple(sorted(reports, key=lambda item: item.session))
