"""Point-in-time exchange market actions: delisting, liquidation trading, administrative issues."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl

from src.core.digest import dataset_digest
from src.core.pit import PITDataError
from src.core.time import KRX_TZ, SessionCalendar
from src.data.dart_disclosures import iter_disclosure_records
from src.data.datasets import DatasetIdentity, DatasetLayer, publish_dataset
from src.data.evidence_sources import KRX_DAILY_MARKET_SOURCE
from src.data.jobs.kind import normalize_kind_title
from src.data.kind_notices import (
    iter_kind_notices,
    kind_source_digest,
    parse_kind_administrative_form,
    parse_kind_delisting_form,
    read_kind_document,
    require_kind_coverage,
)
from src.data.receipt_catalog import ReceiptCatalog

__all__ = [
    "MarketActionKind",
    "classify_market_action_title",
    "corp_bridge_digest",
    "daily_flags_source_digest",
    "disclosure_source_digest",
    "market_actions_dataset_inputs",
    "materialize_market_actions",
]

POLICY_VERSION = "market-actions-v6"

_LOG = logging.getLogger(__name__)

_SCHEMA: dict[str, Any] = {
    "instrument_id": pl.String,
    "ticker": pl.String,
    "kind": pl.String,
    "source": pl.String,
    "rcept_no": pl.String,
    "announced_on": pl.Date,
    "available_at": pl.Datetime("us", "Asia/Seoul"),
    "effective_start": pl.Date,
    "effective_end": pl.Date,
    "cancellation": pl.Boolean,
    "policy_version": pl.String,
}

_ADMIN_FLAG_VALUES = frozenset({"관리종목(소속부없음)", "투자주의환기종목(소속부없음)"})

_BRACKET_TAG = re.compile(r"\[[^\[\]]*\]")


class MarketActionKind(StrEnum):
    """Exchange notices that change whether an instrument may be entered."""

    DELISTING_DECIDED = "delisting_decided"
    LIQUIDATION_TRADING = "liquidation_trading"
    ADMINISTRATIVE_DESIGNATED = "administrative_designated"
    ADMINISTRATIVE_RELEASED = "administrative_released"
    TRADING_HALTED = "trading_halted"
    TRADING_RESUMED = "trading_resumed"


def _base_title(report_nm: str) -> str:
    """Return the title without bracket tags such as corrections."""
    return _BRACKET_TAG.sub("", report_nm or "").strip()


def _is_withdrawal_title(base: str) -> bool:
    """Whether a stripped title withdraws a previous notice."""
    return any(marker in base for marker in ("공시번복", "취소", "철회", "무효"))


def _is_delisting_cancellation_title(base: str) -> bool:
    """Whether a stripped title withdraws or court-suspends a delisting decision."""
    if "상장폐지" not in base:
        return False
    return _is_withdrawal_title(base) or ("가처분" in base and "인용" in base)


_TENTATIVE_MARKERS: tuple[str, ...] = ("우려", "여부", "예고", "미진행", "제외")
_NOT_AN_ACTION_MARKERS: tuple[str, ...] = ("해외", "이전상장", "의안상정", "자회사", "종속회사")


def _halt_kind(base: str) -> MarketActionKind | None:
    """Trading halt or resumption carried by a tentative notice; never a delisting or designation."""
    if ("매매거래정지" in base and "해제" in base) or "거래재개" in base:
        return MarketActionKind.TRADING_RESUMED
    if "매매거래정지" in base:
        return MarketActionKind.TRADING_HALTED
    return None


def classify_market_action_title(report_nm: str) -> MarketActionKind | None:
    """Map an exchange disclosure title to a market action, or None for unrelated titles.

    Only a fixed vocabulary is recognized; titles that merely mention a keyword inside another action
    (for example a company's own review-reason disclosure) are not actions. Corrections (``[기재정정]``)
    map like their base title. Withdrawals (``공시번복``, ``취소``) of a delisting decision map to None
    and are handled by the builder as a cancellation.

    A combined ``매매거래정지 및 정지해제`` notice announces a halt together with its release
    condition; the release is not effective at announcement, so it maps to a halt.

    Args:
        report_nm: Raw DART disclosure title.

    Returns:
        The mapped kind, or None when the title is not an exchange market action.
    """
    base = _base_title(report_nm)
    if not base:
        return None
    if _is_withdrawal_title(base):
        return None
    # 해외증권시장(ADR·GDR 등) 상장폐지, 코스닥 이전상장, 주총 안건 상정, 자회사 공시, 가처분 신청 자체는 국내 거래를 막는 조치가 아니다.
    if any(marker in base for marker in _NOT_AN_ACTION_MARKERS) or ("가처분" in base and "정리매매" not in base):
        return None
    if "정리매매" in base:
        return MarketActionKind.LIQUIDATION_TRADING
    normalized = re.sub(r"\s+", "", base)
    if "매매거래정지및정지해제" in normalized:
        return MarketActionKind.TRADING_HALTED
    # 거래소 안내 중 '우려', '여부 결정 안내', '예고', '미진행', '제외'는 확정된 조치가 아니다.
    # '상장폐지 관련'(상장폐지 사유 발생)도 이의신청·개선기간으로 번복될 수 있어 차단 사유로 쓰지 않는다.
    # 그 기간에는 매매거래정지가 걸리므로 체결 단계의 거래정지 거부가 이미 매수를 막는다.
    if any(marker in base for marker in _TENTATIVE_MARKERS):
        return _halt_kind(base)
    if "상장폐지" in base and "결정" in base:
        return MarketActionKind.DELISTING_DECIDED
    if ("관리종목" in base or "투자주의환기종목" in base) and "해제" in base:
        return MarketActionKind.ADMINISTRATIVE_RELEASED
    if ("관리종목" in base or "투자주의환기종목" in base) and "지정" in base and "사유" not in base:
        return MarketActionKind.ADMINISTRATIVE_DESIGNATED
    if ("매매거래정지" in base and "해제" in base) or "거래재개" in base:
        return MarketActionKind.TRADING_RESUMED
    if "매매거래정지" in base:
        return MarketActionKind.TRADING_HALTED
    return None


def disclosure_source_digest(catalog: ReceiptCatalog) -> str:
    """Stable digest of every retained disclosure row the builder consumes."""
    rows = [
        "\0".join((record.corp_code, record.rcept_no, record.rcept_dt.isoformat(), record.report_nm))
        for record in iter_disclosure_records(catalog)
    ]
    return dataset_digest(rows)


def daily_flags_source_digest(catalog: ReceiptCatalog, calendar: SessionCalendar) -> str:
    """Stable digest of the daily-market Bronze pages the flag reader consumes."""
    keys = {session.astimezone(KRX_TZ).date().isoformat() for session in calendar.sessions}
    entries = catalog.latest(source=KRX_DAILY_MARKET_SOURCE, natural_keys=keys)
    return dataset_digest([entries[key].content_hash for key in sorted(entries)])


def corp_bridge_digest(bridge: Mapping[str, str]) -> str:
    """Stable digest of the corp-code mapping the builder resolves tickers with."""
    return dataset_digest([f"{corp}\0{ticker}" for corp, ticker in sorted(bridge.items())])


def market_actions_dataset_inputs(
    *, bronze_disclosures: str, corp_code_bridge: str, bronze_daily: str, bronze_kind: str
) -> dict[str, str]:
    """Dataset inputs for the market-actions identity."""
    return {
        "bronze_disclosures": bronze_disclosures,
        "corp_code_bridge": corp_code_bridge,
        "bronze_daily": bronze_daily,
        "bronze_kind": bronze_kind,
    }


def _resolve_available_at(announced_on: date, *, calendar: SessionCalendar) -> datetime:
    """Return the open of the first session after the announcement date."""
    for session in calendar.sessions:
        if session.astimezone(KRX_TZ).date() > announced_on:
            return session
    raise PITDataError(f"no certified session open after announcement date {announced_on.isoformat()}")


def _load_daily_flag_rows(
    catalog: ReceiptCatalog, calendar: SessionCalendar
) -> dict[date, dict[str, bool]]:
    """Map each covered session to ``{ticker: flagged}`` from raw daily pages.

    Only the ``SECT_TP_NM`` administrative flag values are interpreted; prices,
    volumes and limits are never read here.
    """
    sessions = [session.astimezone(KRX_TZ).date() for session in calendar.sessions]
    entries = catalog.latest(
        source=KRX_DAILY_MARKET_SOURCE, natural_keys={day.isoformat() for day in sessions}
    )
    by_session: dict[date, dict[str, bool]] = {}
    for day in sessions:
        entry = entries.get(day.isoformat())
        if entry is None:
            continue
        try:
            raw = Path(entry.payload_path).read_bytes()
        except OSError as exc:
            raise PITDataError(f"KRX daily market payload is unreadable for {day}") from exc
        if hashlib.sha256(raw).hexdigest() != entry.content_hash:
            raise PITDataError(f"KRX daily market hash mismatch for {day}")
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise PITDataError(f"invalid KRX daily market JSON for {day}") from exc
        if not isinstance(payload, dict):
            raise PITDataError(f"invalid KRX daily market root for {day}")
        records = payload.get("records")
        if not isinstance(records, list):
            raise PITDataError(f"KRX daily market records must be a list for {day}")
        flags: dict[str, bool] = {}
        for record in records:
            if not isinstance(record, dict):
                raise PITDataError(f"KRX daily market record must be an object for {day}")
            # 일별 시세 페이지는 시기에 따라 ISU_SRT_CD 없이 6자리 ISU_CD만 싣는다(daily_market_silver와 같은 규칙).
            ticker = str(record.get("ISU_SRT_CD") or record.get("ISU_CD") or "").strip()
            if not ticker:
                continue
            flagged = str(record.get("SECT_TP_NM") or "").strip() in _ADMIN_FLAG_VALUES
            flags[ticker] = flags.get(ticker, False) or flagged
        by_session[day] = flags
    return by_session


def _resolve_kind_available_at(disclosed_at: datetime, *, calendar: SessionCalendar) -> datetime:
    """Return the open of the first certified session strictly after ``disclosed_at``."""
    moment = disclosed_at if disclosed_at.tzinfo is not None else disclosed_at.replace(tzinfo=KRX_TZ)
    for session in calendar.sessions:
        if session > moment:
            return session
    raise PITDataError(f"no certified session open after KIND notice at {disclosed_at.isoformat()}")


def _kind_ticker_present(
    ticker: str, announced_on: date, *, covered_days: list[date], flag_pages: dict[date, dict[str, bool]]
) -> bool:
    """Whether ``ticker`` is listed on the latest covered daily page on or before ``announced_on``."""
    latest: date | None = None
    for day in covered_days:
        if day <= announced_on:
            latest = day
        else:
            break
    if latest is None:
        return False
    return ticker in flag_pages.get(latest, {})


def materialize_market_actions(
    *,
    catalog: ReceiptCatalog,
    silver_root: Path,
    calendar: SessionCalendar,
    bridge: Mapping[str, str],
    kind_keywords: Sequence[str],
    kind_coverage_start: date,
) -> Path:
    """Publish ``market_actions_<hash16>`` from three point-in-time channels: DART exchange
    notices (date precision), KIND exchange notices and forms (minute precision), and KOSDAQ
    daily administrative flags. KIND fills KOSPI notices DART does not carry.

    Columns: ``instrument_id``, ``ticker``, ``kind``, ``source`` (``"dart"``, ``"kind"``,
    ``"krx_flag"``), ``rcept_no``, ``announced_on`` (KST date), ``available_at`` (open of the
    first session after the announcement), ``effective_start`` and ``effective_end`` (sessions,
    when the notice states a period, else null), ``policy_version``.

    Args:
        catalog: Bronze receipt catalog holding disclosure rows and daily pages.
        silver_root: Silver layer receiving the dataset.
        calendar: Certified session opens bounding availability.
        bridge: Corp-code to ticker mapping; unknown codes are counted and skipped.
        kind_keywords: KIND title keywords whose quarterly coverage must be complete; an empty
            sequence declares that no KIND channel is consumed.
        kind_coverage_start: First day of the KIND collection windows (the scope's evidence
            start). Window keys are anchored here exactly as the collector anchors them, so the
            coverage check addresses the same receipts the collector wrote.

    Returns:
        The published dataset directory.

    Raises:
        PITDataError: a KIND coverage gap, a missing KIND form body, or no certified
            session opens after an announcement date.
    """
    if len(calendar.sessions) < 2:
        raise PITDataError("market actions require at least two certified sessions")
    require_kind_coverage(
        catalog,
        keywords=kind_keywords,
        start=kind_coverage_start,
        through=calendar.sessions[-1].astimezone(KRX_TZ).date(),
    )
    rows: list[dict[str, Any]] = []
    unmapped_rows = 0
    for record in iter_disclosure_records(catalog):
        base = _base_title(record.report_nm)
        kind = classify_market_action_title(record.report_nm)
        cancellation = kind is None and _is_delisting_cancellation_title(base)
        if kind is None and not cancellation:
            continue
        ticker = bridge.get(record.corp_code)
        if ticker is None:
            unmapped_rows += 1
            continue
        rows.append(
            {
                "instrument_id": f"KRX:{ticker}",
                "ticker": ticker,
                "kind": (
                    kind.value if kind is not None else MarketActionKind.DELISTING_DECIDED.value
                ),
                "source": "dart",
                "rcept_no": record.rcept_no,
                "announced_on": record.rcept_dt,
                "available_at": _resolve_available_at(record.rcept_dt, calendar=calendar),
                "effective_start": None,
                "effective_end": None,
                "cancellation": cancellation,
                "policy_version": POLICY_VERSION,
            }
        )
    designations = 0
    releases = 0
    flagged_before: dict[str, bool] = {}
    flag_pages = _load_daily_flag_rows(catalog, calendar)
    covered_days = sorted(flag_pages)
    kind_actions = 0
    kind_cancellations = 0
    kind_unresolved_rows = 0
    kind_other_share_class = 0
    for notice in iter_kind_notices(catalog):
        if not notice.submitter.endswith("시장본부") or not notice.company_code:
            continue
        announced_on = notice.disclosed_at.astimezone(KRX_TZ).date()
        available_at = _resolve_kind_available_at(notice.disclosed_at, calendar=calendar)
        rcept_no = f"KIND-{notice.acptno}"
        normalized = normalize_kind_title(notice.title)
        if normalized == "상장폐지":
            body = read_kind_document(catalog, notice.acptno)
            if body is None:
                raise PITDataError(f"KIND delisting notice {notice.acptno} has no stored body")
            delisting_form = parse_kind_delisting_form(body)
            for ticker in delisting_form.tickers:
                if not _kind_ticker_present(
                    ticker, announced_on, covered_days=covered_days, flag_pages=flag_pages
                ):
                    kind_unresolved_rows += 1
                    continue
                rows.append(
                    {
                        "instrument_id": f"KRX:{ticker}",
                        "ticker": ticker,
                        "kind": MarketActionKind.DELISTING_DECIDED.value,
                        "source": "kind",
                        "rcept_no": rcept_no,
                        "announced_on": announced_on,
                        "available_at": available_at,
                        "effective_start": delisting_form.liquidation_start,
                        "effective_end": delisting_form.delisting_date,
                        "cancellation": False,
                        "policy_version": POLICY_VERSION,
                    }
                )
                kind_actions += 1
            continue
        if normalized in ("관리종목지정", "관리종목지정해제"):
            body = read_kind_document(catalog, notice.acptno)
            if body is None:
                raise PITDataError(f"KIND administrative notice {notice.acptno} has no stored body")
            admin_form = parse_kind_administrative_form(body)
            if not admin_form.is_common:
                kind_other_share_class += 1
                continue
            ticker = f"{notice.company_code}0"
            if not _kind_ticker_present(
                ticker, announced_on, covered_days=covered_days, flag_pages=flag_pages
            ):
                kind_unresolved_rows += 1
                continue
            rows.append(
                {
                    "instrument_id": f"KRX:{ticker}",
                    "ticker": ticker,
                    "kind": (
                        MarketActionKind.ADMINISTRATIVE_DESIGNATED.value
                        if normalized == "관리종목지정"
                        else MarketActionKind.ADMINISTRATIVE_RELEASED.value
                    ),
                    "source": "kind",
                    "rcept_no": rcept_no,
                    "announced_on": announced_on,
                    "available_at": available_at,
                    "effective_start": admin_form.effective_on,
                    "effective_end": None,
                    "cancellation": False,
                    "policy_version": POLICY_VERSION,
                }
            )
            kind_actions += 1
            continue
        base = _base_title(notice.title)
        kind = classify_market_action_title(notice.title)
        cancellation = kind is None and _is_delisting_cancellation_title(base)
        if kind is None and not cancellation:
            continue
        ticker = f"{notice.company_code}0"
        if not _kind_ticker_present(
            ticker, announced_on, covered_days=covered_days, flag_pages=flag_pages
        ):
            kind_unresolved_rows += 1
            continue
        rows.append(
            {
                "instrument_id": f"KRX:{ticker}",
                "ticker": ticker,
                "kind": (
                    kind.value if kind is not None else MarketActionKind.DELISTING_DECIDED.value
                ),
                "source": "kind",
                "rcept_no": rcept_no,
                "announced_on": announced_on,
                "available_at": available_at,
                "effective_start": None,
                "effective_end": None,
                "cancellation": cancellation,
                "policy_version": POLICY_VERSION,
            }
        )
        if cancellation:
            kind_cancellations += 1
        else:
            kind_actions += 1
    for day in sorted(flag_pages):
        for ticker in sorted(flag_pages[day]):
            flagged = flag_pages[day][ticker]
            was_flagged = flagged_before.get(f"KRX:{ticker}", False)
            if flagged and not was_flagged:
                rows.append(
                    {
                        "instrument_id": f"KRX:{ticker}",
                        "ticker": ticker,
                        "kind": MarketActionKind.ADMINISTRATIVE_DESIGNATED.value,
                        "source": "krx_flag",
                        "rcept_no": f"KRX-SECT-{ticker}-{day.isoformat()}",
                        "announced_on": day,
                        "available_at": _resolve_available_at(day, calendar=calendar),
                        "effective_start": day,
                        "effective_end": None,
                        "cancellation": False,
                        "policy_version": POLICY_VERSION,
                    }
                )
                designations += 1
            elif was_flagged and not flagged:
                rows.append(
                    {
                        "instrument_id": f"KRX:{ticker}",
                        "ticker": ticker,
                        "kind": MarketActionKind.ADMINISTRATIVE_RELEASED.value,
                        "source": "krx_flag",
                        "rcept_no": f"KRX-SECT-{ticker}-{day.isoformat()}",
                        "announced_on": day,
                        "available_at": _resolve_available_at(day, calendar=calendar),
                        "effective_start": day,
                        "effective_end": None,
                        "cancellation": False,
                        "policy_version": POLICY_VERSION,
                    }
                )
                releases += 1
            flagged_before[f"KRX:{ticker}"] = flagged
    rows.sort(key=lambda row: (str(row["announced_on"]), str(row["rcept_no"]), str(row["kind"])))
    frame = (
        pl.DataFrame(rows, schema=_SCHEMA).sort(["announced_on", "rcept_no"])
        if rows
        else pl.DataFrame([], schema=_SCHEMA)
    )
    identity = DatasetIdentity(
        kind="market_actions",
        layer=DatasetLayer.SILVER,
        policy_version=POLICY_VERSION,
        inputs=market_actions_dataset_inputs(
            bronze_disclosures=disclosure_source_digest(catalog),
            corp_code_bridge=corp_bridge_digest(bridge),
            bronze_daily=daily_flags_source_digest(catalog, calendar),
            bronze_kind=kind_source_digest(catalog),
        ),
        params={
            "calendar_digest": hashlib.sha256(
                "\n".join(session.astimezone(KRX_TZ).isoformat() for session in calendar.sessions).encode(
                    "utf-8"
                )
            ).hexdigest(),
        },
    )
    published = publish_dataset(
        layer_root=Path(silver_root),
        identity=identity,
        partitions={"part-00000.parquet": frame},
        details={
            "actions": sum(1 for row in rows if not row["cancellation"]),
            "cancellations": sum(1 for row in rows if row["cancellation"]),
            "administrative_designations": designations,
            "administrative_releases": releases,
            "unmapped_rows": unmapped_rows,
            "kind_actions": kind_actions,
            "kind_cancellations": kind_cancellations,
            "kind_unresolved_rows": kind_unresolved_rows,
            "kind_other_share_class": kind_other_share_class,
        },
    )
    _LOG.info(
        "[DATA] stage=market_actions actions=%d cancellations=%d kind_actions=%d "
        "kind_cancellations=%d kind_unresolved_rows=%d kind_other_share_class=%d",
        sum(1 for row in rows if not row["cancellation"]),
        sum(1 for row in rows if row["cancellation"]),
        kind_actions,
        kind_cancellations,
        kind_unresolved_rows,
        kind_other_share_class,
    )
    return published.path
