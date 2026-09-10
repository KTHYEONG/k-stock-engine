"""DART-primary lifecycle evidence collection."""
from __future__ import annotations

import base64
import hashlib
from dataclasses import replace
from datetime import date, datetime

from src.core.time import SessionCalendar
from src.data.lifecycle import (
    LifecycleCandidate,
    LifecycleEvidence,
    LifecycleResolutionKind,
    parse_dart_lifecycle_notice,
)
from src.data.schemas import PITDataError
from src.integrations.dart.client import DartApiError, DartTerminalError

_PRECEDENCE_KEYWORDS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, ("정리매매", "상장폐지")),
    (1, ("주식교환", "주식의포괄적교환", "주식이전")),
    (2, ("합병",)),
    (3, ("감자", "자본감소")),
)


def _precedence_of(report_name: str) -> int | None:
    for rank, keywords in _PRECEDENCE_KEYWORDS:
        if any(keyword in report_name for keyword in keywords):
            return rank
    return None  # pragma: no cover


class DartLifecycleCollector:
    def __init__(self, *, dart: object, calendar: SessionCalendar, coverage_start: date) -> None:
        self._dart = dart
        self._calendar = calendar
        self._coverage_start = coverage_start
        loader = getattr(dart, "load_corp_code_records", None)
        records = loader() if callable(loader) else dart.load_corp_codes()  # type: ignore[attr-defined]
        mapping: dict[str, tuple[str, str]] = {}
        for record in records or ():
            ticker = str(getattr(record, "ticker", "") or "").strip()
            corp_code = str(getattr(record, "corp_code", "") or "").strip()
            corp_name = str(getattr(record, "corp_name", "") or "").strip()
            if ticker and corp_code:
                mapping[ticker] = (corp_code, corp_name)
        self._ticker_to_corp = mapping

    def collect(self, candidate: LifecycleCandidate) -> LifecycleEvidence:
        mapped = self._ticker_to_corp.get(candidate.ticker)
        if mapped is None:
            return LifecycleEvidence(
                candidate=candidate,
                resolution_kind=LifecycleResolutionKind.UNRESOLVED,
                evidence_status="unresolved",
                evidence_reason="missing_corp_code",
                published_at=None,
                available_at=None,
                cleanup_start=None,
                cleanup_end=None,
                cash_settlement_per_share=None,
                successor_instrument_id=None,
                source_provider="opendart",
                source_url=None,
                document_receipt_no=None,
                document_sha256=None,
            )
        corp_code, _ = mapped
        try:
            disclosures = self._dart.list_disclosures(  # type: ignore[attr-defined]
                self._coverage_start,
                candidate.first_absent_session.date(),
                corp_code=corp_code,
            )
        except (DartApiError, OSError, ValueError, TypeError) as exc:  # pragma: no cover
            raise PITDataError(f"DART lifecycle discovery failed for {candidate.instrument_id!r}: {exc}") from exc
        if not isinstance(disclosures, list):  # pragma: no cover
            raise PITDataError(f"malformed DART lifecycle response for {candidate.instrument_id!r}")
        ranked: list[tuple[int, dict[str, str]]] = []
        for item in disclosures:
            if not isinstance(item, dict):  # pragma: no cover
                raise PITDataError(f"malformed DART lifecycle record for {candidate.instrument_id!r}")
            report_name = str(item.get("report_nm", ""))
            rank = _precedence_of(report_name)
            if rank is not None:
                ranked.append((rank, item))
        if not ranked:
            return LifecycleEvidence(
                candidate=candidate,
                resolution_kind=LifecycleResolutionKind.UNRESOLVED,
                evidence_status="unresolved",
                evidence_reason="missing_terms",
                published_at=None,
                available_at=None,
                cleanup_start=None,
                cleanup_end=None,
                cash_settlement_per_share=None,
                successor_instrument_id=None,
                source_provider="opendart",
                source_url=None,
                document_receipt_no=None,
                document_sha256=None,
            )
        best_rank = min(rank for rank, _item in ranked)
        candidates = sorted(
            (item for rank, item in ranked if rank == best_rank),
            key=lambda item: (str(item.get("rcept_dt", "")), str(item.get("rcept_no", ""))),
            reverse=True,
        )
        fallback: LifecycleEvidence | None = None
        verified_fallback: LifecycleEvidence | None = None
        for selected in candidates:
            receipt_no = str(selected.get("rcept_no", "")).strip()
            receipt_dt = str(selected.get("rcept_dt", "")).strip()
            if len(receipt_no) != 14 or not receipt_no.isdigit():  # pragma: no cover
                raise PITDataError(f"malformed DART lifecycle receipt for {candidate.instrument_id!r}")
            try:
                published_at = datetime(int(receipt_dt[:4]), int(receipt_dt[4:6]), int(receipt_dt[6:8]), tzinfo=__import__("src.core.time", fromlist=["KRX_TZ"]).KRX_TZ)
            except (ValueError, IndexError) as exc:  # pragma: no cover
                raise PITDataError(f"malformed DART lifecycle receipt date for {candidate.instrument_id!r}") from exc
            disclosure_url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt_no}"
            try:
                archive = self._dart.fetch_document_archive(receipt_no)  # type: ignore[attr-defined]
            except DartTerminalError:
                continue
            except (DartApiError, OSError, ValueError, TypeError) as exc:  # pragma: no cover
                raise PITDataError(f"DART lifecycle archive failed for {candidate.instrument_id!r}: {exc}") from exc
            if not isinstance(archive, (bytes, bytearray)) or len(archive) == 0:  # pragma: no cover
                raise PITDataError(f"malformed DART lifecycle archive for {candidate.instrument_id!r}")
            archive_bytes = bytes(archive)
            source_hash = hashlib.sha256(archive_bytes).hexdigest()
            try:
                evidence = parse_dart_lifecycle_notice(
                    candidate=candidate,
                    receipt_no=receipt_no,
                    disclosure_url=disclosure_url,
                    published_at=published_at,
                    archive=archive_bytes,
                    calendar=self._calendar,
                    source_hash=source_hash,
                )
            except PITDataError as exc:
                evidence = LifecycleEvidence(
                    candidate=candidate,
                    resolution_kind=LifecycleResolutionKind.UNRESOLVED,
                    evidence_status="unresolved",
                    evidence_reason=f"document_validation:{exc}",
                    published_at=published_at,
                    available_at=None,
                    cleanup_start=None,
                    cleanup_end=None,
                    cash_settlement_per_share=None,
                    successor_instrument_id=None,
                    source_provider="opendart",
                    source_url=disclosure_url,
                    document_receipt_no=receipt_no,
                    document_sha256=source_hash,
                )
            evidence = replace(evidence, archive_b64=base64.b64encode(archive_bytes).decode("ascii"))
            fallback = fallback or evidence
            if evidence.evidence_status == "verified":
                verified_fallback = verified_fallback or evidence
                if evidence.resolution_kind is LifecycleResolutionKind.CASH_SETTLEMENT:
                    return evidence
        if verified_fallback is not None:
            return verified_fallback
        if fallback is not None:
            return fallback
        return LifecycleEvidence(
            candidate=candidate,
            resolution_kind=LifecycleResolutionKind.UNRESOLVED,
            evidence_status="unresolved",
            evidence_reason="document_unavailable",
            published_at=None,
            available_at=None,
            cleanup_start=None,
            cleanup_end=None,
            cash_settlement_per_share=None,
            successor_instrument_id=None,
            source_provider="opendart",
            source_url=None,
            document_receipt_no=None,
            document_sha256=None,
        )
