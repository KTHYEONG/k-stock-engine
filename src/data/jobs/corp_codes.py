"""Refresh the DART corp-code bridge when the eligible universe needs it."""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, date, timedelta

from src.core.pit import EvidenceKind, PITDataError
from src.data.evidence_sources import DART_CORP_CODES_SOURCE
from src.data.jobs.runner import JobContext, JobUnit
from src.data.jobs.universe import eligible_tickers, read_corp_code_bridge, read_corp_code_bridge_rows
from src.data.receipt_catalog import EvidenceStatus
from src.data.scoped_ingestion import ScopedRawPayload

__all__ = ["DartCorpCodesJob"]

_KST = timedelta(hours=9)
_NATURAL_KEY = "dart_corp_codes"


def _kst_today(ctx: JobContext) -> date:
    moment = ctx.now()
    if moment.tzinfo is None:  # pragma: no cover - writer requires tz-aware time
        moment = moment.replace(tzinfo=UTC)
    return (moment.astimezone(UTC) + _KST).date()


class DartCorpCodesJob:
    """Refresh the DART corp-code bridge when the eligible universe needs it.

    One unit costs one ``corpCode.xml`` request. The new bridge is the union of
    the latest catalog bridge and the fresh records: a corp code once mapped is
    never dropped, because delisted companies disappear from ``corpCode.xml``
    while their history still needs the mapping.
    """

    name = "dart_corp_codes"

    def pending(self, ctx: JobContext) -> Sequence[JobUnit]:
        tickers = eligible_tickers(ctx)
        try:
            mapping, _ = read_corp_code_bridge(ctx.catalog)
        except PITDataError:
            return (
                JobUnit(
                    source=DART_CORP_CODES_SOURCE,
                    natural_key=_NATURAL_KEY,
                    payload={},
                    max_requests=1,
                ),
            )
        have = set(mapping.values())
        if any(ticker not in have for ticker in tickers):
            return (
                JobUnit(
                    source=DART_CORP_CODES_SOURCE,
                    natural_key=_NATURAL_KEY,
                    payload={},
                    max_requests=1,
                ),
            )
        entries = ctx.catalog.latest(source=DART_CORP_CODES_SOURCE, natural_keys={_NATURAL_KEY})
        entry = entries.get(_NATURAL_KEY)
        max_age = int(ctx.provider.dart.corp_codes_max_age_days)
        if entry is not None and entry.as_of is not None:
            age = (_kst_today(ctx) - entry.as_of).days
            if age > max_age:
                return (
                    JobUnit(
                        source=DART_CORP_CODES_SOURCE,
                        natural_key=_NATURAL_KEY,
                        payload={},
                        max_requests=1,
                    ),
                )
        elif entry is not None:
            retrieved = entry.retrieved_at
            if retrieved.tzinfo is None:  # pragma: no cover - writer requires tz-aware time
                retrieved = retrieved.replace(tzinfo=UTC)
            age = (_kst_today(ctx) - (retrieved.astimezone(UTC) + _KST).date()).days
            if age > max_age:
                return (
                    JobUnit(
                        source=DART_CORP_CODES_SOURCE,
                        natural_key=_NATURAL_KEY,
                        payload={},
                        max_requests=1,
                    ),
                )
        return ()

    def fetch(self, ctx: JobContext, units: Sequence[JobUnit]) -> Sequence[ScopedRawPayload]:
        out: list[ScopedRawPayload] = []
        for _unit in units:
            collector = ctx.collector
            if collector is None:
                raise PITDataError("DART corp-code endpoint is not configured")
            loader = getattr(collector, "fetch_corp_code_records", None)
            if loader is None:
                loader = getattr(collector, "load_corp_code_records", None)
            if loader is None:
                raise PITDataError("DART corp-code endpoint is not configured")
            fresh = tuple(loader())
            try:
                old_rows, _mapping, _hash = read_corp_code_bridge_rows(ctx.catalog)
            except PITDataError:
                old_rows = []
            old_by_code: dict[str, dict[str, str]] = {}
            for row in old_rows:
                code = str(row.get("corp_code") or "").strip()
                if code and code not in old_by_code:
                    old_by_code[code] = {
                        "corp_code": code,
                        "corp_name": str(row.get("corp_name") or "").strip(),
                        "ticker": str(row.get("ticker") or "").strip(),
                    }
            merged: dict[str, dict[str, str]] = dict(old_by_code)
            for record in fresh:
                ticker = str(getattr(record, "ticker", "") or "").strip()
                corp_code = str(getattr(record, "corp_code", "") or "").strip()
                corp_name = str(getattr(record, "corp_name", "") or "").strip()
                if not ticker or not corp_code:
                    continue
                previous = merged.get(corp_code)
                if previous is not None and previous["ticker"] != ticker:
                    raise PITDataError(
                        f"DART corp code {corp_code} remapped from {previous['ticker']} to {ticker}"
                    )
                if previous is not None:
                    merged[corp_code] = {
                        "corp_code": corp_code,
                        "corp_name": corp_name or previous["corp_name"],
                        "ticker": ticker,
                    }
                else:
                    merged[corp_code] = {
                        "corp_code": corp_code,
                        "corp_name": corp_name,
                        "ticker": ticker,
                    }
            if not merged:
                raise PITDataError("DART corpCode.xml contained no listed tickers")
            rows = [merged[code] for code in sorted(merged)]
            body = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
            out.append(
                ScopedRawPayload(
                    kind=EvidenceKind.SECURITY_MASTER,
                    source=DART_CORP_CODES_SOURCE,
                    natural_key=_NATURAL_KEY,
                    as_of=_kst_today(ctx),
                    fiscal_period=None,
                    status=EvidenceStatus.SUCCESS,
                    payload=body,
                    retrieved_at=ctx.now(),
                    source_label="opendart:corpCode.xml",
                )
            )
        return out

    def health_check(self, ctx: JobContext) -> None:
        ctx.collector.health_check()
