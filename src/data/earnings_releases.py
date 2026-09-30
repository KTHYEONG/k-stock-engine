"""Silver early-earnings releases built from preliminary and profit-change filings."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.config.providers import EarningsReleasePolicy
from src.core.pit import PITDataError
from src.core.time import KRX_TZ, SessionCalendar
from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    dataset_digest,
    publish_dataset,
    read_dataset,
)
from src.data.evidence_sources import EARNINGS_RELEASE_SOURCE
from src.data.receipt_catalog import ReceiptCatalog
from src.integrations.dart.earnings_release import (
    EarningsRelease,
    EarningsReleaseParseError,
    ReleaseBasis,
    parse_earnings_release,
)

__all__ = [
    "POLICY_VERSION",
    "EarningsReleaseBenchmark",
    "benchmark_earnings_releases",
    "earnings_release_dataset_inputs",
    "materialize_earnings_releases",
]

POLICY_VERSION = "earnings-releases-v2"
_QUARANTINE_FILENAME = "quarantine.json"
_FISCAL_RE = re.compile(r"^(\d{4})Q([1-4])$")
_BENCHMARK_METRICS: tuple[str, ...] = ("sales", "operating_profit")

_LOG = logging.getLogger(__name__)

_SCHEMA: dict[str, Any] = {
    "instrument_id": pl.String,
    "ticker": pl.String,
    "corp_code": pl.String,
    "rcept_no": pl.String,
    "release_kind": pl.String,
    "basis": pl.String,
    "is_correction": pl.Boolean,
    "fiscal_period": pl.String,
    "period_label": pl.String,
    "metric": pl.String,
    "span": pl.String,
    "value_krw": pl.Float64,
    "prior_year_value_krw": pl.Float64,
    "received_on": pl.Date,
    "available_at": pl.Datetime("us", "Asia/Seoul"),
    "policy_version": pl.String,
}


def _load_corp_bridge(catalog: ReceiptCatalog) -> dict[str, str]:
    from src.data.jobs.universe import read_corp_code_bridge

    try:
        mapping, _ = read_corp_code_bridge(catalog)
    except PITDataError as exc:
        raise PITDataError(f"invalid dart corp-code bridge: {exc}") from exc
    return mapping


def _iter_release_envelopes(catalog: ReceiptCatalog) -> list[dict[str, Any]]:
    """Every retained release envelope, hash-verified with ``014`` bodies skipped."""
    envelopes: list[dict[str, Any]] = []
    for blob in catalog.blobs(source=EARNINGS_RELEASE_SOURCE, usable=True):
        path = Path(blob.payload_path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise PITDataError(f"earnings-release Bronze payload is unreadable: {path}") from exc
        if hashlib.sha256(raw).hexdigest() != blob.content_hash:
            raise PITDataError(f"earnings-release Bronze hash mismatch: {path}")
        if raw.lstrip()[:2] == b"PK":
            continue
        try:
            payload = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        if "archive_b64" not in payload or "rcept_no" not in payload:
            continue
        try:
            archive = base64.b64decode(str(payload["archive_b64"]), validate=True)
        except (ValueError, TypeError):
            continue
        if b"<status>014</status>" in archive[:600]:
            continue
        envelopes.append(payload)
    return envelopes


def _release_from_envelope(payload: dict[str, Any], *, max_filing_lag_days: int) -> EarningsRelease:
    try:
        rcept_no = str(payload["rcept_no"]).strip()
        corp_code = str(payload["corp_code"]).strip()
        received_on = date.fromisoformat(str(payload["received_on"])[:10])
    except (KeyError, ValueError, TypeError) as exc:
        raise PITDataError("invalid earnings-release Bronze envelope; certification blocked") from exc
    archive = base64.b64decode(str(payload["archive_b64"]), validate=True)
    return parse_earnings_release(
        archive_bytes=archive,
        rcept_no=rcept_no,
        corp_code=corp_code,
        received_on=received_on,
        report_nm=str(payload.get("report_nm", "") or ""),
        max_filing_lag_days=max_filing_lag_days,
    )


def _resolve_available_at(received_on: date, *, calendar: SessionCalendar) -> datetime:
    for session in calendar.sessions:
        if session.astimezone(KRX_TZ).date() > received_on:
            return session
    raise PITDataError(f"no certified session open after receipt date {received_on.isoformat()}")


def earnings_release_dataset_inputs(*, bronze_earnings_releases: str, corp_code_bridge: str) -> dict[str, str]:
    """Identity inputs of the Silver earnings-release dataset (Bronze digests only)."""
    return {
        "bronze_earnings_releases": bronze_earnings_releases,
        "corp_code_bridge": corp_code_bridge,
    }


def materialize_earnings_releases(
    *,
    catalog: ReceiptCatalog,
    silver_root: Path,
    calendar: SessionCalendar,
    policy: EarningsReleasePolicy,
) -> Path:
    """Build and publish ``earnings_releases_<hash16>`` from release archives.

    Every parsed filing version becomes its own rows with its own ``available_at`` (first certified
    session strictly after the receipt date), so a later correction never rewrites what was known
    earlier. Filings that cannot be parsed without guessing are withheld in ``quarantine.json``
    next to the manifest with their parse reason.

    Raises:
        PITDataError: a Bronze payload is unreadable or fails its hash, or the calendar has no
            session after a receipt date.
    """
    if len(calendar.sessions) < 1:
        raise PITDataError("earnings releases require at least one certified session")
    bridge = _load_corp_bridge(catalog)
    try:
        from src.data.jobs.universe import read_corp_code_bridge as _read_bridge

        _, bridge_receipt_hash = _read_bridge(catalog)
    except PITDataError:  # pragma: no cover - missing bridge preview fallback
        bridge_receipt_hash = ""
    envelopes = sorted(_iter_release_envelopes(catalog), key=lambda item: str(item.get("rcept_no", "")))
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for envelope in envelopes:
        key = str(envelope.get("rcept_no", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(envelope)
    envelope_hashes = [
        hashlib.sha256(
            json.dumps(envelope, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        for envelope in unique
    ]
    parsed: list[EarningsRelease] = []
    quarantine: list[dict[str, str]] = []
    for envelope in unique:
        try:
            parsed.append(
                _release_from_envelope(envelope, max_filing_lag_days=policy.max_filing_lag_days)
            )
        except EarningsReleaseParseError as exc:
            quarantine.append(
                {
                    "rcept_no": str(envelope.get("rcept_no", "")),
                    "corp_code": str(envelope.get("corp_code", "")),
                    "received_on": str(envelope.get("received_on", "")),
                    "reason": exc.reason,
                }
            )
    rows: list[dict[str, Any]] = []
    unmapped = 0
    corrections = 0
    for release in sorted(parsed, key=lambda item: (item.received_on, item.rcept_no)):
        ticker = bridge.get(release.corp_code)
        if ticker is None:
            unmapped += 1
            continue
        if release.is_correction:
            corrections += 1
        available_at = _resolve_available_at(release.received_on, calendar=calendar)
        fiscal_period = f"{release.fiscal_year}Q{release.fiscal_quarter}"
        for value in release.values:
            if value.current_krw is None and value.prior_year_krw is None:
                continue
            rows.append(
                {
                    "instrument_id": f"KRX:{ticker}",
                    "ticker": ticker,
                    "corp_code": release.corp_code,
                    "rcept_no": release.rcept_no,
                    "release_kind": release.kind.value,
                    "basis": release.basis.value,
                    "is_correction": release.is_correction,
                    "fiscal_period": fiscal_period,
                    "period_label": release.period_label,
                    "metric": value.metric,
                    "span": value.span.value,
                    "value_krw": value.current_krw,
                    "prior_year_value_krw": value.prior_year_krw,
                    "received_on": release.received_on,
                    "available_at": available_at,
                    "policy_version": POLICY_VERSION,
                }
            )
    frame = (
        pl.DataFrame(rows, schema=_SCHEMA).sort(
            ["ticker", "fiscal_period", "available_at", "rcept_no", "metric", "span"]
        )
        if rows
        else pl.DataFrame([], schema=_SCHEMA)
    )
    identity = DatasetIdentity(
        kind="earnings_releases",
        layer=DatasetLayer.SILVER,
        policy_version=POLICY_VERSION,
        inputs=earnings_release_dataset_inputs(
            bronze_earnings_releases=dataset_digest(envelope_hashes),
            corp_code_bridge=dataset_digest([bridge_receipt_hash]),
        ),
        params={
            "calendar_digest": hashlib.sha256(
                "\n".join(session.astimezone(KRX_TZ).date().isoformat() for session in calendar.sessions).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "max_filing_lag_days": policy.max_filing_lag_days,
        },
    )
    by_reason: dict[str, int] = {}
    for entry in quarantine:
        by_reason[entry["reason"]] = by_reason.get(entry["reason"], 0) + 1
    published = publish_dataset(
        layer_root=Path(silver_root),
        identity=identity,
        partitions={"part-00000.parquet": frame},
        details={
            "filings": len(unique),
            "parsed_filings": len(parsed),
            "quarantined_filings": len(quarantine),
            "quarantined_by_reason": dict(sorted(by_reason.items())),
            "unmapped_filings": unmapped,
            "corrections": corrections,
            "rows": len(rows),
            "corp_code_bridge": bridge_receipt_hash,
        },
    )
    (published.path / _QUARANTINE_FILENAME).write_text(
        json.dumps(sorted(quarantine, key=lambda item: item["rcept_no"]), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _LOG.info(
        "[DATA] stage=earnings_releases filings=%d parsed=%d quarantined=%d unmapped=%d corrections=%d rows=%d",
        len(unique),
        len(parsed),
        len(quarantine),
        unmapped,
        corrections,
        len(rows),
    )
    return published.path


@dataclass(frozen=True, slots=True)
class EarningsReleaseBenchmark:
    """Agreement of preliminary numbers with the later periodic filing of the same period."""

    metric: str
    matched: int
    within_1pct: float
    within_5pct: float
    median_abs_rel_error: float


def _fact_lookup(facts_path: Path) -> Mapping[tuple[str, str, str], float]:
    """Map ``(instrument_id, fiscal_period, fact)`` to the latest consolidated fact value.

    Facts carry the bare KRX ticker (``company_id`` is the ticker too), so the key is rebuilt as
    ``KRX:<ticker>`` to match release ``instrument_id`` values.
    """
    frame = read_dataset(
        Path(facts_path), columns=["ticker", "fiscal_period", "fact", "value", "consolidated"]
    ).collect()
    lookup: dict[tuple[str, str, str], float] = {}
    try:
        for row in frame.iter_rows(named=True):
            if row["consolidated"] is not True:
                continue
            ticker = row["ticker"]
            period = row["fiscal_period"]
            fact = row["fact"]
            value = row["value"]
            if not isinstance(ticker, str) or not isinstance(period, str) or not isinstance(fact, str):
                raise PITDataError(f"invalid financial fact row for earnings benchmark: {row!r}")
            if value is None:
                continue
            lookup[(f"KRX:{ticker.strip()}", period, str(fact))] = float(value)
    except (TypeError, ValueError) as exc:
        raise PITDataError(f"invalid financial facts for earnings benchmark: {exc}") from exc
    return lookup


def benchmark_earnings_releases(
    *, releases_path: Path, facts_path: Path
) -> tuple[EarningsReleaseBenchmark, ...]:
    """Compare consolidated release values with Silver financial facts for sales and operating profit.

    Quarter span is compared with Q1-Q3 fact values, cumulative/annual span with Q4 facts (annual).
    Report only: preliminary numbers legitimately differ from audited ones, so this never gates a build.
    """
    releases = read_dataset(
        Path(releases_path),
        columns=["instrument_id", "fiscal_period", "metric", "span", "value_krw", "basis"],
    ).collect()
    facts = _fact_lookup(Path(facts_path))
    errors: dict[str, list[float]] = {metric: [] for metric in _BENCHMARK_METRICS}
    try:
        for row in releases.iter_rows(named=True):
            if row["basis"] != ReleaseBasis.CONSOLIDATED.value:
                continue
            metric = str(row["metric"] or "")
            if metric not in errors:
                continue
            value = row["value_krw"]
            if value is None:
                continue
            period = str(row["fiscal_period"] or "")
            match = _FISCAL_RE.fullmatch(period)
            if match is None:
                continue
            quarter = int(match.group(2))
            span = str(row["span"] or "")
            if span == "quarter":
                if quarter == 4:
                    continue
            elif span in {"cumulative", "annual"}:
                if quarter != 4:
                    continue
            else:
                continue
            fact_value = facts.get((str(row["instrument_id"] or ""), period, metric))
            if fact_value is None:
                continue
            if fact_value == 0:
                errors[metric].append(0.0 if float(value) == 0 else float("inf"))
            else:
                errors[metric].append(abs(float(value) - fact_value) / abs(fact_value))
    except (TypeError, ValueError) as exc:
        raise PITDataError(f"invalid earnings releases for benchmark: {exc}") from exc
    out: list[EarningsReleaseBenchmark] = []
    for metric in _BENCHMARK_METRICS:
        values = sorted(errors[metric])
        matched = len(values)
        finite = [item for item in values if item != float("inf")]
        out.append(
            EarningsReleaseBenchmark(
                metric=metric,
                matched=matched,
                within_1pct=(sum(1 for item in values if item <= 0.01) / matched) if matched else 0.0,
                within_5pct=(sum(1 for item in values if item <= 0.05) / matched) if matched else 0.0,
                median_abs_rel_error=statistics.median(finite) if finite else 0.0,
            )
        )
    return tuple(out)
