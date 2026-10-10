"""Point-in-time KRX ordinary-share universe from dated Bronze master pages."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import polars as pl

from src.core.pit import BronzeReceipt, EvidenceKind, PITDataError
from src.core.time import KRX_TZ
from src.data.datasets import (
    DatasetIdentity,
    DatasetLayer,
    dataset_digest,
    file_sha256,
    load_manifest,
    publish_dataset,
)
from src.data.payload_store import read_payload
from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog

REVISION = "krx-ordinary-equity-v1"
_MARKETS = frozenset({"KOSPI", "KOSDAQ"})
_DATED_SOURCE = re.compile(r"^KRX:historical-master:(\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True, slots=True)
class OrdinaryUniverseSnapshot:
    session: date
    source_hash: str
    rows: pl.DataFrame

    @property
    def eligible_tickers(self) -> frozenset[str]:
        return frozenset(self.rows.filter(pl.col("eligible"))["ticker"].to_list())


def classify_krx_master_row(record: dict[str, Any]) -> tuple[bool, str]:
    """Return strict eligibility and a stable exclusion reason for one KRX row."""

    kind = str(record.get("KIND_STKCERT_TP_NM") or "").strip()
    security_group = str(record.get("SECUGRP_NM") or "").strip()
    market = str(record.get("MKT_TP_NM") or "").strip()
    if not kind:
        return False, "unknown_share_kind"
    if kind != "보통주":
        return False, "non_ordinary_share"
    if not security_group:
        return False, "unknown_security_group"
    if security_group != "주권":
        return False, "non_equity_security_group"
    if not market:
        return False, "unknown_market"
    if market not in _MARKETS:
        return False, "excluded_market"
    section = str(record.get("SECT_TP_NM") or "").upper()
    names = " ".join(str(record.get(key) or "") for key in ("ISU_NM", "ISU_ABBRV", "ISU_ENG_NM")).upper()
    if "SPAC" in section or "SPAC" in names or "스팩" in names or "기업인수목적" in names:
        return False, "spac"
    return True, "eligible"


def _snapshot_date(payload: dict[str, Any]) -> date:
    dates: set[date] = set()
    for key in ("as_of", "session"):
        value = payload.get(key)
        if value in (None, ""):
            continue
        try:
            dates.add(date.fromisoformat(str(value)[:10]))
        except ValueError as exc:
            raise PITDataError(f"invalid KRX master {key}") from exc
    if len(dates) != 1:
        raise PITDataError("KRX master requires one unambiguous snapshot date")
    return dates.pop()


def ordinary_universe_snapshot(receipt: BronzeReceipt) -> OrdinaryUniverseSnapshot:
    """Verify raw bytes and classify every ticker in one dated KRX page."""

    if receipt.kind is not EvidenceKind.SECURITY_MASTER:
        raise PITDataError("expected security_master Bronze receipt")
    try:
        raw = read_payload(Path(receipt.payload_path))
    except (OSError, PITDataError) as exc:
        raise PITDataError("security_master Bronze payload is unreadable") from exc
    if hashlib.sha256(raw).hexdigest() != receipt.content_hash:
        raise PITDataError("security_master Bronze hash mismatch")
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise PITDataError("invalid security_master Bronze JSON") from exc
    if not isinstance(payload, dict):
        raise PITDataError("invalid security_master Bronze root")
    session = _snapshot_date(payload)
    label_match = _DATED_SOURCE.fullmatch(receipt.source_path)
    if label_match is not None and date.fromisoformat(label_match.group(1)) != session:
        raise PITDataError("security_master receipt date conflicts with payload date")
    records = payload.get("records")
    if not isinstance(records, list):
        raise PITDataError("security_master records must be a list")
    available_at = datetime.combine(session, time(15, 30), tzinfo=KRX_TZ)
    rows: list[dict[str, Any]] = []
    seen_tickers: set[str] = set()
    seen_isins: dict[str, str] = {}
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise PITDataError("security_master record must be an object")
        ticker = str(raw_record.get("ISU_SRT_CD") or "").strip()
        isin = str(raw_record.get("ISU_CD") or "").strip()
        if not ticker or not isin:
            raise PITDataError("security_master record lacks ticker or ISIN")
        if ticker in seen_tickers:
            raise PITDataError(f"duplicate security_master ticker for {session}: {ticker}")
        if isin in seen_isins and seen_isins[isin] != ticker:
            raise PITDataError(f"conflicting security_master ISIN for {session}: {isin}")
        seen_tickers.add(ticker)
        seen_isins[isin] = ticker
        eligible, reason = classify_krx_master_row(raw_record)
        listed_raw = raw_record.get("LIST_DD")
        if eligible and listed_raw not in (None, ""):
            try:
                listed_on = (
                    date.fromisoformat(str(listed_raw)[:10])
                    if "-" in str(listed_raw)
                    else datetime.strptime(str(listed_raw), "%Y%m%d").date()
                )
            except ValueError as exc:
                raise PITDataError(f"invalid listing date for {ticker}") from exc
            if listed_on > session:
                eligible, reason = False, "not_listed_yet"
        rows.append(
            {
                "session": session,
                "instrument_id": f"KRX:{ticker}",
                "ticker": ticker,
                "source_security_id": isin,
                "market": str(raw_record.get("MKT_TP_NM") or ""),
                "share_kind": str(raw_record.get("KIND_STKCERT_TP_NM") or ""),
                "security_group": str(raw_record.get("SECUGRP_NM") or ""),
                "eligible": eligible,
                "exclusion_reason": reason,
                "available_at": available_at,
                "source_hash": receipt.content_hash,
                "policy_version": REVISION,
            }
        )
    if not rows:
        raise PITDataError(f"empty security_master snapshot for {session}")
    return OrdinaryUniverseSnapshot(session, receipt.content_hash, pl.DataFrame(rows).sort("ticker"))


def build_ordinary_universe(
    receipts: Iterable[BronzeReceipt], *, sessions: Iterable[date]
) -> tuple[OrdinaryUniverseSnapshot, ...]:
    """Build exact requested sessions; missing or duplicate dated pages fail closed."""

    requested = frozenset(sessions)
    if not requested:
        raise PITDataError("ordinary universe requires requested sessions")
    by_day: dict[date, OrdinaryUniverseSnapshot] = {}
    for receipt in receipts:
        snapshot = ordinary_universe_snapshot(receipt)
        if snapshot.session not in requested:
            continue
        if snapshot.session in by_day:
            raise PITDataError(f"ambiguous security_master snapshot for {snapshot.session}")
        by_day[snapshot.session] = snapshot
    missing = requested - by_day.keys()
    if missing:
        raise PITDataError(f"missing security_master snapshot for {min(missing)}")
    return tuple(by_day[day] for day in sorted(requested))


def catalog_master_receipts(
    catalog: ReceiptCatalog, *, sessions: Iterable[date]
) -> tuple[BronzeReceipt, ...]:
    """Resolve one successful, dated master receipt per requested session."""

    requested = frozenset(sessions)
    if not requested:
        raise PITDataError("ordinary universe requires requested sessions")
    entries = catalog.latest(
        source="krx_security_master",
        natural_keys={session.isoformat() for session in requested},
    )
    receipts: list[BronzeReceipt] = []
    for session in sorted(requested):
        entry = entries.get(session.isoformat())
        if entry is None or entry.status is not EvidenceStatus.SUCCESS:
            raise PITDataError(f"missing security_master snapshot for {session}")
        if entry.as_of != session:
            raise PITDataError(f"security_master catalog date conflicts for {session}")
        payload_path = Path(entry.payload_path)
        receipts.append(
            BronzeReceipt(
                kind=EvidenceKind.SECURITY_MASTER,
                content_hash=entry.content_hash,
                source_path=f"KRX:historical-master:{session.isoformat()}",
                retrieved_at=entry.retrieved_at,
                ingested_at=entry.retrieved_at,
                payload_path=payload_path,
                metadata_path=payload_path.parent / "receipt.json",
            )
        )
    return tuple(receipts)


def write_ordinary_universe_silver(
    snapshots: Iterable[OrdinaryUniverseSnapshot], *, root: Path
) -> Path:
    """Publish ordered ordinary-universe snapshots through the dataset contract."""

    ordered = tuple(snapshots)
    if not ordered:
        raise PITDataError("ordinary universe requires dated snapshots")
    partitions: dict[str, pl.DataFrame] = {}
    details: list[dict[str, object]] = []
    source_hashes: list[str] = []
    previous_day: date | None = None
    for snapshot in ordered:
        if previous_day is not None and snapshot.session <= previous_day:
            raise PITDataError("ordinary universe requires unique ascending dates")
        previous_day = snapshot.session
        source_hashes.append(snapshot.source_hash)
        relative_path = f"session={snapshot.session.isoformat()}/part.parquet"
        partitions[relative_path] = snapshot.rows.sort(["session", "instrument_id"])
        details.append(
            {
                "session": snapshot.session.isoformat(),
                "source_hash": snapshot.source_hash,
                "rows": snapshot.rows.height,
                "eligible_rows": len(snapshot.eligible_tickers),
            }
        )
    identity = DatasetIdentity(
        kind="ordinary_universe",
        layer=DatasetLayer.SILVER,
        policy_version=REVISION,
        inputs={"bronze_master": dataset_digest(source_hashes)},
        params={
            "calendar_digest": hashlib.sha256(
                "\n".join(snapshot.session.isoformat() for snapshot in ordered).encode("utf-8")
            ).hexdigest()
        },
    )
    published = publish_dataset(
        layer_root=Path(root),
        identity=identity,
        partitions=partitions,
        details={"partitions": details, "sessions": len(ordered)},
    )
    return published.path


def catalog_master_sessions(catalog: ReceiptCatalog) -> tuple[date, ...]:
    """Return dated successful security-master sessions that were actual trading days.

    KRX serves a master snapshot on any date, including exchange holidays, so a
    snapshot alone does not prove a session. A date on which the daily-market
    endpoint answered "no data" (status ``empty``) is a closure confirmed by the
    exchange itself, and is excluded; ``exchange_calendars`` can lag KRX
    calendar changes (for example a newly designated holiday).
    """

    closed = {
        entry.as_of
        for entry in catalog.entries(source="krx_daily_market")
        if entry.status is EvidenceStatus.EMPTY and entry.as_of is not None
    }
    sessions: set[date] = set()
    for entry in catalog.entries(source="krx_security_master"):
        if entry.status is EvidenceStatus.SUCCESS and entry.as_of is not None and entry.as_of not in closed:
            sessions.add(entry.as_of)
    if not sessions:
        raise PITDataError("no certified security_master snapshots found")
    return tuple(sorted(sessions))


def materialize_ordinary_universe_from_catalog(
    *, catalog: ReceiptCatalog, sessions: Iterable[date], silver_root: Path, reuse_from: Path | None = None
) -> Path:
    """Publish the ordinary-universe Silver; unchanged sessions may be copied from the dataset at reuse_from."""
    requested = tuple(sessions)
    receipts = catalog_master_receipts(catalog, sessions=requested)
    by_session = dict(zip(sorted(set(requested)), receipts, strict=True))
    current = {session: receipt.content_hash for session, receipt in by_session.items()}
    reused = _reusable_session_frames(reuse_from, current)
    ordered: list[OrdinaryUniverseSnapshot] = []
    for session in sorted(by_session):
        frame = reused.get(session)
        if frame is not None:
            ordered.append(OrdinaryUniverseSnapshot(session, current[session], frame))
        else:
            ordered.append(ordinary_universe_snapshot(by_session[session]))
    return write_ordinary_universe_silver(ordered, root=silver_root)


def _reusable_session_frames(reuse_from: Path | None, current: dict[date, str]) -> dict[date, pl.DataFrame]:
    """Prior-session frames whose manifest entry verifies and whose source is unchanged."""
    if reuse_from is None:
        return {}
    try:
        manifest = load_manifest(Path(reuse_from))
    except PITDataError:
        return {}
    if manifest.kind != "ordinary_universe" or manifest.policy_version != REVISION:
        return {}
    raw_partitions = manifest.details.get("partitions")
    if not isinstance(raw_partitions, list):
        return {}
    recorded: dict[str, str] = {}
    for item in raw_partitions:
        if not isinstance(item, dict) or not isinstance(item.get("session"), str) or not isinstance(item.get("source_hash"), str):
            return {}
        recorded[str(item["session"])] = str(item["source_hash"])
    expected_sha = {partition.path: partition.sha256 for partition in manifest.partitions}
    reused: dict[date, pl.DataFrame] = {}
    for session, source_hash in current.items():
        if recorded.get(session.isoformat()) != source_hash:
            continue
        relative_path = f"session={session.isoformat()}/part.parquet"
        if expected_sha.get(relative_path) is None:
            continue
        path = Path(reuse_from) / relative_path
        try:
            if file_sha256(path) != expected_sha[relative_path]:
                continue
            reused[session] = pl.read_parquet(path)
        except Exception:  # noqa: S112 - an unverifiable prior partition is rebuilt from payloads
            continue
    return reused
