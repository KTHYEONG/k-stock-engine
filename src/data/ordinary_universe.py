"""Point-in-time KRX ordinary-share universe from dated Bronze master pages."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import polars as pl

from src.core.time import KRX_TZ
from src.data.schemas import BronzeReceipt, EvidenceKind, PITDataError

POLICY_VERSION = "krx-ordinary-equity-v1"
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
    """Verify raw bytes and classify every ticker in one dated KRX master page.

    `available_at` is the snapshot day's close. Retrieval time remains
    provenance and cannot move a historical observation into the future.
    """
    if receipt.kind is not EvidenceKind.SECURITY_MASTER:
        raise PITDataError("expected security_master Bronze receipt")
    raw = receipt.payload_path.read_bytes()
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
                listed_on = date.fromisoformat(str(listed_raw)[:10]) if "-" in str(listed_raw) else datetime.strptime(str(listed_raw), "%Y%m%d").date()
            except ValueError as exc:
                raise PITDataError(f"invalid listing date for {ticker}") from exc
            if listed_on > session:
                eligible, reason = False, "not_listed_yet"
        rows.append({
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
            "policy_version": POLICY_VERSION,
        })
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


def dated_master_receipts(bronze_root: Path, *, sessions: Iterable[date]) -> tuple[BronzeReceipt, ...]:
    """Select requested dated pages using receipt metadata before opening payloads."""
    requested = frozenset(sessions)
    receipts: list[BronzeReceipt] = []
    for metadata_path in sorted((Path(bronze_root) / EvidenceKind.SECURITY_MASTER.value).glob("*/receipt.json")):
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            label = str(meta["source_path"])
            match = _DATED_SOURCE.fullmatch(label)
            if match is None or date.fromisoformat(match.group(1)) not in requested:
                continue
            if meta["kind"] != EvidenceKind.SECURITY_MASTER.value:
                raise PITDataError("security_master receipt kind mismatch")
            content_hash = str(meta["content_hash"])
            if content_hash != metadata_path.parent.name:
                raise PITDataError("security_master receipt path/hash mismatch")
            receipts.append(BronzeReceipt(
                kind=EvidenceKind.SECURITY_MASTER,
                content_hash=content_hash,
                source_path=label,
                retrieved_at=datetime.fromisoformat(meta["retrieved_at"]),
                ingested_at=datetime.fromisoformat(meta["ingested_at"]),
                payload_path=metadata_path.parent / "payload.json",
                metadata_path=metadata_path,
            ))
        except PITDataError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PITDataError(f"invalid security_master receipt: {metadata_path}") from exc
    return tuple(sorted(receipts, key=lambda receipt: receipt.source_path))


def write_ordinary_universe_silver(
    snapshots: Iterable[OrdinaryUniverseSnapshot], *, root: Path
) -> Path:
    """Write ordered snapshots one day at a time with bounded working memory."""
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ordinary-universe-", dir=root))
    try:
        partitions: list[dict[str, Any]] = []
        identity: list[str] = []
        previous_day: date | None = None
        for snapshot in snapshots:
            if previous_day is not None and snapshot.session <= previous_day:
                raise PITDataError("ordinary universe requires unique ascending dates")
            previous_day = snapshot.session
            identity.append(f"{snapshot.session.isoformat()}:{snapshot.source_hash}")
            rel = Path(f"session={snapshot.session.isoformat()}") / "part.parquet"
            path = staging / rel
            path.parent.mkdir(parents=True)
            snapshot.rows.write_parquet(path)
            partitions.append({
                "session": snapshot.session.isoformat(),
                "path": str(rel),
                "source_hash": snapshot.source_hash,
                "row_count": snapshot.rows.height,
                "eligible_count": len(snapshot.eligible_tickers),
                "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        if not partitions:
            raise PITDataError("ordinary universe requires dated snapshots")
        dataset_hash = hashlib.sha256((POLICY_VERSION + "\n" + "\n".join(identity)).encode()).hexdigest()
        target = Path(root) / f"ordinary_universe_{dataset_hash[:16]}"
        if target.exists():
            raise PITDataError(f"ordinary universe dataset already exists: {target}")
        manifest = {
            "dataset_id": target.name,
            "policy_version": POLICY_VERSION,
            "dataset_hash": dataset_hash,
            "partitions": partitions,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def materialize_ordinary_universe_from_bronze(
    *, bronze_root: Path, sessions: Iterable[date], silver_root: Path
) -> Path:
    """Select exact dated Bronze pages and stream a source-bound Silver dataset."""
    requested = frozenset(sessions)
    if not requested:
        raise PITDataError("ordinary universe requires requested sessions")
    receipts = dated_master_receipts(bronze_root, sessions=requested)
    labels = [date.fromisoformat(receipt.source_path.rsplit(":", 1)[1]) for receipt in receipts]
    if len(labels) != len(set(labels)):
        raise PITDataError("ambiguous security_master snapshot date")
    missing = requested - set(labels)
    if missing:
        raise PITDataError(f"missing security_master snapshot for {min(missing)}")
    return write_ordinary_universe_silver(
        (ordinary_universe_snapshot(receipt) for receipt in receipts), root=silver_root
    )
