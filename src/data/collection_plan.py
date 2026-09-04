"""Historical collection plan, checkpoints, and readiness gates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.data.schemas import EvidenceKind, PITDataError
from src.strategy.universe import UniverseDecision

PLAN_ARTIFACT_DIR = Path("data/artifacts/collection-plans")
CHECKPOINT_ARTIFACT_DIR = Path("data/artifacts/collection-checkpoints")


def _redact_message(message: str) -> str:
    lowered = message.lower()
    for token in ("app_key", "appkey", "app_secret", "appsecret", "token", "authorization", "bearer", "crtfc_key"):
        if token in lowered:
            raise PITDataError("collection failed; see provider status")
    return message


@dataclass(frozen=True, slots=True)
class PlanChunk:
    chunk_id: str
    symbol: str
    sessions: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class HistoricalCollectionPlan:
    plan_id: str
    coverage_start: date
    coverage_end: date
    chunk_size: int
    chunks: tuple[PlanChunk, ...]
    content_hash: str = ""


def _canonical_universe(universe: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in universe or ():
        if not isinstance(entry, dict):
            raise PITDataError("universe entry must be a mapping")
        symbol = str(entry.get("symbol") or "").strip()
        if not symbol:
            raise PITDataError("universe entry missing symbol")
        tradable_from = entry.get("tradable_from")
        tradable_to = entry.get("tradable_to")
        items.append(
            {
                "symbol": symbol,
                "is_common_stock": bool(entry.get("is_common_stock", True)),
                "tradable_from": tradable_from.isoformat() if isinstance(tradable_from, date) else None,
                "tradable_to": tradable_to.isoformat() if isinstance(tradable_to, date) else None,
            }
        )
    return sorted(items, key=lambda e: e["symbol"])


def build_historical_collection_plan(
    sessions: Any = (),
    universe: Any = (),
    start: date | None = None,
    end: date | None = None,
    chunk_size: int = 20,
    *,
    artifact_root: Path | str | None = None,
    input_receipt_digest: str | None = None,
    coverage: tuple[date, date] | None = None,
    symbols: Any | None = None,
) -> HistoricalCollectionPlan:
    if coverage is not None:
        start, end = coverage[0], coverage[1]
    if symbols is not None:
        universe = symbols
    if start is None or end is None:
        raise PITDataError("coverage start and end are required")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise PITDataError("chunk_size must be a positive integer")
    if start > end:
        raise PITDataError("coverage_start must not be after coverage_end")
    raw_sessions = list(sessions or ())
    for session in raw_sessions:
        if not isinstance(session, date):
            raise PITDataError("sessions must contain dates only")
    in_range = sorted(s for s in raw_sessions if start <= s <= end)
    if not in_range:
        raise PITDataError("no sessions inside declared coverage")
    if universe is None or (isinstance(universe, (list, tuple)) and not universe):
        raise PITDataError("universe must list at least one symbol")
    canonical = _canonical_universe(universe)
    common = [entry for entry in universe if isinstance(entry, dict) and entry.get("is_common_stock", True)]
    if not common:
        raise PITDataError("universe must include at least one common stock")
    digest = hashlib.sha256()
    digest.update(str(input_receipt_digest or "legacy-validated").encode("utf-8"))
    digest.update(b"\x00")
    digest.update(start.isoformat().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(end.isoformat().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(chunk_size).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(json.dumps(canonical, sort_keys=True).encode("utf-8"))
    plan_id = f"plan-{digest.hexdigest()[:16]}"
    content_hash = digest.hexdigest()
    chunks: list[PlanChunk] = []
    for entry in sorted(common, key=lambda e: str(e.get("symbol"))):
        symbol = str(entry.get("symbol")).strip()
        tradable_from = entry.get("tradable_from")
        tradable_to = entry.get("tradable_to")
        eligible = tuple(
            s for s in in_range if (tradable_from is None or s >= tradable_from) and (tradable_to is None or s <= tradable_to)
        )
        for index in range(0, len(eligible), chunk_size):
            window = eligible[index : index + chunk_size]
            if not window:
                continue
            chunk_id = f"{plan_id}:{symbol}:{index // chunk_size:04d}"
            chunks.append(PlanChunk(chunk_id=chunk_id, symbol=symbol, sessions=tuple(window)))
    plan = HistoricalCollectionPlan(
        plan_id=plan_id,
        coverage_start=start,
        coverage_end=end,
        chunk_size=chunk_size,
        chunks=tuple(chunks),
        content_hash=content_hash,
    )
    root = Path(artifact_root) if artifact_root is not None else PLAN_ARTIFACT_DIR
    try:
        root.mkdir(parents=True, exist_ok=True)
        receipt_path = root / f"{plan_id}.json"
        receipt_path.write_text(
            json.dumps(
                {
                    "plan_id": plan_id,
                    "content_hash": content_hash,
                    "coverage_start": start.isoformat(),
                    "coverage_end": end.isoformat(),
                    "chunk_size": chunk_size,
                    "input_receipt_digest": input_receipt_digest or "legacy-validated",
                    "universe": canonical,
                    "chunks": [
                        {"chunk_id": c.chunk_id, "symbol": c.symbol, "sessions": [s.isoformat() for s in c.sessions]}
                        for c in chunks
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise PITDataError(_redact_message(f"plan receipt write failed: {type(exc).__name__}")) from exc
    return plan


def _parse_plan_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        pass
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        try:
            return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
        except ValueError:
            return None
    return None


def _normalize_master_record(record: dict[str, Any]) -> dict[str, Any] | None:
    symbol = str(
        record.get("ISU_SRT_CD")
        or record.get("isu_cd")
        or record.get("ISU_CD")
        or record.get("source_identifier")
        or record.get("symbol")
        or record.get("ticker")
        or ""
    ).strip()
    if not symbol:
        return None
    kind_name = record.get("KIND_STKCERT_TP_NM")
    if kind_name is not None:
        is_common = str(kind_name).strip() == "보통주"
    elif "is_common_stock" in record:
        is_common = bool(record.get("is_common_stock"))
    elif record.get("share_class") is not None:
        is_common = str(record.get("share_class")).strip() == "common"
    else:
        is_common = True
    tradable_from = _parse_plan_date(
        record.get("LIST_DD")
        or record.get("tradable_from")
        or record.get("listing_date")
        or record.get("listed_from")
    )
    tradable_to = _parse_plan_date(
        record.get("delisting_date") or record.get("delisted_on") or record.get("tradable_to")
    )
    return {"symbol": symbol, "is_common_stock": is_common, "tradable_from": tradable_from, "tradable_to": tradable_to}


def build_historical_collection_plan_from_bronze(
    *,
    bronze_root: Path | str,
    start: date,
    end: date,
    chunk_size: int = 20,
    symbols: tuple[str, ...] | None = None,
    artifact_root: Path | str | None = None,
) -> HistoricalCollectionPlan:
    """Build a plan only from the retained calendar and historical master evidence."""
    from src.data.bronze_aggregation import discover_verified_bronze_receipts

    root = Path(bronze_root)
    if start > end:
        raise PITDataError("coverage_start must not be after coverage_end")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise PITDataError("chunk_size must be a positive integer")
    wanted = frozenset(symbols or ())
    grouped = discover_verified_bronze_receipts(bronze_root=root)
    calendars: list[tuple[date, Any]] = []
    masters: list[tuple[Any, list[dict[str, Any]]]] = []
    input_hashes: list[str] = []
    if grouped.get(EvidenceKind.CALENDAR) or grouped.get(EvidenceKind.SECURITY_MASTER):
        from src.data.schemas import EvidenceKind as _Kind

        for receipt in grouped.get(_Kind.CALENDAR, ()):
            try:
                payload = json.loads(receipt.payload_path.read_bytes())
            except (OSError, ValueError) as exc:
                raise PITDataError("retained Bronze plan inputs are unreadable") from exc
            raw = payload.get("sessions") if isinstance(payload, dict) else None
            if not isinstance(raw, list):
                raise PITDataError("retained Bronze plan inputs have invalid schema")
            for value in raw:
                parsed = _parse_plan_date(value)
                if parsed is None:
                    raise PITDataError("retained Bronze plan inputs have invalid schema")
                calendars.append((parsed, receipt))
            input_hashes.append(f"calendar:{receipt.content_hash}")
        for receipt in grouped.get(_Kind.SECURITY_MASTER, ()):
            try:
                payload = json.loads(receipt.payload_path.read_bytes())
            except (OSError, ValueError) as exc:
                raise PITDataError("retained Bronze plan inputs are unreadable") from exc
            raw_records = payload.get("records") if isinstance(payload, dict) else None
            if not isinstance(raw_records, list):
                raise PITDataError("retained Bronze plan inputs have invalid schema")
            normalized: list[dict[str, Any]] = []
            for record in raw_records:
                if not isinstance(record, dict):
                    continue
                entry = _normalize_master_record(record)
                if entry is not None:
                    normalized.append(entry)
            masters.append((receipt, normalized))
            input_hashes.append(f"security_master:{receipt.content_hash}")
    else:
        try:
            calendar_paths = sorted((root / "calendar").glob("*/payload.json"))
            master_paths = sorted((root / "security_master").glob("*/payload.json"))
            if not calendar_paths or not master_paths:
                raise PITDataError("missing retained calendar and security master Bronze receipts")
            for path in calendar_paths:
                calendar = json.loads(path.read_bytes())
                raw_sessions = calendar.get("sessions") if isinstance(calendar, dict) else None
                if not isinstance(raw_sessions, list):
                    raise PITDataError("retained Bronze plan inputs have invalid schema")
                for value in raw_sessions:
                    parsed = _parse_plan_date(value)
                    if parsed is None:
                        raise PITDataError("retained Bronze plan inputs have invalid schema")
                    calendars.append((parsed, None))
            for path in master_paths:
                master = json.loads(path.read_bytes())
                raw_records = master.get("records") if isinstance(master, dict) else None
                if not isinstance(raw_records, list):
                    raise PITDataError("retained Bronze plan inputs have invalid schema")
                normalized2: list[dict[str, Any]] = []
                for record in raw_records:
                    if not isinstance(record, dict):
                        continue
                    entry = _normalize_master_record(record)
                    if entry is not None:
                        normalized2.append(entry)
                masters.append((None, normalized2))
            digest_legacy = hashlib.sha256()
            for path in [*calendar_paths, *master_paths]:
                digest_legacy.update(path.read_bytes())
                digest_legacy.update(b"\x00")
            legacy_digest = digest_legacy.hexdigest()
        except (OSError, ValueError) as exc:
            if isinstance(exc, PITDataError):
                raise
            raise PITDataError("retained Bronze plan inputs are unreadable") from exc
        if not calendars or not masters:
            raise PITDataError("missing retained calendar and security master Bronze receipts")
        all_sessions = sorted({day for day, _ in calendars if start <= day <= end})
        if not all_sessions:
            raise PITDataError("no sessions inside declared coverage")
        universe_legacy: dict[str, dict[str, Any]] = {}
        for _, entries in masters:
            for entry in entries:
                if not entry["is_common_stock"]:
                    continue
                symbol = entry["symbol"]
                if wanted and symbol not in wanted:
                    continue
                universe_legacy[symbol] = entry
        if wanted and set(universe_legacy) != wanted:
            missing = sorted(wanted - set(universe_legacy))
            raise PITDataError(f"requested symbols absent from retained security master: {','.join(missing)}")
        if not universe_legacy:
            raise PITDataError("PIT universe has no eligible symbols")
        return build_historical_collection_plan(
            sessions=tuple(all_sessions),
            universe=tuple(universe_legacy.values()),
            start=start,
            end=end,
            chunk_size=chunk_size,
            artifact_root=artifact_root,
            input_receipt_digest=legacy_digest,
        )
    if not calendars or not masters:
        raise PITDataError("missing retained calendar and security master Bronze receipts")
    all_sessions_sorted = sorted({day for day, _ in calendars if start <= day <= end})
    if not all_sessions_sorted:
        raise PITDataError("no sessions inside declared coverage")
    eligible_by_symbol: dict[str, list[date]] = {}
    for day in all_sessions_sorted:
        for receipt, entries in masters:
            retrieved_day = receipt.retrieved_at.date() if receipt is not None else date.min
            if retrieved_day > day:
                continue
            for entry in entries:
                if not entry["is_common_stock"]:
                    continue
                symbol = entry["symbol"]
                if wanted and symbol not in wanted:
                    continue
                tradable_from = entry["tradable_from"]
                tradable_to = entry["tradable_to"]
                if tradable_from is not None and day < tradable_from:
                    continue
                if tradable_to is not None and day > tradable_to:
                    continue
                bucket = eligible_by_symbol.setdefault(symbol, [])
                if not bucket or bucket[-1] != day:
                    bucket.append(day)
    if wanted:
        missing_wanted = sorted(s for s in wanted if not eligible_by_symbol.get(s))
        if missing_wanted:
            raise PITDataError(f"requested symbols absent from retained security master: {','.join(missing_wanted)}")
    if not eligible_by_symbol:
        raise PITDataError("PIT universe has no eligible symbols")
    universe: list[dict[str, Any]] = []
    for symbol in sorted(eligible_by_symbol):
        days = eligible_by_symbol[symbol]
        universe.append(
            {
                "symbol": symbol,
                "is_common_stock": True,
                "tradable_from": min(days),
                "tradable_to": max(days),
            }
        )
    digest = hashlib.sha256()
    for token in sorted(input_hashes):
        digest.update(token.encode("utf-8"))
        digest.update(b"\x00")
    digest.update(start.isoformat().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(end.isoformat().encode("utf-8"))
    input_digest = digest.hexdigest()
    base = build_historical_collection_plan(
        sessions=tuple(all_sessions_sorted),
        universe=tuple(universe),
        start=start,
        end=end,
        chunk_size=chunk_size,
        artifact_root=artifact_root,
        input_receipt_digest=input_digest,
    )
    filtered: list[PlanChunk] = []
    for chunk in base.chunks:
        allowed = tuple(s for s in chunk.sessions if s in eligible_by_symbol.get(chunk.symbol, ()))
        if allowed:
            filtered.append(PlanChunk(chunk_id=chunk.chunk_id, symbol=chunk.symbol, sessions=allowed))
    if not filtered:
        raise PITDataError("PIT universe has no eligible symbols")
    return HistoricalCollectionPlan(
        plan_id=base.plan_id,
        coverage_start=base.coverage_start,
        coverage_end=base.coverage_end,
        chunk_size=base.chunk_size,
        chunks=tuple(filtered),
        content_hash=base.content_hash,
    )


def build_historical_collection_plan_from_universe_decisions(
    *,
    sessions: tuple[date, ...],
    decisions: tuple[UniverseDecision, ...],
    start: date,
    end: date,
    chunk_size: int = 20,
    warmup_sessions: int = 20,
    artifact_root: Path | str | None = None,
    input_receipt_digest: str = "pit-universe",
) -> HistoricalCollectionPlan:
    """Plan KIS requests from PIT eligibility, including only feature warm-up dates."""
    if start > end:
        raise PITDataError("coverage_start must not be after coverage_end")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise PITDataError("chunk_size must be a positive integer")
    if not isinstance(warmup_sessions, int) or isinstance(warmup_sessions, bool) or warmup_sessions < 0:
        raise PITDataError("warmup_sessions must be a non-negative integer")
    in_range = tuple(sorted({session for session in sessions if start <= session <= end}))
    if not in_range:
        raise PITDataError("no sessions inside declared coverage")
    index_by_session = {session: index for index, session in enumerate(in_range)}
    eligible_by_symbol: dict[str, list[int]] = {}
    seen: set[tuple[date, str]] = set()
    for decision in decisions:
        decision_date = decision.decision_session.date()
        if decision_date not in index_by_session:
            continue
        key = (decision_date, decision.instrument_id)
        if key in seen:
            raise PITDataError(f"duplicate PIT universe decision: {decision.instrument_id}:{decision_date.isoformat()}")
        seen.add(key)
        if not decision.eligible:
            continue
        eligible_by_symbol.setdefault(decision.instrument_id, []).append(index_by_session[decision_date])
    if not eligible_by_symbol:
        raise PITDataError("PIT universe has no eligible symbols")

    chunks: list[PlanChunk] = []
    digest = hashlib.sha256()
    digest.update(input_receipt_digest.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(start.isoformat().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(end.isoformat().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(chunk_size).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(warmup_sessions).encode("utf-8"))
    for symbol in sorted(eligible_by_symbol):
        digest.update(symbol.encode("utf-8"))
        digest.update(b"\x00")
        runs: list[list[int]] = []
        for index in sorted(eligible_by_symbol[symbol]):
            if not runs or index != runs[-1][-1] + 1:
                runs.append([index])
            else:
                runs[-1].append(index)
        for run in runs:
            required = list(range(max(0, run[0] - warmup_sessions), run[-1] + 1))
            for offset in range(0, len(required), chunk_size):
                members = required[offset : offset + chunk_size]
                chunk_sessions = tuple(in_range[index] for index in members)
                digest.update("|".join(value.isoformat() for value in chunk_sessions).encode("utf-8"))
                digest.update(b"\x00")
                chunks.append(PlanChunk(chunk_id="", symbol=symbol, sessions=chunk_sessions))
    content_hash = digest.hexdigest()
    plan_id = f"plan-{content_hash[:16]}"
    numbered_chunks = tuple(
        PlanChunk(chunk_id=f"{plan_id}:{chunk.symbol}:{index:04d}", symbol=chunk.symbol, sessions=chunk.sessions)
        for index, chunk in enumerate(chunks)
    )
    plan = HistoricalCollectionPlan(plan_id, start, end, chunk_size, numbered_chunks, content_hash)
    root = Path(artifact_root) if artifact_root is not None else PLAN_ARTIFACT_DIR
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{plan_id}.json").write_text(
            json.dumps(
                {
                    "plan_id": plan.plan_id,
                    "content_hash": plan.content_hash,
                    "coverage_start": start.isoformat(),
                    "coverage_end": end.isoformat(),
                    "chunk_size": chunk_size,
                    "warmup_sessions": warmup_sessions,
                    "input_receipt_digest": input_receipt_digest,
                    "chunks": [
                        {"chunk_id": chunk.chunk_id, "symbol": chunk.symbol, "sessions": [value.isoformat() for value in chunk.sessions]}
                        for chunk in numbered_chunks
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise PITDataError(_redact_message(f"plan receipt write failed: {type(exc).__name__}")) from exc
    return plan


class CollectionCheckpointStore:
    """Checkpoint of completed chunks keyed by receipt digest."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def _chunk_path(self, plan_id: str, chunk_id: str) -> Path:
        safe_plan = str(plan_id).strip().replace("/", "_") or "plan"
        safe_chunk = str(chunk_id).strip().replace("/", "_") or "chunk"
        return self._root / safe_plan / f"{safe_chunk}.json"

    def mark_complete(
        self,
        *,
        plan_id: str,
        chunk_id: str,
        receipt_digest: str,
        plan_digest: str | None = None,
        receipt_hashes: tuple[str, ...] = (),
    ) -> Path:
        if not str(plan_id).strip() or not str(chunk_id).strip() or not str(receipt_digest).strip():
            raise PITDataError("plan_id, chunk_id, and receipt_digest are required")
        path = self._chunk_path(plan_id, chunk_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "plan_id": plan_id,
                    "chunk_id": chunk_id,
                    "receipt_digest": receipt_digest,
                    "receipt_hashes": list(receipt_hashes),
                    "plan_digest": plan_digest,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    def is_pending(
        self,
        *,
        plan_id: str,
        chunk_id: str,
        receipt_digest: str,
        plan_digest: str | None = None,
        expected_plan_digest: str | None = None,
    ) -> bool:
        path = self._chunk_path(plan_id, chunk_id)
        if not path.exists():
            return True
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return True
        if not isinstance(stored, dict):
            return True
        if stored.get("receipt_digest") != receipt_digest:
            return True
        want_plan = expected_plan_digest if expected_plan_digest is not None else plan_digest
        # Stored checkpoint without plan digest cannot prove same plan.
        return want_plan is not None and stored.get("plan_digest") != want_plan  # noqa: SIM103

    def pending_chunks(
        self,
        plan: HistoricalCollectionPlan,
        receipt_digests: Any,
    ) -> tuple[PlanChunk, ...]:
        digests = dict(receipt_digests or {})
        pending: list[PlanChunk] = []
        for chunk in plan.chunks:
            current = digests.get(chunk.chunk_id)
            if current is None or self.is_pending(
                plan_id=plan.plan_id, chunk_id=chunk.chunk_id, receipt_digest=str(current), plan_digest=plan.content_hash,
                expected_plan_digest=plan.content_hash,
            ):
                pending.append(chunk)
        return tuple(pending)

    def has_verified_receipt(
        self, *, plan: HistoricalCollectionPlan, chunk: PlanChunk, bronze_root: Path | str
    ) -> bool:
        path = self._chunk_path(plan.plan_id, chunk.chunk_id)
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            digests = tuple(str(value) for value in stored.get("receipt_hashes", ()) if str(value))
            if not digests:
                digests = (str(stored.get("receipt_digest") or ""),)
            return (
                stored.get("plan_digest") == plan.content_hash
                and bool(digests)
                and all(
                    (payload := Path(bronze_root) / "investor_flow" / digest / "payload.json").exists()
                    and hashlib.sha256(payload.read_bytes()).hexdigest() == digest
                    for digest in digests
                )
            )
        except (OSError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class CollectionReadinessReport:
    certifiable: bool
    unresolved_reasons: tuple[str, ...] = ()
    coverage_gaps: tuple[str, ...] = ()
    pit_lineage_ok: bool = True
    action_provenance_ok: bool = True
    status_provenance_ok: bool = True
    corporate_status_reason: str = ""
    corporate_action_reason: str = ""

    @classmethod
    def incomplete(
        cls,
        corporate_status_reason: str = "",
        corporate_action_reason: str = "",
        coverage_gaps: tuple[str, ...] = (),
        unresolved_reasons: tuple[str, ...] = (),
    ) -> CollectionReadinessReport:
        reasons = list(unresolved_reasons)
        if corporate_status_reason and corporate_status_reason not in reasons:
            reasons.append(corporate_status_reason)
        if corporate_action_reason and corporate_action_reason not in reasons:
            reasons.append(corporate_action_reason)
        for gap in coverage_gaps:
            if gap not in reasons:
                reasons.append(gap)
        if not reasons:
            reasons.append("incomplete collection evidence")
        return cls(
            certifiable=False,
            unresolved_reasons=tuple(reasons),
            coverage_gaps=tuple(coverage_gaps),
            pit_lineage_ok=False,
            action_provenance_ok=not bool(corporate_action_reason),
            status_provenance_ok=not bool(corporate_status_reason),
            corporate_status_reason=corporate_status_reason,
            corporate_action_reason=corporate_action_reason,
        )

    @classmethod
    def certifiable_report(cls) -> CollectionReadinessReport:
        return cls(certifiable=True, unresolved_reasons=())

    def require_certifiable(self) -> CollectionReadinessReport:
        if self.certifiable and not self.unresolved_reasons:
            return self
        detail = "; ".join(self.unresolved_reasons) if self.unresolved_reasons else "unresolved collection gaps"
        raise PITDataError(_redact_message(f"Silver certification blocked: {detail}"))


@dataclass(frozen=True, slots=True)
class CollectionPlanReceipt:
    plan: HistoricalCollectionPlan
    receipt_path: Path


def load_collection_plan(plan_id: str, *, artifact_root: Path | str | None = None) -> HistoricalCollectionPlan:
    root = Path(artifact_root) if artifact_root is not None else PLAN_ARTIFACT_DIR
    path = root / f"{plan_id}.json"
    if not path.exists():
        raise PITDataError(f"unknown collection plan: {plan_id}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PITDataError("collection plan receipt is unreadable") from exc
    chunks = tuple(
        PlanChunk(
            chunk_id=str(item.get("chunk_id")),
            symbol=str(item.get("symbol")),
            sessions=tuple(date.fromisoformat(s) for s in item.get("sessions", ())),
        )
        for item in raw.get("chunks", ())
    )
    return HistoricalCollectionPlan(
        plan_id=str(raw.get("plan_id")),
        coverage_start=date.fromisoformat(str(raw.get("coverage_start"))),
        coverage_end=date.fromisoformat(str(raw.get("coverage_end"))),
        chunk_size=int(raw.get("chunk_size", 0) or 0),
        chunks=chunks,
        content_hash=str(raw.get("content_hash", "")),
    )


__all__ = [
    "CollectionCheckpointStore",
    "CollectionPlanReceipt",
    "CollectionReadinessReport",
    "HistoricalCollectionPlan",
    "PlanChunk",
    "build_historical_collection_plan",
    "build_historical_collection_plan_from_bronze",
    "build_historical_collection_plan_from_universe_decisions",
    "load_collection_plan",
]
