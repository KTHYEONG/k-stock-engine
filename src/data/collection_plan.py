"""Historical collection plan, checkpoints, and readiness gates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.data.schemas import PITDataError

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
    ) -> Path:
        if not str(plan_id).strip() or not str(chunk_id).strip() or not str(receipt_digest).strip():
            raise PITDataError("plan_id, chunk_id, and receipt_digest are required")
        path = self._chunk_path(plan_id, chunk_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"plan_id": plan_id, "chunk_id": chunk_id, "receipt_digest": receipt_digest, "plan_digest": plan_digest},
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
    "load_collection_plan",
]
