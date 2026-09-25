"""Historical collection plan, checkpoints, and readiness gates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final

from src.data.runtime import DataRuntime
from src.data.schemas import PITDataError
from src.data.scope_coverage import ScopeCoverageReport
from src.data.scoped_ingestion import FLOW_SOURCE

LS_MAX_SESSIONS_PER_REQUEST: Final[int] = 700


@dataclass(frozen=True, slots=True)
class PlanChunk:
    chunk_id: str
    symbol: str
    sessions: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class HistoricalCollectionPlan:
    plan_id: str
    coverage_start: date = date(2026, 3, 1)
    coverage_end: date = date(2026, 3, 6)
    chunk_size: int = 1
    chunks: tuple[PlanChunk, ...] = ()
    content_hash: str = ""
    dataset_name: str = ""
    created_at: datetime | None = None


class CollectionCheckpointStore:
    """Checkpoint of completed chunks keyed by receipt digest."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def _chunk_path(self, plan_id: str, chunk_id: str) -> Path:
        safe_plan = str(plan_id).strip().replace("/", "_") or "plan"
        safe_chunk = str(chunk_id).strip().replace("/", "_") or "chunk"
        return self._root / safe_plan / f"{safe_chunk}.json"

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


def load_collection_plan_path(path: Path | str) -> HistoricalCollectionPlan:
    """Load one immutable persisted historical collection plan by exact path.

    Scoped research plans are evidence-derived inputs rather than regenerated
    convenience configuration. Loading by exact path preserves their recorded
    identifier, digest, coverage, chunk order, and session windows for
    checkpoint replay.

    Args:
        path: Existing JSON plan receipt path.

    Returns:
        Parsed immutable historical collection plan.

    Raises:
        PITDataError: The path, JSON structure, plan identity, coverage, or
        chunk contract is invalid.
    """
    try:
        candidate = Path(path)
    except (TypeError, ValueError) as exc:
        raise PITDataError(f"collection plan path is invalid: {path}") from exc
    if not candidate.is_file():
        raise PITDataError(f"collection plan path is invalid: {candidate}")
    try:
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PITDataError("collection plan receipt is unreadable") from exc
    if not isinstance(raw, dict):
        raise PITDataError("collection plan receipt has invalid schema")
    plan_id = raw.get("plan_id")
    content_hash = raw.get("content_hash")
    if not isinstance(plan_id, str) or not plan_id.strip():
        raise PITDataError("collection plan receipt is missing its plan identity")
    if not isinstance(content_hash, str) or not content_hash.strip():
        raise PITDataError("collection plan receipt is missing its digest")
    try:
        coverage_start = date.fromisoformat(str(raw.get("coverage_start")))
        coverage_end = date.fromisoformat(str(raw.get("coverage_end")))
    except (ValueError, TypeError) as exc:
        raise PITDataError("collection plan receipt has invalid coverage") from exc
    if coverage_start > coverage_end:
        raise PITDataError("coverage_start must not be after coverage_end")
    chunk_size = raw.get("chunk_size")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise PITDataError("collection plan receipt has invalid chunk size")
    raw_chunks = raw.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise PITDataError("collection plan has no chunks")
    chunks: list[PlanChunk] = []
    seen_ids: set[str] = set()
    for item in raw_chunks:
        if not isinstance(item, dict):
            raise PITDataError("collection plan chunk has invalid schema")
        chunk_id = item.get("chunk_id")
        symbol = item.get("symbol")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise PITDataError("collection plan chunk is missing its identifier")
        if not isinstance(symbol, str) or not symbol.strip():
            raise PITDataError("collection plan chunk is missing its symbol")
        if chunk_id in seen_ids:
            raise PITDataError(f"collection plan has duplicate chunk identifier: {chunk_id!r}")
        seen_ids.add(chunk_id)
        raw_sessions = item.get("sessions")
        if not isinstance(raw_sessions, list) or not raw_sessions:
            raise PITDataError(f"collection plan chunk {chunk_id!r} has no sessions")
        sessions: list[date] = []
        for value in raw_sessions:
            try:
                session = date.fromisoformat(str(value))
            except (ValueError, TypeError) as exc:
                raise PITDataError(f"collection plan chunk {chunk_id!r} has invalid session") from exc
            if session < coverage_start or session > coverage_end:
                raise PITDataError(f"collection plan chunk {chunk_id!r} has a session outside coverage")
            sessions.append(session)
        chunks.append(PlanChunk(chunk_id=chunk_id, symbol=symbol, sessions=tuple(sessions)))
    return HistoricalCollectionPlan(
        plan_id=plan_id,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        chunk_size=chunk_size,
        chunks=tuple(chunks),
        content_hash=content_hash,
    )


def load_collection_plan(plan_id: str, *, artifact_root: Path | str) -> HistoricalCollectionPlan:
    root = Path(artifact_root)
    path = root / f"{plan_id}.json"
    if not path.exists():
        raise PITDataError(f"unknown collection plan: {plan_id}")
    return load_collection_plan_path(path)


__all__ = [
    "LS_MAX_SESSIONS_PER_REQUEST",
    "CollectionCheckpointStore",
    "HistoricalCollectionPlan",
    "PlanChunk",
    "build_scoped_flow_plan",
    "load_collection_plan",
    "load_collection_plan_path",
    "scoped_checkpoint_dir",
    "scoped_plan_dir",
]


def scoped_plan_dir(*, runtime: DataRuntime) -> Path:
    """Collection plan artifacts namespaced under the workspace state root."""
    return runtime.workspace.state_root / "collection-plans"


def scoped_checkpoint_dir(*, runtime: DataRuntime) -> Path:
    """Collection checkpoints namespaced under the workspace state root."""
    return runtime.workspace.state_root / "collection-checkpoints"


def build_scoped_flow_plan(
    *, runtime: DataRuntime, report: ScopeCoverageReport, max_sessions_per_request: int
) -> HistoricalCollectionPlan:
    """Chunk missing investor-flow sessions from a coverage report with adapter session limits."""
    if (
        not isinstance(max_sessions_per_request, int)
        or isinstance(max_sessions_per_request, bool)
        or max_sessions_per_request < 1
    ):
        raise PITDataError("max_sessions_per_request must be a positive integer")
    scope = runtime.scope
    pending: dict[str, list[date]] = {}
    for item in (*report.missing, *report.unresolved):
        if item.source != FLOW_SOURCE:
            continue
        symbol, _, session_text = item.natural_key.partition(":")
        if not symbol.strip() or not session_text.strip():
            raise PITDataError(f"invalid investor flow natural key {item.natural_key!r}")
        try:
            session = date.fromisoformat(session_text)
        except ValueError as exc:
            raise PITDataError(f"invalid investor flow natural key {item.natural_key!r}") from exc
        pending.setdefault(symbol, []).append(session)
    chunks: list[PlanChunk] = []
    for symbol in sorted(pending):
        ordered_sessions = sorted(set(pending[symbol]))
        for index in range(0, len(ordered_sessions), max_sessions_per_request):
            window = tuple(ordered_sessions[index : index + max_sessions_per_request])
            chunks.append(
                PlanChunk(
                    chunk_id=f"{symbol}-{index // max_sessions_per_request:04d}",
                    symbol=symbol,
                    sessions=window,
                )
            )
    digest = hashlib.sha256()
    digest.update(scope.content_hash.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(
        json.dumps([f"{chunk.symbol}:{day.isoformat()}" for chunk in chunks for day in chunk.sessions]).encode("utf-8")
    )
    digest.update(b"\x00")
    digest.update(str(max_sessions_per_request).encode("utf-8"))
    plan_id = f"scoped-flow-{digest.hexdigest()[:16]}"
    plan = HistoricalCollectionPlan(
        plan_id=plan_id,
        coverage_start=scope.completed_start,
        coverage_end=scope.completed_end,
        chunk_size=max_sessions_per_request,
        chunks=tuple(chunks),
        content_hash=digest.hexdigest(),
    )
    root = scoped_plan_dir(runtime=runtime)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{plan_id}.json").write_text(
        json.dumps(
            {
                "plan_id": plan_id,
                "content_hash": plan.content_hash,
                "scope_hash": scope.content_hash,
                "coverage_start": plan.coverage_start.isoformat(),
                "coverage_end": plan.coverage_end.isoformat(),
                "chunk_size": max_sessions_per_request,
                "chunks": [
                    {"chunk_id": chunk.chunk_id, "symbol": chunk.symbol, "sessions": [day.isoformat() for day in chunk.sessions]}
                    for chunk in chunks
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return plan
