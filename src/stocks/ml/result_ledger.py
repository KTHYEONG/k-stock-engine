"""Bounded ML result ledger: machine-readable projection of terminal runs.

The ledger is an AI-analysis projection, never an audit store: the immutable
artifact files under ``data/artifacts`` remain the source of truth. Every
generated target stays below ``docs/results/ml_runs/`` with fixed byte bounds,
and ``recent.jsonl`` retains only the newest terminal records. Projections
contain no per-instrument values, raw OOF scores, raw label rows, raw block
return vectors, stack traces, or credentials.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from src.core.instruments import AssetKind
from src.stocks.data.lineage import ResolvedDataLineage
from src.stocks.ml.contracts import (
    HorizonJoinEvidence,
    NetAlphaResearchData,
    NetAlphaTrainingRequest,
)
from src.stocks.research.artifacts import (
    ARTIFACT_ID_RE,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    ModelArtifactRegistry,
)
from src.stocks.research.models import ModelManifest

logger = logging.getLogger("stocks.ml.result_ledger")

# DiagnosticReport is imported lazily to avoid circular imports
_DiagnosticReport = None

SCHEMA_VERSION = 2
RETAINED_RECORDS = 128
MAX_SCHEMA_BYTES = 16 * 1024
MAX_LATEST_BYTES = 24 * 1024
MAX_RECORD_BYTES = 24 * 1024
MAX_META_BYTES = 4 * 1024
MAX_POINTER_BYTES = 4 * 1024
MAX_MESSAGE_CHARS = 512

LATEST_FILENAME = "latest.json"
RECENT_FILENAME = "recent.jsonl"
SCHEMA_FILENAME = "schema.json"
META_FILENAME = "ledger_meta.json"
POINTER_FILENAME = "back-res.md"

_REQUIRED_TOP_KEYS = (
    "schema_version",
    "artifact_id",
    "status",
    "started_at",
    "finished_at",
    "runtime",
    "input",
    "outcome",
    "observability",
    "artifact",
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def peak_rss_mib() -> float | None:
    """Return the process peak RSS in MiB, or ``None`` when unavailable.

    ``resource.getrusage`` reports the peak resident set size in KiB on Linux;
    the value is converted to MiB.  A missing ``resource`` module records
    ``None``, never a fabricated zero.
    """
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3)
    except (ImportError, AttributeError, ValueError):
        return None


def current_rss_mib() -> float | None:
    """Return the current resident set size in MiB, or ``None`` when unavailable.

    Reads ``/proc/self/statm`` on Linux; any failure records ``None``, never a
    fabricated zero.
    """
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        page_bytes = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_bytes / (1024.0 * 1024.0), 3)
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _normalize_message(value: object) -> str:
    """Collapse a message to one line and cap it to the ledger message bound."""
    text = str(value)
    text = " ".join(text.splitlines())
    return text[:MAX_MESSAGE_CHARS]


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, int) else 0


def summarize_numeric(values: Iterable[float | int | None]) -> dict[str, object]:
    """Aggregate finite-only statistics over ``values``.

    Empty (or fully non-finite) input returns ``count=0`` with ``None``
    statistics; non-finite values are never averaged or counted.
    """
    finite: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "max": None,
        }
    ordered = sorted(finite)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((value - mean) ** 2 for value in ordered) / count
    median = (
        ordered[count // 2]
        if count % 2
        else (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0
    )
    return {
        "count": count,
        "mean": round(mean, 12),
        "std": round(math.sqrt(variance), 12),
        "min": round(ordered[0], 12),
        "median": round(median, 12),
        "max": round(ordered[-1], 12),
    }


def _sanitize_deep(value: object) -> object:
    """Recursively project a value to a bounded JSON-safe object."""
    if isinstance(value, dict):
        return {str(key): _sanitize_deep(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_deep(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if value is None:
        return None
    return _normalize_message(value)


def _encode(record: Mapping[str, object]) -> bytes:
    """Canonical compact UTF-8 JSON encoding for ledger artifacts."""
    return _canonical_json(record)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_summary(value: object) -> dict[str, object]:
    """Deterministic ``{count, sha256}`` index of an unbounded collection.

    The ledger never copies members of an allowed metric collection; their
    full evidence stays in the referenced artifact ``metrics.json`` while the
    canonical digest pins the exact observed content.
    """
    sanitized = _sanitize_deep(value)
    if isinstance(sanitized, dict):
        count = len(sanitized)
        payload: object = {
            str(key): sanitized[key] for key in sorted(sanitized, key=str)
        }
    elif isinstance(sanitized, (list, tuple)):
        count = len(sanitized)
        payload = list(sanitized)
    else:
        return {}
    return {
        "count": count,
        "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
    }


def _validate_size(obj: Mapping[str, object], limit: int, label: str) -> None:
    size = len(_encode(obj))
    if size > limit:
        raise ValueError(f"{label} exceeds {limit} bytes ({size} bytes encoded)")


def _stable_hash(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_resolve(root: Path, *parts: str) -> Path:
    """Resolve ``parts`` below ``root``, rejecting traversal/symlink escapes."""
    root_resolved = root.resolve()
    target = root_resolved.joinpath(*parts).resolve()
    if not target.is_relative_to(root_resolved):
        raise ValueError(f"generated target escapes results root: {target}")
    return target


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` via a sibling temporary file and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".ledger-", delete=False
    ) as fh:
        temp_path = Path(fh.name)
        fh.write(payload)
    temp_path.replace(path)


def _sort_key(record: Mapping[str, object]) -> tuple[int, str, str]:
    finished = record.get("finished_at")
    if isinstance(finished, str) and finished:
        return (2, finished, str(record.get("artifact_id", "")))
    started = record.get("started_at")
    if isinstance(started, str) and started:
        return (2, started, str(record.get("artifact_id", "")))
    artifact_id = str(record.get("artifact_id", ""))
    match = re.search(r"\d{8}", artifact_id)
    date_key = match.group(0) if match else "00000000"
    return (1 if match else 0, date_key, artifact_id)


def _rebuild_recent(
    recent_path: Path,
    artifact_id: str,
    record: Mapping[str, object],
    retained_cap: int,
) -> tuple[list[dict[str, object]], int, int]:
    """Rebuild the bounded JSONL cache, deduplicating and retaining newest rows.

    Malformed rows are skipped and counted as ``discarded_invalid_records``; a
    duplicate artifact id replaces the cache entry in place. Returns
    ``(retained, discarded_due_to_retention, discarded_invalid)``.
    """
    lines: list[dict[str, object]] = []
    invalid = 0
    if recent_path.exists():
        with recent_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                if not isinstance(obj, dict) or not isinstance(
                    obj.get("artifact_id"), str
                ):
                    invalid += 1
                    continue
                lines.append(obj)
    newest: dict[str, dict[str, object]] = {}
    for obj in lines:
        key = str(obj["artifact_id"])
        if key not in newest or _sort_key(obj) > _sort_key(newest[key]):
            newest[key] = obj
    newest[str(artifact_id)] = dict(record)
    ordered = sorted(newest.values(), key=_sort_key, reverse=True)
    retained = ordered[:retained_cap]
    discarded = len(ordered) - len(retained)
    return retained, discarded, invalid


def _read_prior_counts(meta_path: Path) -> tuple[int, int]:
    """Return previously accumulated discard counts, if any."""
    if not meta_path.exists():
        return 0, 0
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return 0, 0
    if not isinstance(data, dict):
        return 0, 0
    return _as_int(data.get("discarded_count")), _as_int(
        data.get("discarded_invalid_records")
    )


def _build_meta(
    updated_at: datetime,
    retained: int,
    discarded: int,
    invalid: int,
    retained_cap: int,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": updated_at.isoformat(),
        "retained_count": retained,
        "discarded_count": discarded,
        "discarded_invalid_records": invalid,
        "retained_records_cap": retained_cap,
        "record_byte_limit": MAX_RECORD_BYTES,
    }


def _build_schema(retained_cap: int) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_byte_limit": MAX_RECORD_BYTES,
        "retained_records_cap": retained_cap,
        "files": {
            LATEST_FILENAME: "Complete compact projection of the newest terminal run.",
            RECENT_FILENAME: (
                "Newest terminal records, one canonical JSON object per line."
            ),
            META_FILENAME: (
                "Schema version, update time, retained/discarded counts, byte limits."
            ),
            SCHEMA_FILENAME: "Versioned field dictionary and retention policy.",
            POINTER_FILENAME: (
                "Generated 1 KiB pointer to latest.json and recent.jsonl."
            ),
        },
        "record": {
            "schema_version": "int",
            "artifact_id": "str",
            "status": "completed | failed",
            "started_at": "UTC ISO-8601",
            "finished_at": "UTC ISO-8601",
            "runtime": {"elapsed_ms": "int", "peak_rss_mib": "float|null"},
            "input": {
                "snapshot_id": "str",
                "request": "bounded request projection",
                "data": "bounded data shape + join evidence",
                "cost_context": "cost schedule kind and evidence identity",
            },
            "outcome": (
                "promotion/no-trade evidence, selected profile, and gates"
            ),
            "observability": {
                "phases": (
                    "bounded phase samples: fixed scalars plus payload digest"
                ),
                "horizons": (
                    "per-candidate admission/fold scalars and numeric summaries"
                ),
                "summary": (
                    "bounded fold/cohort/multiplicity/operation-count/schema "
                    "diagnostics; frontier maps appear only as digests"
                ),
                "policy_frontier": (
                    "candidate/profile counts plus {count, sha256} digest "
                    "summaries of profile ids, dropout reasons, segment sums"
                ),
                "replay": "aggregate period-return summary",
                "holdout": (
                    "compound certificate (base/stress CAGR, lower CAGR, MDD, "
                    "Calmar, pass flags), cohort counts, eligibility interval"
                ),
            },
            "artifact": {
                "manifest_path": "str",
                "manifest_bytes": "int",
                "manifest_sha256": "sha256 hex",
                "metrics_path": "str",
                "metrics_bytes": "int",
                "metrics_sha256": "sha256 hex",
            },
        },
        "forbidden": [
            "per-instrument values",
            "raw OOF scores",
            "raw label rows",
            "raw block return vectors",
            "stack traces",
            "credentials",
            "arbitrary CLI environment variables",
        ],
    }


def _build_pointer(
    record: Mapping[str, object],
    retained: int,
    discarded: int,
    invalid: int,
    retained_cap: int,
) -> str:
    lines = [
        "# ML Result Ledger",
        "",
        f"- Schema version: {SCHEMA_VERSION}",
        f"- Latest artifact: `{record.get('artifact_id')}`",
        f"- Status: `{record.get('status')}`",
        f"- Finished: {record.get('finished_at') or record.get('started_at') or 'n/a'}",
        f"- Latest JSON: `docs/results/ml_runs/{LATEST_FILENAME}`",
        f"- Recent JSONL: `docs/results/ml_runs/{RECENT_FILENAME}`",
        f"- Retention: newest {retained_cap} records, each <= {MAX_RECORD_BYTES} bytes",
        f"- Retained: {retained} | discarded: {discarded} | invalid: {invalid}",
        "",
    ]
    outcome_raw = record.get("outcome")
    outcome: Mapping[str, object] = outcome_raw if isinstance(outcome_raw, Mapping) else {}
    obs_raw = record.get("observability")
    obs: Mapping[str, object] = obs_raw if isinstance(obs_raw, Mapping) else {}
    holdout_raw = obs.get("holdout")
    holdout: Mapping[str, object] = holdout_raw if isinstance(holdout_raw, Mapping) else {}
    cert_raw = holdout.get("certificate")
    cert: Mapping[str, object] = cert_raw if isinstance(cert_raw, Mapping) else {}
    base_cert_raw = cert.get("base")
    base_cert: Mapping[str, object] = base_cert_raw if isinstance(base_cert_raw, Mapping) else {}
    summary_raw = obs.get("summary")
    summary: Mapping[str, object] = summary_raw if isinstance(summary_raw, Mapping) else {}
    frontier_raw = obs.get("policy_frontier")
    frontier: Mapping[str, object] = frontier_raw if isinstance(frontier_raw, Mapping) else {}
    exec_ev_raw = frontier.get("execution_evidence")
    exec_ev: Mapping[str, object] = exec_ev_raw if isinstance(exec_ev_raw, Mapping) else {}

    model_family = outcome.get("model_family") or outcome.get("model_type") or "n/a"
    promoted = outcome.get("promoted", False)
    horizons = outcome.get("selected_horizons") or summary.get("evidence_horizons")
    selected_horizon = horizons[0] if isinstance(horizons, (list, tuple)) and horizons else None

    lines.extend([
        "## Latest Backtest & Compounding Performance",
        "",
        f"- **Model Family**: `{model_family}`",
        f"- **Promotion Status**: `{'PROMOTED (Active)' if promoted else 'NO_TRADE (Inactive)'}`",
        f"- **Selected Horizon**: `{selected_horizon}d`" if selected_horizon is not None else "- **Selected Horizon**: `N/A`",
    ])
    if base_cert:
        cagr = base_cert.get("cagr")
        mdd = base_cert.get("mdd")
        calmar = base_cert.get("calmar")
        lower_cagr = base_cert.get("lower_cagr")
        if isinstance(cagr, (int, float)):
            lines.append(f"- **Holdout CAGR (Compounding)**: `{float(cagr) * 100:.2f}%`")
        if isinstance(mdd, (int, float)):
            lines.append(f"- **Holdout MDD**: `{float(mdd) * 100:.2f}%`")
        if isinstance(calmar, (int, float)):
            lines.append(f"- **Holdout Calmar Ratio**: `{float(calmar):.2f}`")
        if isinstance(lower_cagr, (int, float)):
            lines.append(f"- **Holdout Lower-Bound CAGR**: `{float(lower_cagr) * 100:.2f}%`")

    holdout_orders = holdout.get("order_count")
    if isinstance(holdout_orders, int):
        lines.append(f"- **Holdout Filled Orders**: `{holdout_orders:,}`")

    if exec_ev:
        for prof_key, ev in exec_ev.items():
            if isinstance(ev, Mapping):
                filled = ev.get("filled_orders")
                turnover = ev.get("turnover")
                cycles = ev.get("planned_cycles")
                if isinstance(filled, int) and isinstance(turnover, (int, float)):
                    lines.append(f"- **OOF Replay Fills ({prof_key})**: `{filled:,} orders` across `{cycles}` cycles (Annual Turnover: `{float(turnover):.2f}x`)")

    lines.extend([
        "",
        "Artifact files under `data/artifacts/` remain the source of truth.",
        "",
    ])
    return "\n".join(lines)


def _read_artifact_json(
    registry: ModelArtifactRegistry, artifact_id: str, filename: str
) -> dict[str, object]:
    if not ARTIFACT_ID_RE.match(artifact_id):
        raise ValueError(f"invalid artifact_id {artifact_id!r}")
    path = Path(registry.root) / artifact_id / filename
    if not path.exists():
        raise ValueError(f"artifact file missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed artifact file: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"artifact file must be a JSON object: {path}")
    return data


def _artifact_reference(root: Path, artifact_id: str) -> dict[str, object]:
    """Exact-byte identity of the artifact files backing one ledger record.

    Returns the validated manifest/metrics filenames, byte lengths, and
    SHA-256 digests of the bytes used for a completed or rebuilt projection.
    """
    reference: dict[str, object] = {}
    for prefix, filename in (
        ("manifest", MANIFEST_FILENAME),
        ("metrics", METRICS_FILENAME),
    ):
        path = Path(root) / artifact_id / filename
        if not path.is_file():
            raise ValueError(f"artifact file missing: {path}")
        payload = path.read_bytes()
        reference[f"{prefix}_path"] = filename
        reference[f"{prefix}_bytes"] = len(payload)
        reference[f"{prefix}_sha256"] = hashlib.sha256(payload).hexdigest()
    return reference


def _validate_artifact_identity(
    context: MlRunContext,
    manifest: ModelManifest,
    manifest_json: Mapping[str, object],
) -> None:
    if context.artifact_id != manifest.artifact_id:
        raise ValueError(
            f"artifact id mismatch: context {context.artifact_id!r}, "
            f"manifest {manifest.artifact_id!r}"
        )
    persisted = manifest_json.get("artifact_id")
    if persisted != context.artifact_id:
        raise ValueError(
            f"artifact id mismatch: context {context.artifact_id!r}, "
            f"persisted {persisted!r}"
        )


def _validate_record(record: Mapping[str, object]) -> None:
    for key in _REQUIRED_TOP_KEYS:
        if key not in record:
            raise ValueError(f"ledger record missing required key {key!r}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported ledger schema_version {record.get('schema_version')!r}"
        )
    if record.get("status") not in ("completed", "failed"):
        raise ValueError(f"invalid terminal status {record.get('status')!r}")
    _validate_size(record, MAX_RECORD_BYTES, "ledger record")


@dataclass(frozen=True, slots=True)
class CostRunContext:
    """Bounded cost context captured for a ledger projection."""

    cost_schedule_kind: str
    cost_evidence_path: str | None = None
    cost_evidence_hash: str | None = None
    has_liquidity_model: bool = False


@dataclass(frozen=True, slots=True)
class MlRunContext:
    """Immutable run context for one ledger record; no frame is copied."""

    artifact_id: str
    snapshot_id: str
    started_at: datetime
    request: NetAlphaTrainingRequest
    feature_rows: int
    instrument_count: int
    session_count: int
    feature_column_count: int
    feature_session_range: tuple[str, str] | None
    label_definition: str
    label_horizon_sessions: int
    feature_schema_hash: str
    universe_policy_hash: str
    join_evidence: tuple[HorizonJoinEvidence, ...] = ()
    cost_context: CostRunContext = field(
        default_factory=lambda: CostRunContext(cost_schedule_kind="base")
    )
    data_lineage: ResolvedDataLineage | None = None
    input_ids: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_cli(
        cls,
        *,
        request: NetAlphaTrainingRequest,
        snapshot_id: str,
        data: NetAlphaResearchData,
        cost_context: CostRunContext,
        started_at: datetime,
        data_lineage: ResolvedDataLineage | None = None,
        input_ids: Mapping[str, str] | None = None,
    ) -> MlRunContext:
        """Capture request, snapshot identity, and composed-data shape."""
        frame = data.feature_frame
        sessions = sorted(frame["session"].unique().to_list())
        session_range = (
            (str(sessions[0]), str(sessions[-1])) if sessions else None
        )
        return cls(
            artifact_id=request.artifact_id,
            snapshot_id=snapshot_id,
            started_at=started_at,
            request=request,
            feature_rows=int(frame.height),
            instrument_count=int(frame["instrument_id"].n_unique()),
            session_count=len(sessions),
            feature_column_count=len(frame.columns),
            feature_session_range=session_range,
            label_definition=data.manifest.label_definition,
            label_horizon_sessions=data.manifest.label_horizon_sessions,
            feature_schema_hash=data.manifest.schema_hash or "net-alpha-v1",
            universe_policy_hash=data.manifest.universe_policy_hash or "net-alpha-v1",
            join_evidence=tuple(data.join_evidence),
            cost_context=cost_context,
            data_lineage=data_lineage,
            input_ids=input_ids or {},
        )


def _project_request(request: NetAlphaTrainingRequest) -> dict[str, object]:
    portfolio = request.portfolio
    risk = request.risk
    projection: dict[str, object] = {
        "candidate_horizon_sessions": [
            int(horizon) for horizon in request.candidate_horizon_sessions
        ],
        "execution_frontier": {
            "candidate_horizon_sessions": [
                int(h) for h in request.execution_frontier.candidate_horizon_sessions
            ],
            "candidate_rebalance_frequency_sessions": [
                int(c) for c in request.execution_frontier.candidate_rebalance_frequency_sessions
            ],
            "candidate_top_k": [
                int(k) for k in request.execution_frontier.candidate_top_k
            ],
        },
        "policy_profiles": [
            {"profile_id": profile.profile_id, "no_trade_band_bps": profile.no_trade_band_bps}
            for profile in request.policy_profiles
        ],
        "fold_count": request.fold_count,
        "embargo_sessions": request.embargo_sessions,
        "forward_holdout_sessions": request.forward_holdout_sessions,
        "bootstrap_alpha": request.bootstrap_alpha,
        "bootstrap_resamples": request.bootstrap_resamples,
        "model_threads": request.model_threads,
        "max_rss_mib": request.max_rss_mib,
        # NetAlphaTrainingRequest.max_training_lookback_sessions joins the
        # fingerprint so different rolling fit windows never share an identity.
        "max_training_lookback_sessions": request.max_training_lookback_sessions,
        "seed": request.seed,
        "enable_sparse_retained_rewaterfill": request.enable_sparse_retained_rewaterfill,
        "portfolio": {
            "top_k": portfolio.top_k,
            "max_single_weight": portfolio.max_single_weight,
            "max_exposure": portfolio.max_exposure,
            "participation_limit": portfolio.participation_limit,
            "portfolio_value": portfolio.portfolio_value,
            "initial_cash": portfolio.initial_cash,
            "reference_notional": portfolio.reference_notional,
        },
        "risk": {
            "calibration_bucket_count": risk.calibration_bucket_count,
            "min_calibration_sessions": risk.min_calibration_sessions,
            "risk_aversion": risk.risk_aversion,
            "no_trade_band_bps": risk.no_trade_band_bps,
        },
    }
    projection["request_fingerprint"] = _stable_hash(projection)
    return projection


def _project_data(context: MlRunContext) -> dict[str, object]:
    session_range = (
        list(context.feature_session_range) if context.feature_session_range else []
    )
    return {
        "feature_rows": context.feature_rows,
        "instrument_count": context.instrument_count,
        "session_count": context.session_count,
        "feature_column_count": context.feature_column_count,
        "feature_session_range": session_range,
        "label_definition": context.label_definition,
        "label_horizon_sessions": context.label_horizon_sessions,
        "feature_schema_hash": context.feature_schema_hash,
        "universe_policy_hash": context.universe_policy_hash,
        "horizons": [
            {
                "horizon_sessions": evidence.horizon_sessions,
                "feature_rows": evidence.feature_rows,
                "label_rows": evidence.label_rows,
                "joined_rows": evidence.joined_rows,
                "drop_reasons_digest": _digest_summary(
                    list(evidence.drop_reasons)
                ),
            }
            for evidence in context.join_evidence
        ],
    }


def _project_cost_context(cost: CostRunContext) -> dict[str, object]:
    return {
        "cost_schedule_kind": cost.cost_schedule_kind,
        "cost_evidence_path": (
            _normalize_message(cost.cost_evidence_path)
            if cost.cost_evidence_path
            else None
        ),
        "cost_evidence_hash": cost.cost_evidence_hash,
        "liquidity_model": cost.has_liquidity_model,
    }


def _project_outcome(
    manifest: ModelManifest, metrics: Mapping[str, object]
) -> dict[str, object]:
    model_type = str(metrics.get("model_type") or manifest.model_type)
    promoted = bool(metrics.get("promoted", model_type != "no_trade"))
    no_trade = bool(metrics.get("no_trade", model_type == "no_trade"))
    gates = metrics.get("gates")
    if not isinstance(gates, dict):
        gates = {}
    raw_reasons = metrics.get("promotion_reasons")
    if not isinstance(raw_reasons, list):
        raw_reasons = gates.get("reasons", []) if isinstance(gates, dict) else []
    reasons = [_normalize_message(reason) for reason in raw_reasons if isinstance(reason, str)]
    selected_profile = metrics.get("selected_profile")
    gate_reasons = (
        [
            _normalize_message(reason)
            for reason in gates.get("reasons", [])
            if isinstance(reason, str)
        ]
        if isinstance(gates, dict)
        else []
    )
    return {
        "model_family": model_type,
        "model_type": model_type,
        "promoted": promoted,
        "no_trade": no_trade,
        "promotion_reasons_digest": _digest_summary(reasons),
        "selected_horizons": _selected_horizons(metrics, manifest),
        "selected_profile": {
            "profile_id": (
                str(selected_profile.get("profile_id", ""))
                if isinstance(selected_profile, dict)
                else str(metrics.get("primary_profile_id") or "")
            ),
            "no_trade_band_bps": (
                _as_float(selected_profile.get("no_trade_band_bps"))
                if isinstance(selected_profile, dict)
                else None
            ),
        },
        "gates": {
            "passed": bool(gates.get("passed", promoted)),
            "reasons_digest": _digest_summary(gate_reasons),
        },
    }


def _selected_horizons(
    metrics: Mapping[str, object], manifest: ModelManifest
) -> list[int]:
    selection = metrics.get("horizon_selection")
    if isinstance(selection, dict):
        candidates = (
            selection.get("primary_horizon_sessions"),
            selection.get("secondary_horizon_sessions"),
        )
        selected = [int(h) for h in candidates if isinstance(h, int) and h > 0]
        if selected:
            return selected
    if not metrics.get("no_trade", False) and manifest.label_horizon_sessions:
        return [manifest.label_horizon_sessions]
    return []


def _project_replay(metrics: Mapping[str, object]) -> dict[str, object]:
    replay = metrics.get("replay")
    if not isinstance(replay, dict):
        return {}
    order_count = replay.get("order_count")
    block_count = replay.get("block_count")
    decisions = replay.get("decisions")
    period_net_returns = replay.get("period_net_returns")
    if not isinstance(period_net_returns, (list, tuple)):
        # Read-side compatibility for legacy artifacts that serialized the
        # arithmetic block returns under the old misnamed key.
        period_net_returns = replay.get("block_log_excess")
    return {
        "order_count": int(order_count) if isinstance(order_count, int) else 0,
        "block_count": int(block_count) if isinstance(block_count, int) else 0,
        "decision_count": (
            len(decisions) if isinstance(decisions, (list, tuple)) else 0
        ),
        "block_excess_summary": summarize_numeric(
            period_net_returns
            if isinstance(period_net_returns, (list, tuple))
            else []
        ),
    }


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _project_certificate_path(path: object) -> dict[str, object]:
    if not isinstance(path, dict):
        return {}
    return {
        "passed": bool(path.get("passed", False)),
        "reasons_digest": _digest_summary(
            [reason for reason in path.get("reasons", []) if isinstance(reason, str)]
        ),
        "cagr": _as_float(path.get("cagr")),
        "lower_cagr": _as_float(path.get("lower_cagr")),
        "mdd": _as_float(path.get("mdd")),
        "calmar": _as_float(path.get("calmar")),
    }


def _project_holdout(metrics: Mapping[str, object]) -> dict[str, object]:
    holdout = metrics.get("holdout")
    if not isinstance(holdout, dict):
        return {}
    order_count = holdout.get("order_count")
    block_count = holdout.get("block_count")
    certificate = holdout.get("certificate")
    cert = certificate if isinstance(certificate, dict) else {}
    cohorts = holdout.get("cohorts")
    cohort_counts = cohorts if isinstance(cohorts, dict) else {}
    eligibility = holdout.get("eligibility")
    eligibility = eligibility if isinstance(eligibility, dict) else {}
    return {
        "passed": bool(holdout.get("passed", False)),
        "reason": _normalize_message(holdout.get("reason", "")),
        "order_count": int(order_count) if isinstance(order_count, int) else 0,
        "block_count": int(block_count) if isinstance(block_count, int) else 0,
        "cohorts": {
            "scored_sessions": _as_int(cohort_counts.get("scored_sessions")),
            "realized_sessions": _as_int(cohort_counts.get("realized_sessions")),
            "eligible_sessions": _as_int(cohort_counts.get("eligible_sessions")),
            "period_count": _as_int(cohort_counts.get("period_count")),
            "observed_sessions": _as_int(cohort_counts.get("observed_sessions")),
            "active_cohort_count": _as_int(
                cohort_counts.get("active_cohort_count")
            ),
            "missing_realized_cohorts": _as_int(
                cohort_counts.get("missing_realized_cohorts")
            ),
        },
        "certificate": {
            "passed": bool(cert.get("passed", False)),
            "reasons": [
                _normalize_message(reason)
                for reason in cert.get("reasons", [])
                if isinstance(reason, str)
            ],
            "base": _project_certificate_path(cert.get("base")),
            "stress": _project_certificate_path(cert.get("stress")),
        },
        "eligibility": {
            "eligible_from": _normalize_message(
                eligibility.get("eligible_from", "")
            ),
            "eligible_to": _normalize_message(eligibility.get("eligible_to", "")),
        },
    }


_COMPACT_SCALAR_KEY_LIMIT = 16


def _json_scalar(value: object) -> object:
    """Project one scalar to a bounded finite JSON-safe value."""
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return _normalize_message(value)


def _compact_entry(entry: Mapping[str, object]) -> dict[str, object]:
    """Fixed-scalar projection of one phase/horizon entry.

    Scalar fields survive (capped count); numeric arrays collapse to
    finite-only summaries; every other collection is replaced by a
    deterministic ``payload_digest`` instead of being copied.
    """
    scalars: dict[str, object] = {}
    kept = 0
    overflow: dict[str, object] = {}
    for key in sorted(str(item) for item in entry):
        value = entry[key]
        if (
            isinstance(value, (list, tuple))
            and bool(value)
            and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            )
        ):
            scalars[f"{key}_summary"] = summarize_numeric(value)
            continue
        if isinstance(value, (dict, list, tuple)):
            overflow[key] = value
            continue
        if kept < _COMPACT_SCALAR_KEY_LIMIT:
            scalars[key] = _json_scalar(value)
            kept += 1
        else:
            overflow[key] = value
    if overflow:
        scalars["payload_digest"] = _digest_summary(overflow)
    return scalars


def _compact_entries(source: Mapping[str, object], key: str) -> list[dict[str, object]]:
    entries = source.get(key)
    if not isinstance(entries, (list, tuple)):
        return []
    return [
        _compact_entry(entry) for entry in entries if isinstance(entry, Mapping)
    ]


def _compact_phases(source: Mapping[str, object]) -> list[dict[str, object]]:
    return _compact_entries(source, "phases")


def _compact_horizons(source: Mapping[str, object]) -> list[dict[str, object]]:
    return _compact_entries(source, "horizons")


def _compact_policy_frontier(metrics: Mapping[str, object]) -> dict[str, object]:
    """Digest-backed frontier index: counts plus ``{count, sha256}`` summaries.

    Never copies policy-frontier maps; candidate/profile counts and the
    deterministic digests of profile ids, dropout reasons, and segment sums
    pin the evidence that stays in the artifact metrics.
    """
    frontier = metrics.get("policy_frontier")
    if not isinstance(frontier, dict):
        return {}
    profile_ids = frontier.get("profile_ids")
    return {
        "candidate_count": _as_int(frontier.get("candidate_count")),
        "profile_count": (
            len(profile_ids) if isinstance(profile_ids, (list, tuple)) else 0
        ),
        "profile_ids_digest": _digest_summary(profile_ids),
        "dropout_reasons_digest": _digest_summary(frontier.get("dropout_reasons")),
        "segment_sums_digest": _digest_summary(frontier.get("segment_sums")),
    }


def _compact_growth_route(metrics: Mapping[str, object]) -> dict[str, object]:
    """Bounded growth-route index: scalars plus normalized rejection counts.

    Projects the artifact metrics' ``growth_route`` projection into fixed
    scalars (candidate/segment counts, selected policy label, lower-growth
    CAGRs, coverage, fills) and per-reason counts; raw return arrays, policy
    lists, and any other collection are pinned by digest instead of copied.
    """
    route = metrics.get("growth_route")
    if not isinstance(route, Mapping):
        return {}
    reasons = route.get("rejection_reason_counts")
    reason_counts: dict[str, int] = {}
    if isinstance(reasons, Mapping):
        for key, value in reasons.items():
            count = _as_int(value)
            if count:
                reason_counts[_normalize_message(key)] = count
    summary = {
        "version": _json_scalar(route.get("version")),
        "candidate_count": _as_int(route.get("candidate_count")),
        "segment_count": _as_int(route.get("segment_count")),
        "cash_segment_count": _as_int(route.get("cash_segment_count")),
        "selected_policy": _json_scalar(route.get("selected_policy")),
        "base_lower_cagr": _as_float(route.get("base_lower_cagr")),
        "stress_lower_cagr": _as_float(route.get("stress_lower_cagr")),
        "matched_lower_excess_cagr": _as_float(
            route.get("matched_lower_excess_cagr")
        ),
        "mdd": _as_float(route.get("mdd")),
        "observed_intervals": _as_int(route.get("observed_intervals")),
        "invested_intervals": _as_int(route.get("invested_intervals")),
        "filled_orders": _as_int(route.get("filled_orders")),
        "rejection_reason_counts": reason_counts,
        "policies_digest": _digest_summary(route.get("selected_policies_digest")),
    }
    outcome = route
    summary['promotion_status'] = outcome.get('promotion_status', 'NO_TRADE')
    summary['hedge_sleeve'] = _compact_hedge_sleeve(route)
    return summary


def _compact_hedge_sleeve(route: Mapping[str, object]) -> dict[str, object]:
    """Bounded hedge-sleeve scalars from the growth-route projection."""
    sleeve = route.get("hedge_sleeve_projection")
    if not isinstance(sleeve, Mapping):
        return {}
    return {
        "leverage_rung_count": _as_int(sleeve.get("leverage_rung_count")),
        "admissible_rung_count": _as_int(sleeve.get("admissible_rung_count")),
        "max_admissible_leverage": _as_float(
            sleeve.get("max_admissible_leverage")
        ),
        "vol_managed_max_admissible_leverage": _as_float(
            sleeve.get("vol_managed_max_admissible_leverage")
        ),
        "excess_point_cagr": _as_float(sleeve.get("excess_point_cagr")),
    }


def _compact_observability(
    metrics: Mapping[str, object],
    telemetry: Mapping[str, object] | None,
) -> dict[str, object]:
    """Single bounded observability projector for live and rebuilt records."""
    source: Mapping[str, object] = {}
    if isinstance(telemetry, Mapping):
        source = telemetry
    else:
        run_obs = metrics.get("run_observability")
        if isinstance(run_obs, Mapping):
            source = run_obs
    return {
        "phases": _compact_phases(source),
        "horizons": _compact_horizons(source),
        "summary": _bounded_observability_summary(source),
        "policy_frontier": _compact_policy_frontier(metrics),
        "growth_route": _compact_growth_route(metrics),
    }


def _bounded_observability_summary(
    telemetry: Mapping[str, object],
) -> dict[str, object]:
    """Project bounded fold/cohort/multiplicity/operation/schema diagnostics.

    Scans the terminal telemetry phases for the bounded scalars the redesigned
    trainer publishes (schema fingerprint, path counts, fold geometry, cohort
    completeness, adjusted evidence, challenger reason, policy-frontier dropout
    reasons) and never extracts raw score, label, order, or return arrays.
    """
    phases = telemetry.get("phases")
    if not isinstance(phases, list):
        return {}
    by_name: dict[str, Mapping[str, object]] = {}
    for phase in phases:
        if isinstance(phase, dict) and isinstance(phase.get("name"), str):
            by_name[str(phase["name"])] = phase
    summary: dict[str, object] = {}
    feature = by_name.get("feature_transform", {})
    if isinstance(feature.get("schema_fingerprint"), str):
        summary["schema_fingerprint"] = feature["schema_fingerprint"]
    discovery = by_name.get("horizon_discovery", {})
    for key in (
        "path_evaluation_count",
        "path_evaluation_bound",
        "evidence_horizons",
        "diagnostics_count",
    ):
        if key in discovery:
            summary[key] = discovery[key]
    frontier = by_name.get("policy_frontier", {})
    if isinstance(frontier.get("candidate_count"), int):
        summary["frontier_candidate_count"] = frontier["candidate_count"]
        summary["frontier_candidate_bound"] = frontier.get("candidate_bound")
        profile_ids = frontier.get("profile_ids")
        summary["frontier_profile_count"] = (
            len(profile_ids) if isinstance(profile_ids, (list, tuple)) else 0
        )
        summary["frontier_profile_ids_digest"] = _digest_summary(profile_ids)
        # Frontier maps stay in the artifact; the summary pins them by digest
        # instead of duplicating them beside observability.policy_frontier.
        summary["frontier_dropout_reasons_digest"] = _digest_summary(
            frontier.get("dropout_reasons")
        )
        summary["frontier_segment_sums_digest"] = _digest_summary(
            frontier.get("segment_sums")
        )
    horizons = telemetry.get("horizons")
    if isinstance(horizons, list):
        # Bounded replay runtime scalars, aggregated across horizon entries:
        # disjoint prepare/execute timers plus actual builds and live bytes.
        prepare_total = 0
        execute_total = 0
        build_total = 0
        cache_peak = 0
        has_replay_runtime = False
        for entry in horizons:
            if not isinstance(entry, dict):
                continue
            prepare_value = entry.get("replay_prepare_elapsed_ms")
            execute_value = entry.get("replay_execute_elapsed_ms")
            build_value = entry.get("prepared_segment_build_count")
            cache_value = entry.get("prepared_cache_bytes")
            if any(
                isinstance(value, (int, float))
                for value in (prepare_value, execute_value, build_value, cache_value)
            ):
                has_replay_runtime = True
            prepare_total += int(prepare_value or 0)
            execute_total += int(execute_value or 0)
            build_total += int(build_value or 0)
            cache_peak = max(cache_peak, int(cache_value or 0))
        if has_replay_runtime:
            summary["replay_prepare_elapsed_ms"] = prepare_total
            summary["replay_execute_elapsed_ms"] = execute_total
            summary["prepared_segment_build_count"] = build_total
            summary["prepared_cache_bytes_peak"] = cache_peak
    selection = by_name.get("primary_selection", {})
    if "primary_horizon_sessions" in selection:
        summary["primary_horizon_sessions"] = selection["primary_horizon_sessions"]
    if "primary_rebalance_frequency_sessions" in selection:
        summary["primary_rebalance_frequency_sessions"] = (
            selection["primary_rebalance_frequency_sessions"]
        )
    if "primary_top_k" in selection:
        summary["primary_top_k"] = selection["primary_top_k"]
    if "primary_profile_id" in selection:
        summary["primary_profile_id"] = selection["primary_profile_id"]
    if "rankability_reason" in selection:
        summary["rankability_reason"] = selection["rankability_reason"]
    comparison = by_name.get("model_comparison", {})
    if "selected_model_type" in comparison:
        summary["selected_model_type"] = comparison["selected_model_type"]
        summary["challenger_failure_reason"] = comparison.get(
            "challenger_failure_reason", ""
        )
    sanitized = _sanitize_deep(summary)
    if not isinstance(sanitized, dict):
        return {}
    return {str(key): value for key, value in sanitized.items()}


def _observability_from(
    metrics: Mapping[str, object], telemetry: Mapping[str, object] | None
) -> dict[str, object]:
    """Compact observability for both live telemetry and persisted state."""
    return _compact_observability(metrics, telemetry)


_FIT_DROP_ORDER = ("runtime", "input", "outcome")


def _minimal_terminal_projection(record: Mapping[str, object]) -> dict[str, object]:
    """Deterministic terminal record for projections exceeding the byte budget.

    Retains identity/status/timing/outcome scalars and the artifact digest
    references; everything omitted is pinned by ``omitted_record_sha256``.
    """
    input_raw = record.get("input")
    input_map = input_raw if isinstance(input_raw, Mapping) else {}
    snapshot_id = input_map.get("snapshot_id")
    outcome_raw = record.get("outcome")
    outcome_map = outcome_raw if isinstance(outcome_raw, Mapping) else {}
    profile = outcome_map.get("selected_profile")
    profile = profile if isinstance(profile, Mapping) else {}
    horizons = outcome_map.get("selected_horizons")
    runtime = record.get("runtime")
    artifact_raw = record.get("artifact")
    reference = dict(artifact_raw) if isinstance(artifact_raw, Mapping) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": _normalize_message(record.get("artifact_id", "")),
        "status": record.get("status"),
        "started_at": _normalize_message(record.get("started_at") or ""),
        "finished_at": _normalize_message(record.get("finished_at") or ""),
        "runtime": dict(runtime) if isinstance(runtime, Mapping) else {},
        "input": {
            "snapshot_id": (
                _normalize_message(snapshot_id) if isinstance(snapshot_id, str) else None
            )
        },
        "outcome": {
            "model_family": _json_scalar(outcome_map.get("model_family")),
            "model_type": _json_scalar(outcome_map.get("model_type")),
            "promoted": bool(outcome_map.get("promoted", False)),
            "no_trade": bool(outcome_map.get("no_trade", False)),
            "selected_profile": {
                "profile_id": _json_scalar(profile.get("profile_id")),
                "no_trade_band_bps": _as_float(profile.get("no_trade_band_bps")),
            },
            "selected_horizons": [
                int(horizon)
                for horizon in (horizons if isinstance(horizons, (list, tuple)) else [])
                if isinstance(horizon, int)
            ][:8],
        },
        "observability": {},
        "artifact": reference,
        "compaction": {
            "reason": "record exceeded the terminal byte budget",
            "omitted_record_sha256": hashlib.sha256(_encode(record)).hexdigest(),
        },
    }


def _fit_record_to_limit(record: Mapping[str, object]) -> dict[str, object]:
    """Return the record when it fits, else the deterministic minimal projection.

    Applied before cache validation so a valid artifact can never fail the
    terminal write merely because its diagnostics are verbose; only real
    filesystem or malformed-artifact errors still prevent persistence.
    """
    fitted: dict[str, object] = dict(record)
    if len(_encode(fitted)) <= MAX_RECORD_BYTES:
        return fitted
    fitted = _minimal_terminal_projection(record)
    # Structural guarantee: shed optional sections in fixed order until the
    # bounded core (identity/status/artifact/compaction) fits.
    while len(_encode(fitted)) > MAX_RECORD_BYTES:
        for key in _FIT_DROP_ORDER:
            if key in fitted:
                del fitted[key]
                break
        else:
            break
    return fitted


def _project_completed(
    context: MlRunContext,
    manifest: ModelManifest,
    metrics: Mapping[str, object],
    observability: Mapping[str, object],
    clock: Callable[[], datetime],
) -> dict[str, object]:
    finished_at = clock()
    elapsed_ms = int((finished_at - context.started_at).total_seconds() * 1000)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": context.artifact_id,
        "status": "completed",
        "started_at": context.started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "runtime": {"elapsed_ms": elapsed_ms, "peak_rss_mib": peak_rss_mib()},
        "input": {
            "snapshot_id": context.snapshot_id,
            "input_ids": dict(context.input_ids) if context.input_ids else None,
            "data_lineage": (
                context.data_lineage.to_json()
                if context.data_lineage is not None
                else None
            ),
            "request": _project_request(context.request),
            "data": _project_data(context),
            "cost_context": _project_cost_context(context.cost_context),
        },
        "outcome": _project_outcome(manifest, metrics),
        "observability": {
            "phases": _sanitize_deep(observability.get("phases")),
            "horizons": _sanitize_deep(observability.get("horizons")),
            "summary": _sanitize_deep(observability.get("summary")),
            "policy_frontier": _sanitize_deep(
                observability.get("policy_frontier")
            ),
            "growth_route": _sanitize_deep(
                observability.get("growth_route")
            ),
            "replay": _project_replay(metrics),
            "holdout": _project_holdout(metrics),
        },
        "artifact": {
            "manifest_path": MANIFEST_FILENAME,
            "metrics_path": METRICS_FILENAME,
        },
    }


def _project_failed(
    context: MlRunContext,
    phase: str,
    exc: BaseException,
    telemetry: Mapping[str, object] | None,
    clock: Callable[[], datetime],
) -> dict[str, object]:
    finished_at = clock()
    elapsed_ms = int((finished_at - context.started_at).total_seconds() * 1000)
    phases = _compact_phases(telemetry if isinstance(telemetry, Mapping) else {})
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": context.artifact_id,
        "status": "failed",
        "started_at": context.started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "runtime": {"elapsed_ms": elapsed_ms, "peak_rss_mib": peak_rss_mib()},
        "input": {
            "snapshot_id": context.snapshot_id,
            "input_ids": dict(context.input_ids) if context.input_ids else None,
            "request": _project_request(context.request),
            "data": _project_data(context),
            "cost_context": _project_cost_context(context.cost_context),
        },
        "outcome": {},
        "observability": {
            "phases": phases,
            "horizons": [],
            "summary": {},
            "replay": {},
            "holdout": {},
        },
        "failure": {
            "phase": _normalize_message(phase),
            "exception_type": type(exc).__name__,
            "message": _normalize_message(str(exc) or type(exc).__name__),
        },
        "artifact": {},
    }


def _project_reconcile_record(
    artifact_id: str,
    manifest_json: Mapping[str, object],
    metrics: Mapping[str, object],
    *,
    mtime: str | None = None,
    artifact_reference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    model_type = str(
        metrics.get("model_type") or manifest_json.get("model_type") or "no_trade"
    )
    manifest = ModelManifest(
        artifact_id=artifact_id,
        asset_kind=AssetKind(str(manifest_json.get("asset_kind") or "STOCK")),
        feature_set=str(manifest_json.get("feature_set") or ""),
        feature_schema_hash=str(manifest_json.get("feature_schema_hash") or ""),
        universe_policy_hash=str(manifest_json.get("universe_policy_hash") or ""),
        label_definition=str(manifest_json.get("label_definition") or ""),
        label_horizon_sessions=_as_int(manifest_json.get("label_horizon_sessions")),
        eligible_from=str(manifest_json.get("eligible_from") or ""),
        eligible_to=str(manifest_json.get("eligible_to") or ""),
        model_type=model_type,
    )
    observability = _compact_observability(metrics, None)
    finished_at = (
        metrics.get("finished_at")
        or manifest_json.get("finished_at")
        or mtime
    )
    reference: dict[str, object] = {
        "manifest_path": MANIFEST_FILENAME,
        "metrics_path": METRICS_FILENAME,
    }
    if artifact_reference is not None:
        reference.update(artifact_reference)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "status": "completed",
        "started_at": manifest_json.get("eligible_from"),
        "finished_at": str(finished_at) if finished_at else None,
        "runtime": {"elapsed_ms": None, "peak_rss_mib": None},
        "input": {
            "snapshot_id": None,
            "request": {},
            "data": {},
            "cost_context": {},
        },
        "outcome": _project_outcome(manifest, metrics),
        "observability": {
            "phases": observability["phases"],
            "horizons": observability["horizons"],
            "summary": observability["summary"],
            "policy_frontier": observability["policy_frontier"],
            "growth_route": observability.get("growth_route", {}),
            "replay": _project_replay(metrics),
            "holdout": _project_holdout(metrics),
        },
        "artifact": reference,
    }


class ResultLedgerObserver(Protocol):
    """Optional observer injected into programmatic training workflows."""

    def record_completed(
        self,
        context: MlRunContext,
        manifest: ModelManifest,
        registry: ModelArtifactRegistry,
        telemetry: Mapping[str, object] | None = None,
        diagnostic_report: object | None = None,
    ) -> None: ...

    def record_failed(
        self,
        context: MlRunContext,
        phase: str,
        exc: BaseException,
        telemetry: Mapping[str, object] | None = None,
    ) -> None: ...


class NullResultLedger:
    """No-op observer that never touches repository ``docs``."""

    def record_completed(
        self,
        context: MlRunContext,
        manifest: ModelManifest,
        registry: ModelArtifactRegistry,
        telemetry: Mapping[str, object] | None = None,
        diagnostic_report: object | None = None,
    ) -> None:
        del context, manifest, registry, telemetry, diagnostic_report

    def record_failed(
        self,
        context: MlRunContext,
        phase: str,
        exc: BaseException,
        telemetry: Mapping[str, object] | None = None,
    ) -> None:
        del context, phase, exc, telemetry


class MlResultLedger:
    """Bounded filesystem ledger under ``<project>/docs/results``."""

    def __init__(
        self,
        results_root: Path,
        *,
        retained_records: int = RETAINED_RECORDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = Path(results_root)
        self.runs_root = self.root / "ml_runs"
        self.retained_records = retained_records
        self._clock = clock or _utc_now

    def record_completed(
        self,
        context: MlRunContext,
        manifest: ModelManifest,
        registry: ModelArtifactRegistry,
        telemetry: Mapping[str, object] | None = None,
        diagnostic_report: object | None = None,
    ) -> None:
        """Project and atomically record one completed terminal run."""
        metrics = _read_artifact_json(registry, context.artifact_id, METRICS_FILENAME)
        manifest_json = _read_artifact_json(
            registry, context.artifact_id, MANIFEST_FILENAME
        )
        _validate_artifact_identity(context, manifest, manifest_json)
        reference = _artifact_reference(Path(registry.root), context.artifact_id)
        observability = _observability_from(metrics, telemetry)
        record = _project_completed(
            context, manifest, metrics, observability, self._clock
        )
        record["artifact"] = reference
        if diagnostic_report is not None:
            if hasattr(diagnostic_report, "to_json"):
                record["diagnostic_report"] = diagnostic_report.to_json()
            else:
                record["diagnostic_report"] = diagnostic_report
        self._write_record(context.artifact_id, record)

    def record_failed(
        self,
        context: MlRunContext,
        phase: str,
        exc: BaseException,
        telemetry: Mapping[str, object] | None = None,
    ) -> None:
        """Record a terminal failure without hiding the original exception."""
        record = _project_failed(context, phase, exc, telemetry, self._clock)
        self._write_record(context.artifact_id, record)

    def rebuild_from_registry(self, registry: ModelArtifactRegistry) -> dict[str, int]:
        """Rebuild the bounded cache from published artifact manifests/metrics.

        Recovery is best-effort: reconstructed records carry no runtime or
        request context, only the evidence persisted in the artifacts.
        """
        root = Path(registry.root)
        if not root.exists():
            raise ValueError(f"registry root missing: {root}")
        records: list[dict[str, object]] = []
        for artifact_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            artifact_id = artifact_dir.name
            if not ARTIFACT_ID_RE.match(artifact_id):
                continue
            manifest_path = artifact_dir / MANIFEST_FILENAME
            metrics_path = artifact_dir / METRICS_FILENAME
            if not manifest_path.exists() or not metrics_path.exists():
                continue
            try:
                manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(manifest_json, dict) or not isinstance(metrics, dict):
                continue
            if manifest_json.get("artifact_id") != artifact_id:
                continue
            mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime, tz=UTC).isoformat()
            records.append(
                _project_reconcile_record(
                    artifact_id,
                    manifest_json,
                    metrics,
                    mtime=mtime,
                    artifact_reference=_artifact_reference(root, artifact_id),
                )
            )
        ordered = sorted(records, key=_sort_key, reverse=True)
        retained = ordered[: self.retained_records]
        discarded = len(ordered) - len(retained)
        self._write_cache(retained, discarded, invalid=0)
        if retained:
            pointer = _build_pointer(
                retained[0], len(retained), discarded, 0, self.retained_records
            )
            pointer_path = _safe_resolve(self.root, POINTER_FILENAME)
            _write_atomic(pointer_path, pointer.encode("utf-8"))
        return {
            "scanned": len(records),
            "retained": len(retained),
            "discarded": discarded,
        }

    def _write_record(self, artifact_id: str, record: dict[str, object]) -> None:
        if not ARTIFACT_ID_RE.match(artifact_id):
            raise ValueError(f"invalid artifact_id {artifact_id!r}")
        if record.get("artifact_id") != artifact_id:
            raise ValueError("record artifact id does not match key")
        fitted = _fit_record_to_limit(record)
        _validate_record(fitted)
        runs_root = self.runs_root
        runs_root.mkdir(parents=True, exist_ok=True)
        recent_path = _safe_resolve(runs_root, RECENT_FILENAME)
        retained, discarded, invalid = _rebuild_recent(
            recent_path, artifact_id, fitted, self.retained_records
        )
        self._write_cache(retained, discarded, invalid)
        pointer = _build_pointer(
            fitted, len(retained), discarded, invalid, self.retained_records
        )
        pointer_path = _safe_resolve(self.root, POINTER_FILENAME)
        pointer_bytes = pointer.encode("utf-8")
        if len(pointer_bytes) > MAX_POINTER_BYTES:
            raise ValueError(f"generated pointer exceeds {MAX_POINTER_BYTES} bytes")
        _write_atomic(pointer_path, pointer_bytes)

    def _write_cache(
        self,
        retained: list[dict[str, object]],
        discarded: int,
        invalid: int,
    ) -> None:
        latest_path = _safe_resolve(self.runs_root, LATEST_FILENAME)
        recent_path = _safe_resolve(self.runs_root, RECENT_FILENAME)
        meta_path = _safe_resolve(self.runs_root, META_FILENAME)
        schema_path = _safe_resolve(self.runs_root, SCHEMA_FILENAME)
        prior_discarded, prior_invalid = _read_prior_counts(meta_path)
        if retained:
            _validate_size(retained[0], MAX_LATEST_BYTES, "latest.json")
            _write_atomic(latest_path, _encode(retained[0]))
        _write_atomic(
            recent_path, b"".join(_encode(record) + b"\n" for record in retained)
        )
        meta = _build_meta(
            self._clock(),
            len(retained),
            prior_discarded + discarded,
            prior_invalid + invalid,
            self.retained_records,
        )
        _validate_size(meta, MAX_META_BYTES, "ledger_meta.json")
        _write_atomic(meta_path, _encode(meta))
        refresh_schema = True
        if schema_path.exists():
            try:
                stored = json.loads(schema_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, ValueError):
                stored = None
            refresh_schema = not (
                isinstance(stored, dict)
                and stored.get("schema_version") == SCHEMA_VERSION
            )
        if refresh_schema:
            schema = _build_schema(self.retained_records)
            _validate_size(schema, MAX_SCHEMA_BYTES, "schema.json")
            _write_atomic(schema_path, _encode(schema))
