"""Gold 검증 구간 bounded Parquet loading."""
from __future__ import annotations

import hashlib
import json
import zoneinfo
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from src.core.datasets import validate_dataset_manifest
from src.core.instruments import AssetKind
from src.core.time import KRX_TZ, SessionCalendar
from src.data.gold import WARMUP_SESSIONS
from src.data.schemas import PITDataError, SilverTable
from src.data.silver import load_latest_silver_table
from src.features.contracts import QvefFeaturePolicy
from src.storage.parquet_datasets import ParquetDatasetStore
from src.strategy.universe import UniversePolicy

_DAILY_MARKET_COLUMNS = [
    "session",
    "instrument_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trading_value",
    "market_cap",
    "shares_outstanding",
    "available_at",
    "source_hash",
]

_INVESTOR_FLOW_COLUMNS = [
    "session",
    "instrument_id",
    "foreign_buy_value",
    "foreign_sell_value",
    "foreign_net_value",
    "institution_net_value",
    "retail_net_value",
    "available_at",
    "source_hash",
]

_SECURITY_MASTER_COLUMNS = [
    "instrument_id",
    "ticker",
    "company_id",
    "market",
    "sector",
    "listing_date",
    "delisting_date",
    "share_class",
    "status",
    "valid_from",
    "valid_to",
    "available_at",
    "source_hash",
]

_FINANCIAL_FACTS_COLUMNS = [
    "company_id",
    "fiscal_period",
    "filing_id",
    "fact",
    "published_at",
    "available_at",
    "value",
    "unit",
    "consolidated",
    "restatement_id",
    "source_hash",
    "source_kind",
    "mapping_version",
    "raw_document_hash",
]

_CORPORATE_ACTIONS_COLUMNS = [
    "instrument_id",
    "effective_date",
    "coverage_end",
    "action_id",
    "type",
    "factor",
    "cash_amount",
    "source",
    "available_at",
    "source_hash",
]

_LIFECYCLE_EVENTS_COLUMNS = [
    "instrument_id",
    "ticker",
    "event_type",
    "published_at",
    "available_at",
    "cleanup_start",
    "cleanup_end",
    "last_tradable_session",
    "delisting_date",
    "cash_settlement_per_share",
    "source_url",
    "source_hash",
    "evidence_status",
    "evidence_reason",
    "lifecycle_event_id",
    "source_security_id",
    "successor_allocations_json",
    "successor_delivery_date",
    "source_provider",
    "document_receipt_no",
    "document_sha256",
    "resolution_kind",
]


def assert_common_silver_coverage(
    *,
    coverage_ends: Mapping[SilverTable, date],
    required_tables: frozenset[SilverTable],
    required_end: date,
) -> date:
    """Assert every required Silver table shares certified coverage at or after ``required_end``.

    Every required table must carry a selected immutable manifest end. Returns
    the common (minimum) coverage end.
    """
    common: date | None = None
    for table in sorted(required_tables, key=lambda item: item.value):
        end = coverage_ends.get(table)
        if end is None:
            raise PITDataError(f"missing certified Silver coverage for {table.value}")
        if common is None or end < common:
            common = end
    if common is None:
        return required_end
    for table in sorted(required_tables, key=lambda item: item.value):
        end = coverage_ends.get(table)
        if end is None:  # pragma: no cover - rejected in the complete first pass
            raise PITDataError(f"missing certified Silver coverage for {table.value}")
        if end < required_end:
            raise PITDataError(
                f"insufficient certified Silver coverage for {table.value}: "
                f"ends {end.isoformat()}, required {required_end.isoformat()}"
            )
    return common


def _selected_manifest_time_end(
    *, silver_root: Path, table: SilverTable, decision_time: datetime, dataset_id: str | None = None
) -> date:
    """Return the selected immutable manifest coverage end without scanning rows."""
    dataset_id, store = (_resolve_latest_dataset(silver_root, table) if dataset_id is None else (dataset_id, ParquetDatasetStore(Path(silver_root) / table.value)))
    try:
        manifest = store.read_manifest(dataset_id)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise PITDataError(f"invalid certified Silver table: {table.value}") from exc
    time_end = getattr(manifest, "time_end", None)
    if not isinstance(time_end, datetime) or time_end.tzinfo is None:
        raise PITDataError(f"invalid certified Silver table: {table.value}")
    # Manifest boundaries are stored as UTC midnight for a market *date*.
    # Converting that midnight to KST would move the boundary to the prior
    # date (e.g. 2026-09-09T00:00Z -> 2026-09-08 KST).
    return time_end.date()


@dataclass(frozen=True, slots=True)
class GoldWindowInputs:
    calendar: SessionCalendar
    security_master: pl.DataFrame
    daily_market: pl.DataFrame
    financial_facts: pl.DataFrame
    corporate_actions: pl.DataFrame
    investor_flow: pl.DataFrame
    silver_dataset_ids: Mapping[SilverTable, str] | None = None


def parse_silver_dataset_bindings(values: Sequence[str]) -> dict[SilverTable, str]:
    """Parse explicit TABLE=DATASET_ID bindings for every SilverTable."""
    bindings: dict[SilverTable, str] = {}
    for token in values:
        table_name, sep, raw_id = token.partition("=")
        if not sep or not table_name.strip() or not raw_id.strip():
            raise PITDataError(f"malformed silver dataset binding blank table or id: {token!r}")
        clean_name = table_name.strip()
        dataset_id = raw_id.strip()
        try:
            table = SilverTable(clean_name)
        except ValueError:
            raise PITDataError(f"unknown silver table: {clean_name!r}") from None
        if table in bindings:
            raise PITDataError(f"duplicate silver table binding: {clean_name!r}")
        bindings[table] = dataset_id
    if set(bindings) != set(SilverTable):
        missing = sorted(t.value for t in set(SilverTable) - set(bindings))
        raise PITDataError(f"missing silver dataset bindings: {missing}")
    return bindings


def resolve_gold_dataset_bindings(
    *, silver_root: Path, requested: Mapping[SilverTable, str], decision_time: datetime
) -> dict[SilverTable, str]:
    """Validate explicit bindings against manifest lineage before any scan."""
    if decision_time.tzinfo is None:
        raise PITDataError("invalid certified Silver table: decision_time must be aware")
    resolved: dict[SilverTable, str] = {}
    for table in SilverTable:
        dataset_id = requested.get(table) if isinstance(requested, Mapping) else None
        table_root = Path(silver_root) / table.value
        dataset_dir = table_root / dataset_id if dataset_id else table_root / "__absent__"
        if dataset_id is None or not dataset_dir.is_dir():
            raise PITDataError(f"missing silver dataset directory for {table.value}: {dataset_id!r}")
        resolved[table] = dataset_id
    for table in SilverTable:
        dataset_id = resolved[table]
        table_root = Path(silver_root) / table.value
        manifest = ParquetDatasetStore(table_root).read_manifest(dataset_id)
        generated = getattr(manifest, "generated_time", None)
        time_end = getattr(manifest, "time_end", None)
        generated_ok = isinstance(generated, datetime) and generated.tzinfo is not None
        time_end_ok = isinstance(time_end, datetime) and time_end.tzinfo is not None
        if manifest.provider_version == "fixture" or not generated_ok or not time_end_ok:
            raise PITDataError(
                f"fixture silver dataset or naive time rejected (must be aware) for {table.value}: {dataset_id!r}"
            )
    return resolved


def write_gold_input_binding_artifact(
    *,
    artifact_root: Path,
    dataset_ids: Mapping[SilverTable, str],
    decision_time: datetime,
    validation_start: date,
    validation_end: date,
) -> Path:
    """Persist the immutable explicit input binding artifact."""
    if decision_time.tzinfo is None:
        raise PITDataError("invalid certified Silver table: decision_time must be aware")
    if validation_start > validation_end:
        raise PITDataError("invalid certified Silver table: validation range inverted")
    if set(dataset_ids) != set(SilverTable):
        missing = sorted(t.value for t in set(SilverTable) - set(dataset_ids))
        raise PITDataError(f"missing silver dataset bindings: {missing}")
    ordered = sorted(dataset_ids, key=lambda item: item.value)
    payload = {
        "dataset_ids": {table.value: dataset_ids[table] for table in ordered},
        "decision_time": decision_time.isoformat(),
        "validation_end": validation_end.isoformat(),
        "validation_start": validation_start.isoformat(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    out_dir = Path(artifact_root) / "gold_input_bindings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{digest}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def apply_lifecycle_master_overlay(
    *, security_master: pl.DataFrame, lifecycle_events: pl.DataFrame
) -> pl.DataFrame:
    """Overlay verified lifecycle delisting dates onto known-market master rows.

    The unknown derived row cannot replace market, sector, listing_date, or
    ordinary availability fields and is never a PIT admission source.
    """
    if security_master.is_empty() or lifecycle_events.is_empty():
        return security_master
    if "instrument_id" not in set(security_master.columns):
        return security_master
    available = set(lifecycle_events.columns)
    scoped = lifecycle_events
    if "evidence_status" in available and lifecycle_events.schema.get("evidence_status") == pl.String:
        scoped = scoped.filter(pl.col("evidence_status") == "verified")
    if scoped.is_empty():
        return security_master
    keys = scoped.select("instrument_id", "delisting_date").unique(maintain_order=True)
    joined = security_master.join(keys, on="instrument_id", how="left")
    return joined.with_columns(
        pl.coalesce(pl.col("delisting_date_right"), pl.col("delisting_date")).alias("delisting_date")
    ).drop("delisting_date_right")


def _compact_master_snapshots(frame: pl.DataFrame) -> pl.DataFrame:
    """Retain one earliest PIT snapshot for each unchanged master state."""
    semantic = [
        "instrument_id", "ticker", "company_id", "market", "sector",
        "delisting_date", "share_class", "status", "valid_to",
    ]
    aggregate_columns = {"listing_date", "valid_from", "available_at", "source_hash"}
    if frame.is_empty() or any(
        column not in frame.columns for column in [*semantic, *aggregate_columns]
    ):
        return frame
    aggregates = [
        pl.col("listing_date").min().alias("listing_date"),
        pl.col("valid_from").min().alias("valid_from"),
        pl.col("available_at").min().alias("available_at"),
        pl.col("source_hash").first().alias("source_hash"),
    ]
    return frame.group_by(semantic, maintain_order=True).agg(aggregates).select(frame.columns)


def _align_session_dates(frame: pl.DataFrame) -> pl.DataFrame:
    """Use the certified KRX trading-date key, not the bar publication hour."""
    if frame.is_empty() or "session" not in frame.columns:
        return frame
    return frame.with_columns(pl.col("session").dt.truncate("1d").alias("session"))


def _to_krx_date(value: object) -> date:
    if isinstance(value, datetime):
        tz = value.tzinfo
        return value.astimezone(KRX_TZ).date() if tz is not None else value.date()
    if isinstance(value, date):
        return value
    raise PITDataError(f"invalid certified Silver table: bad session value {type(value)}")


def _resolve_latest_dataset(
    silver_root: Path, table: SilverTable
) -> tuple[str, ParquetDatasetStore]:
    table_root = Path(silver_root) / table.value
    if not table_root.exists():
        raise PITDataError(f"invalid certified Silver table: missing {table.value}")
    candidates = [p for p in table_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not candidates:
        raise PITDataError(f"invalid certified Silver table: missing {table.value}")
    store = ParquetDatasetStore(table_root)
    best_id: str | None = None
    best_key: tuple[datetime, datetime, str] | None = None
    for cand in candidates:
        try:
            manifest = store.read_manifest(cand.name)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise PITDataError(f"invalid certified Silver table: {table.value}") from exc
        generated = getattr(manifest, "generated_time", None)
        if not isinstance(generated, datetime) or generated.tzinfo is None:
            raise PITDataError(f"invalid certified Silver table: {table.value}")
        # Coverage-versioned immutable datasets intentionally retain the same
        # row content hash.  Use the dataset id as a deterministic tie-breaker
        # so the newer coverage publication is selected.
        manifest_end = getattr(manifest, "time_end", None)
        if not isinstance(manifest_end, datetime) or manifest_end.tzinfo is None:
            # Minimal contract fixtures may omit coverage metadata; generated
            # time remains a deterministic fallback for dataset selection.
            manifest_end = generated
        key = (generated, manifest_end, cand.name)
        if best_key is None or key > best_key:
            best_key = key
            best_id = cand.name
    if best_id is None:  # pragma: no cover - candidates is fail-closed above
        raise PITDataError(f"invalid certified Silver table: {table.value}")
    return best_id, store


def _feature_set(table: SilverTable) -> str:
    return f"stock_pit_{table.value}_v1"


def _load_manifest_silver_table(
    *,
    silver_root: Path,
    table: SilverTable,
    decision_time: datetime,
    columns: list[str],
    dataset_id: str | None = None,
) -> pl.DataFrame:
    """Load a manifest-bound Silver table without optional skips."""
    return _read_full_projected(
        silver_root=Path(silver_root),
        table=table,
        decision_time=decision_time,
        columns=columns,
        dataset_id=dataset_id,
    )


def _read_bounded_table(
    *,
    silver_root: Path,
    table: SilverTable,
    decision_time: datetime,
    session_start: date,
    session_end: date,
    columns: list[str],
    dataset_id: str | None = None,
) -> pl.DataFrame:
    dataset_id, store = (_resolve_latest_dataset(silver_root, table) if dataset_id is None else (dataset_id, ParquetDatasetStore(Path(silver_root) / table.value)))
    try:
        frame = store.read_bounded(
            dataset_id,
            AssetKind.STOCK,
            _feature_set(table),
            decision_time,
            session_start=session_start,
            session_end=session_end,
            columns=columns,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise PITDataError(f"invalid certified Silver table: {table.value}") from exc
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise PITDataError(f"invalid certified Silver table: {table.value} missing {missing}")
    return frame


def _read_full_projected(
    *,
    silver_root: Path,
    table: SilverTable,
    decision_time: datetime,
    columns: list[str],
    valid_from_end: date | None = None,
    dataset_id: str | None = None,
) -> pl.DataFrame:
    dataset_id, store = (_resolve_latest_dataset(silver_root, table) if dataset_id is None else (dataset_id, ParquetDatasetStore(Path(silver_root) / table.value)))
    try:
        if valid_from_end is None:
            frame = store.read(dataset_id, AssetKind.STOCK, _feature_set(table), decision_time)
        else:
            # Reference tables can be large (security master is multi-million
            # rows).  Verify the manifest/partition digests, then scan only
            # buckets that can contain valid_from <= validation_end.
            manifest = store.read_manifest(dataset_id)
            validate_dataset_manifest(
                manifest, AssetKind.STOCK, _feature_set(table), decision_time
            )
            paths = store.bounded_partition_paths(
                dataset_id,
                session_start=date(1900, 1, 1),
                session_end=valid_from_end,
            )
            if not paths:
                frame = pl.DataFrame({column: [] for column in columns})
            else:
                scan = pl.scan_parquet([str(path) for path in paths]).select(columns)
                if "valid_from" in columns:
                    scan = scan.filter(pl.col("valid_from").dt.date() <= valid_from_end)
                frame = scan.collect()
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise PITDataError(f"invalid certified Silver table: {table.value}") from exc
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise PITDataError(f"invalid certified Silver table: {table.value} missing {missing}")
    lazy = frame.lazy().select(columns)
    return lazy.collect()


def load_gold_window_inputs(
    *,
    silver_root: Path,
    validation_start: date,
    validation_end: date,
    decision_time: datetime,
    universe_policy: UniversePolicy | None = None,
    qvef_policy: QvefFeaturePolicy | None = None,
    silver_dataset_ids: Mapping[SilverTable, str] | None = None,
) -> GoldWindowInputs:
    if decision_time.tzinfo is None:
        raise PITDataError("invalid certified Silver table: decision_time must be aware")
    if validation_start > validation_end:
        raise PITDataError("invalid certified Silver table: validation range inverted")
    _ = qvef_policy
    u_policy = universe_policy if universe_policy is not None else UniversePolicy()
    lookback = max(int(WARMUP_SESSIONS), int(u_policy.liquidity_window_sessions), 20)
    silver_root = Path(silver_root)
    explicit_ids = (
        resolve_gold_dataset_bindings(
            silver_root=silver_root,
            requested=silver_dataset_ids,
            decision_time=decision_time,
        )
        if silver_dataset_ids is not None
        else None
    )
    # Manifest coverage certifies the immutable artifact at load time; PIT is
    # enforced below from each record's available_at at each decision session.
    certification_time = datetime.now(UTC)
    from src.data.silver import load_silver_table_by_dataset_id as _load_calendar_by_id
    _calendar_id = None if explicit_ids is None else explicit_ids.get(SilverTable.CALENDAR)
    try:
        calendar_df = load_latest_silver_table(root=silver_root, table=SilverTable.CALENDAR, decision_time=certification_time) if _calendar_id is None else _load_calendar_by_id(root=silver_root, table=SilverTable.CALENDAR, dataset_id=_calendar_id, decision_time=certification_time)
    except (PITDataError, ValueError, OSError) as exc:
        raise PITDataError("invalid certified Silver table: calendar") from exc
    if calendar_df.is_empty() or "session" not in calendar_df.columns:
        raise PITDataError("invalid certified Silver table: calendar")
    try:
        sessions = tuple(sorted(calendar_df["session"].to_list()))
    except Exception as exc:  # pragma: no cover - Polars datetime column is schema-validated
        raise PITDataError("invalid certified Silver table: calendar") from exc
    if not sessions:  # pragma: no cover - empty certified calendar is rejected upstream
        raise PITDataError("invalid certified Silver table: calendar")
    calendar = SessionCalendar(sessions)
    session_dates = [_to_krx_date(s) for s in sessions]
    val_indices = [i for i, d in enumerate(session_dates) if d >= validation_start]
    if not val_indices:
        raise PITDataError("invalid certified Silver table: calendar")
    first_val_idx = val_indices[0]
    # Guard: first validation sessions must retain warmup/liquidity history.
    if first_val_idx < lookback:
        raise PITDataError(
            "invalid certified Silver table: calendar lacks warmup history"
        )
    # Guard: validation dates themselves must never be shortened.
    last_indices = [i for i, d in enumerate(session_dates) if d <= validation_end]
    if not last_indices or max(last_indices) < first_val_idx:  # pragma: no cover - range guard
        raise PITDataError("invalid certified Silver table: calendar")
    history_start_session = sessions[first_val_idx - lookback]
    session_start = _to_krx_date(history_start_session)
    session_end = validation_end
    if session_start > validation_start:  # pragma: no cover - derived from prior index guard
        raise PITDataError("invalid certified Silver table: calendar")
    if session_end != validation_end:  # pragma: no cover - direct assignment invariant
        raise PITDataError("invalid certified Silver table: calendar")
    # Common-coverage gate over selected immutable manifests (manifest reads
    # only; no Bronze scan and no row materialization for this check).
    required_tables = frozenset(SilverTable) - {SilverTable.INVESTOR_FLOW}
    coverage_ends: dict[SilverTable, date] = {}
    for table in sorted(required_tables, key=lambda item: item.value):
        try:
            coverage_ends[table] = _selected_manifest_time_end(silver_root=silver_root, table=table, decision_time=certification_time, dataset_id=None if explicit_ids is None else explicit_ids.get(table))
        except (PITDataError, OSError, ValueError) as exc:
            raise PITDataError(f"invalid certified Silver table: {table.value}") from exc
    try:
        coverage_ends[SilverTable.INVESTOR_FLOW] = _selected_manifest_time_end(silver_root=silver_root, table=SilverTable.INVESTOR_FLOW, decision_time=certification_time, dataset_id=None if explicit_ids is None else explicit_ids.get(SilverTable.INVESTOR_FLOW))
    except (PITDataError, OSError, ValueError) as exc:
        flow_root = silver_root / SilverTable.INVESTOR_FLOW.value
        if flow_root.exists() and any(path.is_dir() and not path.name.startswith(".") for path in flow_root.iterdir()):
            raise PITDataError("invalid certified Silver table: investor_flow") from exc
    coverage_tables = required_tables | frozenset(coverage_ends.keys() & {SilverTable.INVESTOR_FLOW})
    assert_common_silver_coverage(coverage_ends=coverage_ends, required_tables=coverage_tables, required_end=validation_end)
    # Monthly partition pruning plus column projection; one final collect per table.
    daily_market = _read_bounded_table(silver_root=silver_root, table=SilverTable.DAILY_MARKET, decision_time=certification_time, session_start=session_start, session_end=session_end, columns=_DAILY_MARKET_COLUMNS, dataset_id=None if explicit_ids is None else explicit_ids.get(SilverTable.DAILY_MARKET))
    daily_market = _align_session_dates(daily_market)
    try:
        investor_flow = _read_bounded_table(silver_root=silver_root, table=SilverTable.INVESTOR_FLOW, decision_time=certification_time, session_start=session_start, session_end=session_end, columns=_INVESTOR_FLOW_COLUMNS, dataset_id=None if explicit_ids is None else explicit_ids.get(SilverTable.INVESTOR_FLOW))
        investor_flow = _align_session_dates(investor_flow)
    except PITDataError as exc:
        table_root = silver_root / SilverTable.INVESTOR_FLOW.value
        if not table_root.exists() or not [
            p for p in table_root.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]:
            investor_flow = pl.DataFrame()
        else:
            raise exc
    # Reference tables use manifest coverage start; filter only after digest validation.
    security_master_full = _read_full_projected(silver_root=silver_root, table=SilverTable.SECURITY_MASTER, decision_time=certification_time, columns=_SECURITY_MASTER_COLUMNS, valid_from_end=validation_end, dataset_id=None if explicit_ids is None else explicit_ids.get(SilverTable.SECURITY_MASTER))
    financial_facts_full = _read_full_projected(silver_root=silver_root, table=SilverTable.FINANCIAL_FACTS, decision_time=certification_time, columns=_FINANCIAL_FACTS_COLUMNS, dataset_id=None if explicit_ids is None else explicit_ids.get(SilverTable.FINANCIAL_FACTS))
    corporate_actions = _read_full_projected(silver_root=silver_root, table=SilverTable.CORPORATE_ACTIONS, decision_time=certification_time, columns=_CORPORATE_ACTIONS_COLUMNS, dataset_id=None if explicit_ids is None else explicit_ids.get(SilverTable.CORPORATE_ACTIONS))
    lifecycle_events = _load_manifest_silver_table(silver_root=silver_root, table=SilverTable.LIFECYCLE_EVENTS, decision_time=certification_time, columns=_LIFECYCLE_EVENTS_COLUMNS, dataset_id=None if explicit_ids is None else explicit_ids.get(SilverTable.LIFECYCLE_EVENTS))
    # Wiring: apply_lifecycle_master_overlay(security_master=security_master_full, lifecycle_events=lifecycle_events) before __UNKNOWN__ filtering and _compact_master_snapshots
    security_master_full = apply_lifecycle_master_overlay(security_master=security_master_full, lifecycle_events=lifecycle_events)
    # Guard: prior master/fact records needed for PIT eligibility must be kept.
    if (
        "market" in security_master_full.columns
        and not security_master_full.is_empty()
        and security_master_full["market"].dtype in (pl.String, pl.Categorical)
    ):
        known = security_master_full.filter((pl.col("market") != "__UNKNOWN__").fill_null(True))
        if not known.is_empty():
            security_master_full = known
    if "valid_from" in security_master_full.columns and not security_master_full.is_empty():
        try:
            security_master = security_master_full.filter(
                pl.col("valid_from").dt.date() <= validation_end
            )
        except Exception as exc:
            raise PITDataError("invalid certified Silver table: security_master") from exc
    else:
        security_master = security_master_full
    security_master = _compact_master_snapshots(security_master)
    if (
        not security_master.is_empty()
        and "valid_from" in security_master.columns
        and "listing_date" in security_master.columns
        and security_master["listing_date"].dtype == security_master["valid_from"].dtype
    ):
        earliest_vf = security_master.group_by("instrument_id").agg(pl.col("valid_from").min().alias("_min_vf"))
        security_master = security_master.join(earliest_vf, on="instrument_id").with_columns(
            pl.when(pl.col("listing_date") == pl.col("valid_from"))
            .then(pl.col("_min_vf"))
            .otherwise(pl.col("listing_date"))
            .alias("listing_date")
        ).drop("_min_vf")
    try:
        if "available_at" in financial_facts_full.columns and financial_facts_full["available_at"].dtype == pl.Datetime:
            dt_tz = getattr(financial_facts_full["available_at"].dtype, "time_zone", None)
            target_dt = decision_time.astimezone(zoneinfo.ZoneInfo(dt_tz)) if dt_tz else decision_time
            financial_facts = financial_facts_full.filter(pl.col("available_at") <= target_dt)
        else:
            financial_facts = financial_facts_full.filter(pl.col("available_at") <= decision_time)
    except Exception as exc:
        raise PITDataError("invalid certified Silver table: financial_facts") from exc
    # Guard: invalid manifest or missing projected column never becomes empty frame.
    for label, frame, required in (
        ("daily_market", daily_market, _DAILY_MARKET_COLUMNS),
        ("security_master", security_master, _SECURITY_MASTER_COLUMNS),
        ("financial_facts", financial_facts, _FINANCIAL_FACTS_COLUMNS),
        ("corporate_actions", corporate_actions, _CORPORATE_ACTIONS_COLUMNS),
    ):
        absent = [c for c in required if c not in frame.columns]
        if absent and not (label == "financial_facts" and frame.is_empty()):
            raise PITDataError(f"invalid certified Silver table: {label}")
        if frame.is_empty() and label in ("daily_market", "security_master", "financial_facts"):
            raise PITDataError(f"invalid certified Silver table: {label}")
    return GoldWindowInputs(calendar=calendar, security_master=security_master, daily_market=daily_market, financial_facts=financial_facts, corporate_actions=corporate_actions, investor_flow=investor_flow, silver_dataset_ids=explicit_ids)


@dataclass(frozen=True, slots=True)
class DailyMarketBackfillPlan:
    history_start: date
    validation_end: date
    covered_sessions: tuple[date, ...]
    missing_sessions: tuple[date, ...]


def plan_daily_market_backfill(
    *,
    silver_root: Path,
    validation_start: date,
    validation_end: date,
    decision_time: datetime,
    universe_policy: UniversePolicy | None = None,
) -> DailyMarketBackfillPlan:
    """Plan PIT-safe daily-market coverage from calendar and availability cutoffs."""
    from datetime import time as _time

    if decision_time.tzinfo is None:
        raise PITDataError("invalid certified Silver table: decision_time must be aware")
    if validation_start > validation_end:
        raise PITDataError("invalid certified Silver table: validation range inverted")
    u_policy = universe_policy if universe_policy is not None else UniversePolicy()
    lookback = max(int(WARMUP_SESSIONS), int(u_policy.liquidity_window_sessions), 20)
    try:
        calendar_df = load_latest_silver_table(
            root=Path(silver_root), table=SilverTable.CALENDAR, decision_time=decision_time
        )
    except (PITDataError, ValueError, OSError) as exc:
        raise PITDataError("invalid certified Silver table: calendar") from exc
    if calendar_df.is_empty() or "session" not in calendar_df.columns:
        raise PITDataError("invalid certified Silver table: calendar")
    try:
        sessions = tuple(sorted(calendar_df["session"].to_list()))
    except Exception as exc:
        raise PITDataError("invalid certified Silver table: calendar") from exc
    if not sessions:
        raise PITDataError("invalid certified Silver table: calendar")
    session_dates = [_to_krx_date(s) for s in sessions]
    val_indices = [i for i, d in enumerate(session_dates) if d >= validation_start]
    if not val_indices:
        raise PITDataError("invalid certified Silver table: calendar")
    first_val_idx = val_indices[0]
    if first_val_idx < lookback:
        raise PITDataError("invalid certified Silver table: calendar lacks warmup history")
    last_indices = [i for i, d in enumerate(session_dates) if d <= validation_end]
    if not last_indices:
        raise PITDataError("invalid certified Silver table: calendar")
    last_idx = max(last_indices)
    if session_dates[last_idx] != validation_end:
        raise PITDataError("invalid certified Silver table: calendar")
    history_start = _to_krx_date(sessions[first_val_idx - lookback])
    required = tuple(session_dates[first_val_idx - lookback : last_idx + 1])
    try:
        frame = _read_bounded_table(
            silver_root=Path(silver_root),
            table=SilverTable.DAILY_MARKET,
            decision_time=decision_time,
            session_start=history_start,
            session_end=validation_end,
            columns=["session", "available_at"],
        )
    except PITDataError:
        frame = pl.DataFrame({"session": [], "available_at": []})
    timely: dict[date, bool] = {}
    if not frame.is_empty() and "session" in frame.columns and "available_at" in frame.columns:
        for sess_val, avail_val in zip(frame["session"].to_list(), frame["available_at"].to_list(), strict=False):
            try:
                sess_date = _to_krx_date(sess_val)
            except PITDataError:
                continue
            if sess_date not in required:
                continue
            if not isinstance(avail_val, datetime) or avail_val.tzinfo is None:
                continue
            # Guard: a later-retrieved historical snapshot (e.g. 2026
            # available_at for a 2016 session) must never count as covered.
            cutoff = datetime.combine(sess_date, _time(15, 30), tzinfo=KRX_TZ)
            if avail_val <= cutoff:
                timely[sess_date] = True
    covered = tuple(d for d in required if timely.get(d, False))
    missing = tuple(d for d in required if not timely.get(d, False))
    return DailyMarketBackfillPlan(
        history_start=history_start,
        validation_end=validation_end,
        covered_sessions=covered,
        missing_sessions=missing,
    )
