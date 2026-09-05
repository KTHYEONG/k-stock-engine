"""Gold 검증 구간 bounded Parquet loading."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

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
    "action_id",
    "type",
    "factor",
    "cash_amount",
    "source",
    "available_at",
    "source_hash",
]


@dataclass(frozen=True, slots=True)
class GoldWindowInputs:
    calendar: SessionCalendar
    security_master: pl.DataFrame
    daily_market: pl.DataFrame
    financial_facts: pl.DataFrame
    corporate_actions: pl.DataFrame
    investor_flow: pl.DataFrame


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
    best_key: tuple[datetime, str] | None = None
    for cand in candidates:
        try:
            manifest = store.read_manifest(cand.name)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise PITDataError(f"invalid certified Silver table: {table.value}") from exc
        generated = getattr(manifest, "generated_time", None)
        if not isinstance(generated, datetime) or generated.tzinfo is None:
            raise PITDataError(f"invalid certified Silver table: {table.value}")
        key = (generated, str(getattr(manifest, "content_hash", "") or cand.name))
        if best_key is None or key > best_key:
            best_key = key
            best_id = cand.name
    if best_id is None:  # pragma: no cover - candidates is fail-closed above
        raise PITDataError(f"invalid certified Silver table: {table.value}")
    return best_id, store


def _feature_set(table: SilverTable) -> str:
    return f"stock_pit_{table.value}_v1"


def _read_bounded_table(
    *,
    silver_root: Path,
    table: SilverTable,
    decision_time: datetime,
    session_start: date,
    session_end: date,
    columns: list[str],
) -> pl.DataFrame:
    dataset_id, store = _resolve_latest_dataset(silver_root, table)
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
) -> pl.DataFrame:
    dataset_id, store = _resolve_latest_dataset(silver_root, table)
    try:
        frame = store.read(dataset_id, AssetKind.STOCK, _feature_set(table), decision_time)
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
) -> GoldWindowInputs:
    if decision_time.tzinfo is None:
        raise PITDataError("invalid certified Silver table: decision_time must be aware")
    if validation_start > validation_end:
        raise PITDataError("invalid certified Silver table: validation range inverted")
    _ = qvef_policy
    u_policy = universe_policy if universe_policy is not None else UniversePolicy()
    lookback = max(int(WARMUP_SESSIONS), int(u_policy.liquidity_window_sessions), 20)
    silver_root = Path(silver_root)
    # Manifest coverage certifies the immutable artifact at load time; PIT is
    # enforced below from each record's available_at at each decision session.
    certification_time = datetime.now(UTC)
    try:
        calendar_df = load_latest_silver_table(
            root=silver_root, table=SilverTable.CALENDAR, decision_time=certification_time
        )
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
    # Monthly partition pruning plus column projection; one final collect per table.
    daily_market = _read_bounded_table(
        silver_root=silver_root,
        table=SilverTable.DAILY_MARKET,
        decision_time=certification_time,
        session_start=session_start,
        session_end=session_end,
        columns=_DAILY_MARKET_COLUMNS,
    )
    daily_market = _align_session_dates(daily_market)
    try:
        investor_flow = _read_bounded_table(
            silver_root=silver_root,
            table=SilverTable.INVESTOR_FLOW,
            decision_time=certification_time,
            session_start=session_start,
            session_end=session_end,
            columns=_INVESTOR_FLOW_COLUMNS,
        )
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
    security_master_full = _read_full_projected(
        silver_root=silver_root,
        table=SilverTable.SECURITY_MASTER,
        decision_time=certification_time,
        columns=_SECURITY_MASTER_COLUMNS,
    )
    financial_facts_full = _read_full_projected(
        silver_root=silver_root,
        table=SilverTable.FINANCIAL_FACTS,
        decision_time=certification_time,
        columns=_FINANCIAL_FACTS_COLUMNS,
    )
    corporate_actions = _read_full_projected(
        silver_root=silver_root,
        table=SilverTable.CORPORATE_ACTIONS,
        decision_time=certification_time,
        columns=_CORPORATE_ACTIONS_COLUMNS,
    )
    # Guard: prior master/fact records needed for PIT eligibility must be kept.
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
    try:
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
    return GoldWindowInputs(
        calendar=calendar,
        security_master=security_master,
        daily_market=daily_market,
        financial_facts=financial_facts,
        corporate_actions=corporate_actions,
        investor_flow=investor_flow,
    )
