"""Canonical stock data curation: legacy feature files -> v2 partitioned datasets.

Curation is a pure, deterministic migration from the immutable legacy
``year=*/*_feat.parquet`` files below ``data/processed/features`` into
manifest-backed, Hive-partitioned canonical datasets below
``data/curated/stocks``. It never reads ``.env``, never mutates the legacy
source, and rejects a ``dataset_id`` that already exists so a rewrite always
produces a new dataset version.

The canonical projection is fail-closed:

- sources are read in lexical order and must share one schema;
- every source row must carry a six-digit KRX ticker (index rows such as
  ``KOSPI`` are excluded, never mixed into the stock dataset);
- forward ``target_*`` / ``label_*`` columns are never persisted;
- NaN numerics are normalized to null (missing markers); residual non-finite
  (Inf) aborts the migration; invalid OHLC rows are quarantined;
- ``observation_time``/``available_time`` are timezone-aware UTC derived from
  the KRX local close (15:30/15:31 Asia/Seoul).

Certification defaults to ``PROVISIONAL``. Any higher tier requires explicit
coverage evidence (calendar, corporate-action, cost-source hashes), and
``PRODUCTION`` additionally must pass ``validate_production_manifest``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path

import polars as pl

from src.core.datasets import (
    HIVE_PARTITION_LAYOUT,
    DatasetCertification,
    DatasetManifest,
    make_manifest,
    validate_production_manifest,
)
from src.core.instruments import AssetKind
from src.stocks.data.feature_contracts import (
    DuplicateRule,
    FeatureContractBook,
    contracts_to_json,
    feature_contract_book_from_allowlist,
    feature_set_hash,
    resolve_raw_source_names,
)
from src.stocks.data.quality import (
    CorporateActionSnapshot,
    FeatureAvailabilityRecord,
    InstrumentMasterSnapshot,
    KRXSessionCalendar,
    StockDataQualityPolicy,
    StockDataQualityReport,
    validate_canonical_stock_panel,
)
from src.stocks.research.datasets import (
    QUALITY_REASON_COLUMN,
    QUALITY_STATUS_COLUMN,
)
from src.storage.parquet_datasets import (
    CONTENT_MANIFEST_NAME,
    ParquetDatasetStore,
    canonical_content_hash,
    file_sha256,
)

logger = logging.getLogger("stocks.data.curation")

CURATION_VERSION = "curation-v2"

_REQUIRED_SOURCE_COLUMNS = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trading_value",
    "market_cap",
    "sector",
)
_OHLC_COLUMNS = ("open", "high", "low", "close")
_KRX_CLOSE_TIME = time(15, 30)
_KRX_AVAILABLE_TIME = time(15, 31)

_CANONICAL_BASE_COLUMNS = (
    "instrument_id",
    "session",
    "observation_time",
    "available_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trading_value",
    "market_cap",
    "sector",
    "action_interval_covered",
    QUALITY_STATUS_COLUMN,
    QUALITY_REASON_COLUMN,
)

# Versioned allowlist of predictor columns projected from the legacy schema
# into the ``feature__<name>`` namespace. Descriptive, identity, canonical
# OHLCV/market fields and forward targets are deliberately excluded.
FEATURE_ALLOWLIST_V1 = (
    "per",
    "pbr",
    "bps",
    "eps",
    "div",
    "roe",
    "capital_erosion_rate",
    "fluc_rate",
    "foreign_net_buy",
    "institution_net_buy",
    "individual_net_buy",
    "pension_net_buy",
    "turnover_ratio",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "revenue",
    "operating_income",
    "net_income",
    "capital",
    "bps_calc",
    "eps_calc",
    "log_return_1d",
    "log_return_5d",
    "log_return_20d",
    "log_return_60d",
    "log_return_120d",
    "volatility_20d",
    "volatility_60d",
    "disparity_5d",
    "disparity_20d",
    "disparity_60d",
    "disparity_120d",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_ratio_60d",
    "intraday_vol",
    "amihud_20d",
    "bp_ratio",
    "ep_ratio",
    "sp_ratio",
    "op_ratio",
    "debt_ratio",
    "relative_trend_score",
    "net_purchase_total",
    "np_mkt_cap",
    "np_vol",
    "np_cum_60d",
    "z_flow",
    "avg_trading_value_5d",
    "rank_volatility_20d",
    "rank_volatility_60d",
    "rank_volume_ratio_5d",
    "rank_volume_ratio_20d",
    "rank_volume_ratio_60d",
    "rank_log_return_5d",
    "rank_log_return_20d",
    "rank_log_return_60d",
    "rank_log_return_120d",
    "rank_amihud_20d",
    "rank_turnover_ratio",
    "rank_relative_trend_score",
    "rel_ep_ratio_sector",
    "rel_bp_ratio_sector",
    "rel_sp_ratio_sector",
    "rel_op_ratio_sector",
    "rel_roe_sector",
    "adtv_20d",
    "min_vol_5d",
    "overnight_ret",
    "intraday_ret",
    "ret_2_5d",
    "ret_6_20d",
    "ret_21_60d",
    "vol_regime",
    "volume_shock",
    "flow_intensity_20d",
    "trend_120d_rank",
    "vol_20d_rank",
    "flow_consensus",
    "mcap_rank",
    "sector_ret_5d",
    "vol_asymmetry_20d",
    "close_high_ratio_10d",
    "info_ratio_20d",
    "vpt_20d",
    "vix_zscore_20d",
    "total_assets_right",
    "total_liabilities_right",
    "total_equity_right",
    "revenue_right",
    "operating_income_right",
    "net_income_right",
    "capital_right",
)

_FEATURE_ALLOWLISTS: dict[str, tuple[str, ...]] = {
    "v1": FEATURE_ALLOWLIST_V1,
}

_TARGET_PREFIXES = ("target_", "label_")

BASE_PANEL_FEATURE_SET = "base_panel"
BASE_PANEL_LABEL_DEFINITION = "none"

# Raw/derived duplicate fundamental lineage resolved explicitly: the canonical
# field survives; ``_right`` alternatives are never projected for the same
# feature. Pairs without an explicit rule are rejected, never guessed.
_RIGHT_DUPLICATE_ROOTS = (
    "total_assets",
    "total_liabilities",
    "total_equity",
    "revenue",
    "operating_income",
    "net_income",
    "capital",
)


def default_duplicate_rules() -> tuple[DuplicateRule, ...]:
    return tuple(
        DuplicateRule(canonical=root, alternatives=(f"{root}_right",))
        for root in _RIGHT_DUPLICATE_ROOTS
    )


def default_feature_contract_book(version: str = "v1") -> FeatureContractBook:
    """The versioned default contract book built from the frozen allowlist."""
    return feature_contract_book_from_allowlist(
        version=version,
        allowlist=feature_allowlist(version),
        duplicate_rules=default_duplicate_rules(),
    )


def feature_allowlist(version: str) -> tuple[str, ...]:
    """Return the frozen predictor allowlist for ``version``."""
    try:
        return _FEATURE_ALLOWLISTS[version]
    except KeyError as exc:
        raise ValueError(f"unknown feature allowlist version {version!r}") from exc


@dataclass(frozen=True, slots=True)
class StockCurationRequest:
    """Immutable input contract for one curation run.

    ``dataset_id`` is an explicit versioned identifier; an existing identifier
    is rejected so a rewrite always creates a new dataset version.
    """

    dataset_id: str
    start_date: date
    end_date: date
    feature_allowlist_version: str = "v1"
    feature_set: str = "stock_alpha_v1"
    label_definition: str = "fwd_ret_5d"
    label_horizon_sessions: int = 5
    provider_version: str = "legacy-feature-files"
    universe_policy_version: str = "provisional-legacy"
    certification: DatasetCertification = DatasetCertification.PROVISIONAL
    calendar_hash: str = ""
    corporate_action_hash: str = ""
    cost_source_hash: str = ""
    instrument_master: InstrumentMasterSnapshot | None = None
    corporate_actions: CorporateActionSnapshot | None = None
    calendar: KRXSessionCalendar | None = None
    feature_availability: tuple[FeatureAvailabilityRecord, ...] = ()
    generated_time: datetime | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        if self.label_horizon_sessions <= 0:
            raise ValueError("label_horizon_sessions must be positive")


@dataclass(frozen=True, slots=True)
class CuratedDatasetResult:
    """Immutable outcome of a curation run."""

    dataset_id: str
    manifest: DatasetManifest
    content_manifest_path: Path
    partition_paths: tuple[Path, ...]
    row_count: int
    source_file_count: int
    quality_report_path: Path | None = None


def curate_legacy_feature_panel(
    source_root: Path,
    destination_root: Path,
    request: StockCurationRequest,
) -> CuratedDatasetResult:
    """Migrate legacy feature files into a new canonical partitioned dataset.

    Every source row is routed through the quality validator into one of three
    partitions: ``eligible`` (canonical predictor dataset), ``quarantined``
    (raw rows with stable reason codes), and ``non_equity`` (index/ETF rows with
    lineage for a benchmark dataset). An unknown identifier is quarantined
    rather than aborting the migration. A ``quality_report.json`` is written
    with counts, affected identifiers/files, null statistics, coverage, action
    coverage, and evidence hashes.
    """
    feature_names = feature_allowlist(request.feature_allowlist_version)
    panel, source_entries, source_rows = _read_source_panel(Path(source_root))
    in_window = _normalize_and_reject_non_finite(
        _window_and_dedupe(panel, request.start_date, request.end_date)
    )

    policy = StockDataQualityPolicy(
        certification=request.certification,
        calendar=request.calendar,
        feature_availability=request.feature_availability,
    )
    report = validate_canonical_stock_panel(
        in_window,
        request.instrument_master,
        request.corporate_actions,
        policy,
    )
    if report.eligible is None:
        raise ValueError("quality validation produced no eligible partition")

    present_features = [
        name for name in feature_names if name in panel.columns and name not in report.fully_null_columns
    ]
    canonical = _project_canonical(report.eligible, present_features)

    ordered_columns = _ordered_columns(present_features)
    canonical = canonical.select(ordered_columns).sort(["instrument_id", "session"])

    generated_time = request.generated_time or datetime.now(UTC)
    manifest = _build_manifest(request, canonical, ordered_columns, generated_time, report)
    _validate_certification_evidence(manifest)

    content_manifest: dict[str, object] = {
        "curation_version": CURATION_VERSION,
        "feature_allowlist_version": request.feature_allowlist_version,
        "generated_time": generated_time.isoformat(),
        "source": {
            "root": str(Path(source_root)),
            "file_count": len(source_entries),
            "row_count": source_rows,
            "files": source_entries,
        },
    }
    store = ParquetDatasetStore(Path(destination_root))
    dataset_dir = store.write_partitioned(
        canonical,
        dataset_id=request.dataset_id,
        manifest=manifest,
        expected_feature_set=request.feature_set,
        decision_time=generated_time,
        content_manifest=content_manifest,
    )
    _write_audit_partitions(dataset_dir, report)
    quality_report_path = _write_quality_report(dataset_dir, report, generated_time)

    partition_paths = tuple(sorted((dataset_dir / "partitions").rglob("*.parquet")))
    logger.info(
        "curated %s: %s eligible / %s quarantined / %s non-equity rows from %s source files",
        request.dataset_id,
        report.eligible_row_count,
        report.quarantined_row_count,
        report.non_equity_row_count,
        len(source_entries),
    )
    return CuratedDatasetResult(
        dataset_id=request.dataset_id,
        manifest=manifest,
        content_manifest_path=dataset_dir / CONTENT_MANIFEST_NAME,
        partition_paths=partition_paths,
        row_count=canonical.height,
        source_file_count=len(source_entries),
        quality_report_path=quality_report_path,
    )


def _read_source_panel(
    source_root: Path,
) -> tuple[pl.DataFrame, list[dict[str, object]], int]:
    files = sorted(source_root.glob("year=*/*_feat.parquet"))
    if not files:
        raise FileNotFoundError(f"no *_feat.parquet under {source_root}")

    scans: list[pl.LazyFrame] = []
    expected_schema: dict[str, pl.DataType] | None = None
    entries: list[dict[str, object]] = []
    total_rows = 0
    for path in files:
        lazy = pl.scan_parquet(path)
        schema = lazy.collect_schema()
        missing = [c for c in _REQUIRED_SOURCE_COLUMNS if c not in schema]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        if not isinstance(schema["date"], (pl.Date, pl.Datetime)):
            raise ValueError(f"{path}: unknown date semantics {schema['date']}")
        if expected_schema is None:
            expected_schema = schema
        elif schema != expected_schema:
            raise ValueError(f"{path}: schema variant vs first file")
        relative = str(path.relative_to(source_root))
        lazy = lazy.with_columns(pl.lit(relative).alias("source_file"))
        row_count = int(lazy.select(pl.len()).collect().item())
        total_rows += row_count
        entries.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "row_count": row_count,
            }
        )
        scans.append(lazy)
    return pl.concat(scans).collect(), entries, total_rows


def _window_and_dedupe(
    panel: pl.DataFrame, start_date: date, end_date: date
) -> pl.DataFrame:
    duplicates = (
        panel.group_by(["date", "ticker"]).len().filter(pl.col("len") > 1)
    )
    if not duplicates.is_empty():
        raise ValueError(f"{duplicates.height} duplicate (date, ticker) rows")
    in_window = panel.filter(
        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
    )
    if in_window.is_empty():
        raise ValueError(
            f"no legacy rows in {start_date.isoformat()}..{end_date.isoformat()}"
        )
    return in_window


def _normalize_and_reject_non_finite(panel: pl.DataFrame) -> pl.DataFrame:
    """Null NaN (missing) values, then fail closed on residual non-finite values.

    NaN is a missing-value marker from the legacy pipeline and is normalized to
    null so the row can be retained or quarantined deterministically. Inf cannot
    be a missing marker and aborts the migration.
    """
    float_columns = [c for c, dtype in panel.schema.items() if dtype.is_float()]
    out = panel
    if float_columns:
        out = panel.with_columns(
            pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
            for c in float_columns
        )
    for column in float_columns:
        if out[column].is_infinite().any():
            raise ValueError(f"non-finite numeric value in {column}")
    return out


def _project_canonical(
    eligible: pl.DataFrame, feature_names: list[str]
) -> pl.DataFrame:
    """Project an eligible partition into the canonical predictor schema.

    Identity, timestamps, quality status, and action coverage were already
    computed by the quality validator; this projects the eligible rows into the
    canonical base + namespaced feature columns and normalizes ``session`` to a
    UTC-midnight datetime (the canonical store key).
    """
    missing = [c for c in _CANONICAL_BASE_COLUMNS if c not in eligible.columns]
    if missing:
        raise ValueError(f"eligible panel missing canonical columns {missing}")

    session_utc = (
        pl.col("session")
        .cast(pl.Date)
        .dt.combine(pl.lit(time.min))
        .dt.replace_time_zone("UTC")
    )
    base = [pl.col(c) for c in _CANONICAL_BASE_COLUMNS]
    features = [pl.col(name).alias(f"feature__{name}") for name in feature_names]
    return eligible.select([*base, *features]).with_columns(
        session_utc.alias("session")
    )


def _ordered_columns(feature_names: list[str]) -> list[str]:
    return list(_CANONICAL_BASE_COLUMNS) + [
        f"feature__{name}" for name in feature_names
    ]


def _build_manifest(
    request: StockCurationRequest,
    canonical: pl.DataFrame,
    ordered_columns: list[str],
    generated_time: datetime,
    report: StockDataQualityReport,
) -> DatasetManifest:
    report_hash = report.hashes.get("quality_report", "")
    master_hash = report.hashes.get("master", "")
    return make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=ordered_columns,
        feature_set=request.feature_set,
        label_definition=request.label_definition,
        label_horizon_sessions=request.label_horizon_sessions,
        time_start=_as_utc_datetime(canonical["observation_time"].min()),
        time_end=_as_utc_datetime(canonical["observation_time"].max()),
        provider_version=request.provider_version,
        universe_policy_version=request.universe_policy_version,
        row_count=canonical.height,
        generated_time=generated_time,
        certification=request.certification,
        calendar_hash=request.calendar_hash or (request.calendar.content_hash if request.calendar else ""),
        corporate_action_hash=request.corporate_action_hash,
        cost_source_hash=request.cost_source_hash,
        master_hash=master_hash,
        quality_report_hash=report_hash,
        schema_version="v2",
        content_hash=canonical_content_hash(canonical, ordered_columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    raise ValueError(f"expected a datetime timestamp, got {value!r}")


def _validate_certification_evidence(manifest: DatasetManifest) -> None:
    if manifest.certification is DatasetCertification.PROVISIONAL:
        return
    evidence = {
        "calendar_hash": manifest.calendar_hash,
        "corporate_action_hash": manifest.corporate_action_hash,
        "cost_source_hash": manifest.cost_source_hash,
        "master_hash": manifest.master_hash,
        "quality_report_hash": manifest.quality_report_hash,
    }
    missing = [name for name, value in evidence.items() if not value]
    if missing:
        raise ValueError(
            f"{manifest.certification.value} certification requires coverage "
            f"evidence hashes, missing {missing}"
        )
    if manifest.certification is DatasetCertification.PRODUCTION:
        validate_production_manifest(manifest)


_AUDIT_COLUMNS = (
    "instrument_id",
    "session",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trading_value",
    "market_cap",
    "sector",
    QUALITY_STATUS_COLUMN,
    QUALITY_REASON_COLUMN,
)


def _audit_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Restrict a routed frame to deterministic audit columns."""
    keep = [c for c in _AUDIT_COLUMNS if c in frame.columns]
    extra = [c for c in ("source_file", "action_interval_covered") if c in frame.columns]
    return frame.select([*keep, *extra]).sort(["instrument_id", "session"])


def _write_audit_partitions(
    dataset_dir: Path, report: StockDataQualityReport
) -> None:
    """Write quarantined and non-equity audit partitions with lineage."""
    if report.quarantined is not None and report.quarantined.height:
        out_dir = dataset_dir / "bars" / "quarantined"
        out_dir.mkdir(parents=True, exist_ok=True)
        _audit_frame(report.quarantined).write_parquet(out_dir / "part-00000.parquet")
    if report.non_equity is not None and report.non_equity.height:
        out_dir = dataset_dir / "benchmarks" / "non_equity"
        out_dir.mkdir(parents=True, exist_ok=True)
        _audit_frame(report.non_equity).write_parquet(out_dir / "part-00000.parquet")


def _write_quality_report(
    dataset_dir: Path,
    report: StockDataQualityReport,
    generated_time: datetime,
) -> Path:
    """Write a deterministic ``quality_report.json`` next to the dataset."""
    report = report.with_generated_time(generated_time)
    report_path = dataset_dir / "quality_report.json"
    payload = json.dumps(report.to_json_dict(), indent=2, sort_keys=True, default=str)
    report_path.write_text(payload, encoding="utf-8")
    return report_path


_BASE_RESERVED_COLUMNS = frozenset(
    (
        "date",
        "ticker",
        "source_file",
        "name",
        "market",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trading_value",
        "market_cap",
        "sector",
        "instrument_id",
        "session",
        "observation_time",
        "available_time",
        "action_interval_covered",
        QUALITY_STATUS_COLUMN,
        QUALITY_REASON_COLUMN,
    )
)
_BASE_RAW_COLUMNS = (
    "instrument_id",
    "session",
    "observation_time",
    "available_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trading_value",
    "market_cap",
    "sector",
    "action_interval_covered",
    QUALITY_STATUS_COLUMN,
    QUALITY_REASON_COLUMN,
)


@dataclass(frozen=True, slots=True)
class BasePanelRequest:
    """Immutable input contract for one base-panel build.

    The base panel is the immutable, hash-bound store of stable reusable facts
    plus the deduplicated raw source predictor fields. It never carries forward
    ``target_*``/``label_*`` columns.
    """

    dataset_id: str
    start_date: date
    end_date: date
    provider_version: str = "legacy-feature-files"
    universe_policy_version: str = "provisional-legacy"
    certification: DatasetCertification = DatasetCertification.PROVISIONAL
    calendar_hash: str = ""
    corporate_action_hash: str = ""
    cost_source_hash: str = ""
    master_hash: str = ""
    duplicate_rules: tuple[DuplicateRule, ...] = ()
    instrument_master: InstrumentMasterSnapshot | None = None
    corporate_actions: CorporateActionSnapshot | None = None
    calendar: KRXSessionCalendar | None = None
    feature_availability: tuple[FeatureAvailabilityRecord, ...] = ()
    generated_time: datetime | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")


@dataclass(frozen=True, slots=True)
class BasePanelResult:
    """Immutable outcome of a base-panel build."""

    dataset_id: str
    manifest: DatasetManifest
    partition_paths: tuple[Path, ...]
    row_count: int
    raw_column_count: int
    source_file_count: int
    quality_report_path: Path | None = None


def build_base_panel(
    source_root: Path,
    destination_root: Path,
    request: BasePanelRequest,
) -> BasePanelResult:
    """Build an immutable base panel from the legacy feature source.

    Reuses the deterministic source read, windowing, NaN normalization, and
    quality routing of legacy curation, then projects the canonical stable
    facts plus ``raw__*`` source predictors. The panel is written to the
    canonical root with a manifest bound to the supplied evidence hashes. The
    legacy source and existing curated datasets are never modified.
    """
    panel, source_entries, source_rows = _read_source_panel(Path(source_root))
    in_window = _normalize_and_reject_non_finite(
        _window_and_dedupe(panel, request.start_date, request.end_date)
    )
    policy = StockDataQualityPolicy(
        certification=request.certification,
        calendar=request.calendar,
        feature_availability=request.feature_availability,
    )
    report = validate_canonical_stock_panel(
        in_window,
        request.instrument_master,
        request.corporate_actions,
        policy,
    )
    if report.eligible is None:
        raise ValueError("quality validation produced no eligible partition")

    raw_columns = _raw_source_columns(report.eligible, request.duplicate_rules)
    canonical = _project_base_panel(report.eligible, raw_columns)
    ordered_columns = _BASE_RAW_COLUMNS + tuple(
        f"raw__{name}" for name in raw_columns
    )
    canonical = canonical.select(ordered_columns).sort(["instrument_id", "session"])

    generated_time = request.generated_time or datetime.now(UTC)
    manifest = _build_base_manifest(request, canonical, ordered_columns, generated_time, report)
    _validate_certification_evidence(manifest)

    content_manifest: dict[str, object] = {
        "base_panel_version": CURATION_VERSION,
        "generated_time": generated_time.isoformat(),
        "source": {
            "root": str(Path(source_root)),
            "file_count": len(source_entries),
            "row_count": source_rows,
            "files": source_entries,
        },
    }
    store = ParquetDatasetStore(Path(destination_root))
    dataset_dir = store.write_partitioned(
        canonical,
        dataset_id=request.dataset_id,
        manifest=manifest,
        expected_feature_set=BASE_PANEL_FEATURE_SET,
        decision_time=generated_time,
        content_manifest=content_manifest,
    )
    _write_audit_partitions(dataset_dir, report)
    quality_report_path = _write_quality_report(dataset_dir, report, generated_time)
    partition_paths = tuple(sorted((dataset_dir / "partitions").rglob("*.parquet")))
    logger.info(
        "base panel %s: %s rows, %s raw columns from %s source files",
        request.dataset_id,
        canonical.height,
        len(raw_columns),
        len(source_entries),
    )
    return BasePanelResult(
        dataset_id=request.dataset_id,
        manifest=manifest,
        partition_paths=partition_paths,
        row_count=canonical.height,
        raw_column_count=len(raw_columns),
        source_file_count=len(source_entries),
        quality_report_path=quality_report_path,
    )


def _raw_source_columns(
    eligible: pl.DataFrame, duplicate_rules: tuple[DuplicateRule, ...]
) -> tuple[str, ...]:
    """Deterministic, deduplicated raw predictor columns from the source."""
    candidates = tuple(
        column
        for column in eligible.columns
        if column not in _BASE_RESERVED_COLUMNS
        and not column.startswith(_TARGET_PREFIXES)
    )
    return resolve_raw_source_names(candidates, duplicate_rules)


def _project_base_panel(
    eligible: pl.DataFrame, raw_columns: tuple[str, ...]
) -> pl.DataFrame:
    missing = [c for c in _BASE_RAW_COLUMNS if c not in eligible.columns]
    if missing:
        raise ValueError(f"eligible panel missing base columns {missing}")
    session_utc = (
        pl.col("session")
        .cast(pl.Date)
        .dt.combine(pl.lit(time.min))
        .dt.replace_time_zone("UTC")
    )
    base = [pl.col(c) for c in _BASE_RAW_COLUMNS]
    raw = [pl.col(name).alias(f"raw__{name}") for name in raw_columns]
    return eligible.select([*base, *raw]).with_columns(session_utc.alias("session"))


def _build_base_manifest(
    request: BasePanelRequest,
    canonical: pl.DataFrame,
    ordered_columns: tuple[str, ...],
    generated_time: datetime,
    report: StockDataQualityReport,
) -> DatasetManifest:
    report_hash = report.hashes.get("quality_report", "")
    return make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=list(ordered_columns),
        feature_set=BASE_PANEL_FEATURE_SET,
        label_definition=BASE_PANEL_LABEL_DEFINITION,
        label_horizon_sessions=1,
        time_start=_as_utc_datetime(canonical["observation_time"].min()),
        time_end=_as_utc_datetime(canonical["observation_time"].max()),
        provider_version=request.provider_version,
        universe_policy_version=request.universe_policy_version,
        row_count=canonical.height,
        generated_time=generated_time,
        certification=request.certification,
        calendar_hash=request.calendar_hash
        or (request.calendar.content_hash if request.calendar else ""),
        corporate_action_hash=request.corporate_action_hash,
        cost_source_hash=request.cost_source_hash,
        master_hash=request.master_hash or report.hashes.get("master", ""),
        quality_report_hash=report_hash,
        schema_version="v2",
        content_hash=canonical_content_hash(canonical, list(ordered_columns)),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )


@dataclass(frozen=True, slots=True)
class FeaturePanelRequest:
    """Immutable input contract for one feature-panel projection."""

    dataset_id: str
    base_panel_id: str
    feature_set: str = "stock_alpha_v1"
    feature_contract_book: FeatureContractBook | None = None
    provider_version: str = "base-panel"
    universe_policy_version: str = "provisional-legacy"
    certification: DatasetCertification = DatasetCertification.PROVISIONAL
    generated_time: datetime | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if not self.base_panel_id:
            raise ValueError("base_panel_id must be non-empty")


@dataclass(frozen=True, slots=True)
class FeaturePanelResult:
    """Immutable outcome of a feature-panel projection."""

    dataset_id: str
    manifest: DatasetManifest
    contract_path: Path
    partition_paths: tuple[Path, ...]
    row_count: int
    base_panel_id: str


def build_feature_panel(
    base_root: Path,
    destination_root: Path,
    request: FeaturePanelRequest,
) -> FeaturePanelResult:
    """Project a reusable, label-free wide feature panel from one base version.

    The base panel is read in full once and projected through the feature
    contract book into ``feature__*`` columns. ``feature_contract.json`` is
    written next to the dataset and records the exact contract version, the
    feature-set hash, and the referenced base-panel id.
    """
    book = request.feature_contract_book or default_feature_contract_book()
    generated_time = request.generated_time or datetime.now(UTC)
    store = ParquetDatasetStore(Path(base_root))
    base_manifest = store.read_manifest(request.base_panel_id)
    base_frame = store.read(
        request.base_panel_id, AssetKind.STOCK, BASE_PANEL_FEATURE_SET, generated_time
    )
    projected = book.project(base_frame, source_prefix="raw__")
    if projected.is_empty():
        raise ValueError("feature projection produced no rows")
    if any(c.startswith(_TARGET_PREFIXES) for c in projected.columns):
        raise ValueError("feature projection leaked a target/label column")

    ordered_columns = projected.columns
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=ordered_columns,
        feature_set=request.feature_set,
        label_definition=BASE_PANEL_LABEL_DEFINITION,
        label_horizon_sessions=1,
        time_start=_as_utc_datetime(base_frame["observation_time"].min()),
        time_end=_as_utc_datetime(base_frame["observation_time"].max()),
        provider_version=request.provider_version,
        universe_policy_version=request.universe_policy_version,
        row_count=projected.height,
        generated_time=generated_time,
        certification=request.certification,
        calendar_hash=base_manifest.calendar_hash,
        corporate_action_hash=base_manifest.corporate_action_hash,
        cost_source_hash=base_manifest.cost_source_hash,
        master_hash=base_manifest.master_hash,
        quality_report_hash=base_manifest.quality_report_hash,
        schema_version="v2",
        content_hash=canonical_content_hash(projected, ordered_columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    _validate_certification_evidence(manifest)

    content_manifest: dict[str, object] = {
        "feature_contract_version": book.version,
        "feature_set_hash": feature_set_hash(book.contracts),
        "base_panel_id": request.base_panel_id,
        "base_panel_content_hash": base_manifest.content_hash,
        "generated_time": generated_time.isoformat(),
    }
    out_store = ParquetDatasetStore(Path(destination_root))
    dataset_dir = out_store.write_partitioned(
        projected,
        dataset_id=request.dataset_id,
        manifest=manifest,
        expected_feature_set=request.feature_set,
        decision_time=generated_time,
        content_manifest=content_manifest,
    )
    contract_payload = {
        "contract_version": book.version,
        "feature_set_hash": feature_set_hash(book.contracts),
        "base_panel_id": request.base_panel_id,
        "contracts": contracts_to_json(book.contracts),
    }
    contract_path = dataset_dir / "feature_contract.json"
    contract_path.write_text(
        json.dumps(contract_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    partition_paths = tuple(sorted((dataset_dir / "partitions").rglob("*.parquet")))
    logger.info(
        "feature panel %s: %s rows, %s features from base %s",
        request.dataset_id,
        projected.height,
        len(book.contracts),
        request.base_panel_id,
    )
    return FeaturePanelResult(
        dataset_id=request.dataset_id,
        manifest=manifest,
        contract_path=contract_path,
        partition_paths=partition_paths,
        row_count=projected.height,
        base_panel_id=request.base_panel_id,
    )


V2_READINESS_NAME = "v2_readiness.json"


def build_stock_alpha_v2_feature_panel(
    base_root: Path,
    destination_root: Path,
    request: FeaturePanelRequest,
    *,
    min_coverage: float = 0.75,
) -> FeaturePanelResult:
    """Project the frozen 34-name ``stock_alpha_v2`` feature panel.

    Reuses :func:`build_feature_panel` for the deterministic ``raw__`` ->
    ``feature__`` projection and persistence, but first validates the v2
    publication gates against the projected frame before any directory or
    manifest is created:

    - all 34 ``raw__`` source columns of ``stock_alpha_v2_allowlist()`` exist;
    - the projected feature set is exactly the ordered allowlist;
    - no ``target_``/``label_`` column leaks;
    - no selected feature is fully null;
    - every selected feature has non-null coverage >= ``min_coverage``;
    - no NaN/Infinity in a selected non-null feature.

    Coverage is a publication gate, not a training-time imputation decision;
    per-feature coverage, null counts, source base hash, allowlist hash, and the
    threshold are recorded in an adjacent ``v2_readiness.json`` referenced from
    ``feature_contract.json``.
    """
    from src.stocks.research.features import stock_alpha_v2_allowlist

    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be within (0, 1]")
    allowlist = stock_alpha_v2_allowlist()
    book = feature_contract_book_from_allowlist("stock_alpha_v2", allowlist)
    generated_time = request.generated_time or datetime.now(UTC)
    store = ParquetDatasetStore(Path(base_root))
    base_manifest = store.read_manifest(request.base_panel_id)
    base_frame = store.read(
        request.base_panel_id, AssetKind.STOCK, BASE_PANEL_FEATURE_SET, generated_time
    )

    missing_sources = [
        name for name in allowlist if f"raw__{name}" not in base_frame.columns
    ]
    if missing_sources:
        raise ValueError(
            f"base panel {request.base_panel_id} missing v2 raw source columns "
            f"{missing_sources}"
        )
    projected = book.project(base_frame, source_prefix="raw__")
    if projected.is_empty():
        raise ValueError("v2 feature projection produced no rows")
    coverage = _validate_v2_projection(projected, allowlist, min_coverage)

    v2_request = FeaturePanelRequest(
        dataset_id=request.dataset_id,
        base_panel_id=request.base_panel_id,
        feature_set="stock_alpha_v2",
        feature_contract_book=book,
        provider_version=request.provider_version,
        universe_policy_version=request.universe_policy_version,
        certification=request.certification,
        generated_time=request.generated_time,
    )
    result = build_feature_panel(base_root, destination_root, v2_request)
    _write_v2_readiness(result, book, base_manifest, projected, coverage, min_coverage)
    logger.info(
        "v2 feature panel %s: %s rows, min coverage %.6f",
        request.dataset_id,
        projected.height,
        min(coverage.values()),
    )
    return result


def _validate_v2_projection(
    projected: pl.DataFrame,
    allowlist: tuple[str, ...],
    min_coverage: float,
) -> dict[str, float]:
    """Fail closed unless the projected v2 frame satisfies publication gates."""
    expected_columns = ["instrument_id", "session"] + [
        f"feature__{name}" for name in allowlist
    ]
    if projected.columns != expected_columns:
        raise ValueError(
            "v2 projection is not exactly the ordered allowlist; "
            f"expected {len(expected_columns)} columns, got {projected.columns}"
        )
    if any(c.startswith(_TARGET_PREFIXES) for c in projected.columns):
        raise ValueError("v2 feature projection leaked a target/label column")

    coverage: dict[str, float] = {}
    height = projected.height
    for name in allowlist:
        column = f"feature__{name}"
        null_count = int(projected[column].null_count())
        non_null = height - null_count
        if non_null == 0:
            raise ValueError(f"v2 feature {name!r} is fully null")
        rate = non_null / height
        if rate < min_coverage:
            raise ValueError(
                f"v2 feature {name!r} coverage {rate:.6f} is below {min_coverage}"
            )
        non_finite = projected.filter(
            pl.col(column).is_not_null() & ~pl.col(column).is_finite()
        )
        if not non_finite.is_empty():
            raise ValueError(f"v2 feature {name!r} contains NaN/Infinity values")
        coverage[name] = rate
    return coverage


def _write_v2_readiness(
    result: FeaturePanelResult,
    book: FeatureContractBook,
    base_manifest: DatasetManifest,
    projected: pl.DataFrame,
    coverage: dict[str, float],
    min_coverage: float,
) -> Path:
    """Write per-feature coverage evidence next to ``feature_contract.json``."""
    payload: dict[str, object] = {
        "v2_readiness_version": 1,
        "feature_set": "stock_alpha_v2",
        "allowlist_hash": feature_set_hash(book.contracts),
        "base_panel_id": result.base_panel_id,
        "base_panel_content_hash": base_manifest.content_hash,
        "min_coverage": min_coverage,
        "row_count": projected.height,
        "features": [
            {
                "name": contract.name,
                "coverage": coverage[contract.name],
                "null_count": int(
                    projected[f"feature__{contract.name}"].null_count()
                ),
            }
            for contract in book.contracts
        ],
    }
    dataset_dir = result.contract_path.parent
    path = dataset_dir / V2_READINESS_NAME
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    _reference_v2_readiness(result.contract_path, V2_READINESS_NAME)
    return path


def _reference_v2_readiness(contract_path: Path, readiness_name: str) -> None:
    """Point ``feature_contract.json`` at the adjacent ``v2_readiness.json``."""
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["v2_readiness"] = readiness_name
    contract_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
