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
