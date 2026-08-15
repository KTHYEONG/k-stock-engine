"""Deterministic stock fixture builders shared by unit tests."""
from __future__ import annotations

from datetime import date, datetime, timedelta, UTC
from math import inf
from pathlib import Path

import numpy as np
import polars as pl

from src.core.costs import LiquiditySlippageModel, TickSizeRule, TickSizeSchedule
from src.core.instruments import AssetKind
from src.core.datasets import DatasetManifest, make_manifest
from src.stocks.data.contracts import CoverageRange
from src.stocks.data.costs import (
    CommissionRule,
    CostEvidence,
    LiquidityModelSpec,
    SellTaxRule,
    SourceRecord,
)
from src.storage.parquet_datasets import (
    HIVE_PARTITION_LAYOUT,
    ParquetDatasetStore,
    canonical_content_hash,
)


def cost_evidence_fixture(
    effective_from: datetime | None = None,
) -> CostEvidence:
    """Minimal hash-bound cost evidence covering 2024 for engine/simulator tests.

    Uses the 2024 statutory sell tax (KOSPI 0.03% STT + 0.15% rural = 0.18%,
    KOSDAQ 0.18% STT) and the unified KRX regular-session tick table, mirroring
    the real counterfactual artifact's structure.
    """
    eff = effective_from or datetime(2024, 1, 1, tzinfo=UTC)
    source = SourceRecord(
        uri="https://law.go.kr/fixture",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        content_hash="fixture-source-hash",
    )
    return CostEvidence(
        schema_version=1,
        coverage=CoverageRange(start=date(2024, 1, 1), end=date(2024, 3, 31)),
        assumption_id="fixture_kis_v1",
        sources=(source,),
        commission=(
            CommissionRule(
                effective_from=eff, buy_rate=0.000036396, sell_rate=0.000036396
            ),
        ),
        sell_taxes=(
            SellTaxRule(
                effective_from=eff,
                market="KOSPI",
                securities_transaction_tax_rate=0.0003,
                rural_special_tax_rate=0.0015,
                source_uri=source.uri,
                source_hash=source.content_hash,
            ),
            SellTaxRule(
                effective_from=eff,
                market="KOSDAQ",
                securities_transaction_tax_rate=0.0018,
                rural_special_tax_rate=0.0,
                source_uri=source.uri,
                source_hash=source.content_hash,
            ),
        ),
        tick_size_rules=(
            TickSizeRule("krx_test_0", eff, 0.0, 1000.0, 1.0),
            TickSizeRule("krx_test_1", eff, 1000.0, 5000.0, 5.0),
            TickSizeRule("krx_test_2", eff, 5000.0, 10000.0, 10.0),
            TickSizeRule("krx_test_3", eff, 10000.0, 50000.0, 50.0),
            TickSizeRule("krx_test_4", eff, 50000.0, 100000.0, 100.0),
            TickSizeRule("krx_test_5", eff, 100000.0, 500000.0, 500.0),
            TickSizeRule("krx_test_6", eff, 500000.0, inf, 1000.0),
        ),
        liquidity_model=LiquidityModelSpec(
            model_id="sqrt_impact_v1", impact_coefficient=0.1, stress_multiplier=1.5
        ),
        content_hash="fixture-cost-hash",
    )


def stock_instrument_df(
    n_sessions: int = 40,
    n_tickers: int = 5,
    horizon: int = 5,
    start: datetime | None = None,
) -> pl.DataFrame:
    """Deterministic point-in-time daily bars for stock research tests."""
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    for t in range(n_tickers):
        for s in range(n_sessions):
            obs = start + timedelta(days=s)
            close = 100.0 + float((t * 7 + s) % 20)
            rows.append(
                {
                    "session_index": s,
                    "session": obs,
                    "instrument_id": f"KRX:0{t + 1:05d}",
                    "observation_time": obs.replace(hour=15, minute=30, tzinfo=UTC),
                    "available_time": obs.replace(hour=15, minute=31, tzinfo=UTC),
                    "open": close - 1.0,
                    "high": close + 1.0,
                    "low": close - 1.5,
                    "close": close,
                    "volume": 1_000_000.0 + float(t) * 100_000.0,
                    "trading_value": close * (1_000_000.0 + float(t) * 100_000.0),
                    "market_cap": close * 10_000_000.0,
                    "feature_momentum_5d": float((t + s) % 7) / 7.0,
                    "is_universe": True,
                    "sector": f"S{t % 2}",
                    "adtv": close * (1_000_000.0 + float(t) * 100_000.0),
                }
            )
    return pl.DataFrame(rows)


def stock_manifest(
    columns: list[str] | None = None,
    asset_kind: AssetKind = AssetKind.STOCK,
    feature_set: str = "stock_alpha_v1",
    horizon: int = 5,
    decision_time: datetime | None = None,
) -> DatasetManifest:
    cols = columns or [
        "session_index",
        "session",
        "instrument_id",
        "feature_momentum_5d",
    ]
    return make_manifest(
        asset_kind=asset_kind,
        columns=cols,
        feature_set=feature_set,
        label_definition="fwd_ret_5d",
        label_horizon_sessions=horizon,
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=datetime(2024, 3, 1, tzinfo=UTC),
        provider_version="fixture",
        universe_policy_version="fixture",
        row_count=len(cols) * 10,
        generated_time=decision_time,
    )


def stock_v2_composed_df(
    n_sessions: int = 40,
    n_tickers: int = 40,
    horizon: int = 5,
    start: datetime | None = None,
    seed: int = 7,
) -> pl.DataFrame:
    """Deterministic composed v2 training panel: base + feature__* + residual labels.

    Every row carries the 34 ``feature__`` stock_alpha_v2 columns, the
    ``residual_o2o_5d`` label, its LambdaRank ``relevance``, and a
    ``label_available_time`` so the trainer and scorer share one contract.
    """
    from src.stocks.research.features import stock_alpha_v2_allowlist
    from src.stocks.research.labels import residual_open_to_open_label

    allowlist = stock_alpha_v2_allowlist()
    rng = np.random.default_rng(seed)
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    for t in range(n_tickers):
        price = 100.0
        for s in range(n_sessions):
            obs = start + timedelta(days=s)
            price = max(10.0, price * (1.0 + float(rng.normal(0.0, 0.02))))
            rows.append(
                {
                    "session_index": s,
                    "session": obs,
                    "instrument_id": f"KRX:0{t + 1:05d}",
                    "observation_time": obs.replace(hour=15, minute=30, tzinfo=UTC),
                    "available_time": obs.replace(hour=15, minute=31, tzinfo=UTC),
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price * 1.001,
                    "volume": 1_000_000.0,
                    "trading_value": price * 1_000_000.0,
                    "market_cap": price * 10_000_000.0,
                    "sector": f"S{t % 4}",
                    "adtv": price * 1_000_000.0,
                    **{
                        f"feature__{name}": float(rng.normal(t * 0.01, 1.0))
                        for name in allowlist
                    },
                }
            )
    df = pl.DataFrame(rows)
    labels = residual_open_to_open_label(
        df.select(["instrument_id", "session", "open"]), horizon_sessions=horizon
    )
    return df.join(labels, on=["instrument_id", "session"], how="inner")


def stock_v2_manifest(
    columns: list[str] | None = None,
    horizon: int = 5,
    decision_time: datetime | None = None,
) -> DatasetManifest:
    return stock_manifest(
        columns=columns,
        feature_set="stock_alpha_v2",
        horizon=horizon,
        decision_time=decision_time,
    )


def publish_baseline_artifact(
    registry,
    *,
    artifact_id: str,
    feature_set: str = "stock_alpha_v1",
    feature_schema_hash: str = "fixture-schema",
    eligible_from: str = "2024-01-01T00:00:00+00:00",
    eligible_to: str = "2024-03-31T00:00:00+00:00",
    ranking_feature: str = "feature_momentum_5d",
    promoted: bool = True,
) -> str:
    """Publish an immutable deterministic baseline artifact for cycle tests.

    The artifact is promoted (not a ``NO_TRADE`` composite), so a planning
    cycle can actually allocate against it.
    """
    from src.stocks.research.models import DeterministicBaseline, ModelManifest

    manifest = ModelManifest(
        artifact_id=artifact_id,
        asset_kind=AssetKind.STOCK,
        feature_set=feature_set,
        feature_schema_hash=feature_schema_hash,
        universe_policy_hash="fixture-universe",
        label_definition="fwd_ret_5d",
        label_horizon_sessions=5,
        eligible_from=eligible_from,
        eligible_to=eligible_to,
        model_type="deterministic_baseline",
    )
    registry.publish(
        DeterministicBaseline(manifest, ranking_feature=ranking_feature), manifest
    )
    registry.write_metrics(artifact_id, {"promoted": promoted})
    return artifact_id


def feature_readiness_dataset(root: Path) -> Path:
    """Build a partitioned feature dataset for readiness-gate tests.

    Columns: ``feature__overnight_ret`` (all finite), ``feature__ret_21_60d``
    (warm-up nulls in January only), ``feature__inactive`` (fully null), and
    ``feature__bad`` (January nulls, February +infinity).
    """
    rows = []
    for i in range(4):
        month = 1 if i < 2 else 2
        rows.append(
            {
                "instrument_id": f"KRX:0000{i + 1}",
                "session": datetime(2024, month, 2 + i % 2, tzinfo=UTC),
                "feature__overnight_ret": 0.001 * (i + 1),
                "feature__ret_21_60d": None if month == 1 else 0.01 * (i + 1),
                "feature__inactive": None,
                "feature__bad": None if i < 2 else inf,
            }
        )
    frame = pl.DataFrame(rows)
    store = ParquetDatasetStore(root)
    manifest = make_manifest(
        asset_kind=AssetKind.STOCK,
        columns=frame.columns,
        feature_set="stock_alpha_v1",
        label_definition="fwd_ret_2d",
        label_horizon_sessions=2,
        time_start=datetime(2024, 1, 2, tzinfo=UTC),
        time_end=datetime(2024, 2, 3, tzinfo=UTC),
        provider_version="fixture",
        universe_policy_version="fixture",
        row_count=frame.height,
        schema_version="v2",
        content_hash=canonical_content_hash(frame, frame.columns),
        storage_layout=HIVE_PARTITION_LAYOUT,
    )
    return store.write_partitioned(
        frame,
        dataset_id="features_readiness_v1",
        manifest=manifest,
        expected_feature_set="stock_alpha_v1",
        decision_time=datetime(2026, 1, 1, tzinfo=UTC),
        content_manifest={"curation_version": "curation-v1"},
    )


def stock_net_alpha_composed_df(
    n_sessions: int = 100,
    n_tickers: int = 8,
    candidate_horizon_sessions: tuple[int, ...] = (3, 5, 8, 10, 15, 20),
    seed: int = 11,
    audit_clean: bool = False,
    label_scale: float = 1.0,
) -> pl.DataFrame:
    """Deterministic composed net-alpha panel: raw sources + per-horizon labels.

    Every row carries the canonical ``stock_net_alpha_v1`` raw source columns
    (by raw name, so ``build_model_features`` can canonicalize them), the
    ``net_alpha_<h>d_target`` continuous label for every candidate horizon, and
    the exact exit-open ``label_available_time_<h>d`` availability columns the
    trainer gates on. With ``audit_clean`` the panel clears the net-alpha
    integrity audit: the ADTV source varies across sessions and the first
    observation of each instrument carries the declared warm-up nulls.
    ``label_scale`` scales the synthetic label/residual so the ElasticNet
    baseline can recover the linear signal at small training-sample sizes.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict] = []
    for t in range(n_tickers):
        for s in range(n_sessions):
            obs = start + timedelta(days=s)
            momentum = float(rng.normal(0.0, 1.0))
            rows.append(
                {
                    "session_index": s,
                    "session": obs,
                    "instrument_id": f"KRX:0{t + 1:05d}",
                    "observation_time": obs.replace(hour=15, minute=30, tzinfo=UTC),
                    "available_time": obs.replace(hour=15, minute=31, tzinfo=UTC),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1_000_000.0,
                    "trading_value": 100.5 * 1_000_000.0,
                    "market_cap": 1.0e11,
                    "sector": f"S{t % 4}",
                    "adtv": 1.0e8,
                    "beta": 1.0,
                    "volatility": 0.02,
                    "ep_ratio": float(rng.normal(0.0, 1.0)),
                    "bp_ratio": float(rng.normal(0.0, 1.0)),
                    "overnight_ret": momentum,
                    "intraday_ret": -momentum,
                    "ret_2_5d": momentum * 0.5,
                    "ret_6_20d": momentum * 0.3,
                    "ret_21_60d": momentum * 0.2,
                    "close_high_ratio_10d": momentum * 0.4,
                    "relative_trend_score": momentum * 0.6,
                    "sector_ret_5d": momentum * 0.1,
                    "disparity_120d": momentum * 0.7,
                    "foreign_net_buy": float(rng.normal(0.0, 1.0)),
                    "institution_net_buy": float(rng.normal(0.0, 1.0)),
                    "individual_net_buy": float(rng.normal(0.0, 1.0)),
                    "flow_consensus": float(rng.normal(0.0, 1.0)),
                    "flow_intensity_20d": float(rng.normal(0.0, 1.0)),
                    "volume_shock": float(rng.normal(0.0, 1.0)),
                    "vpt_20d": float(rng.normal(0.0, 1.0)),
                    "info_ratio_20d": float(rng.normal(0.0, 1.0)),
                    "vol_asymmetry_20d": float(rng.normal(0.0, 1.0)),
                    "mcap_rank": float(t) / n_tickers,
                    "min_vol_5d": float(rng.normal(0.02, 0.005)),
                    "volatility_20d": float(rng.normal(0.02, 0.005)),
                    "volatility_60d": float(rng.normal(0.02, 0.005)),
                    "vol_regime": float(rng.normal(0.0, 1.0)),
                    "adtv_20d": 1.0e8,
                    "turnover_ratio": float(rng.normal(0.01, 0.001)),
                    "amihud_20d": float(rng.normal(1.0e-9, 1.0e-10)),
                    "fluc_rate": float(rng.normal(0.02, 0.005)),
                }
            )
    frame = pl.DataFrame(rows)
    if audit_clean:
        frame = frame.with_columns(
            (
                pl.col("adtv_20d")
                * (1.0 + 0.1 * pl.col("session_index").cast(pl.Float64) / n_sessions)
            ).alias("adtv_20d")
        )
        frame = frame.with_columns(
            pl.int_range(0, pl.len()).over("instrument_id").alias("__obs")
        )
        for column in ("fluc_rate", "intraday_ret", "overnight_ret", "sector_ret_5d"):
            frame = frame.with_columns(
                pl.when(pl.col("__obs") == 0)
                .then(None)
                .otherwise(pl.col(column))
                .alias(column)
            )
        frame = frame.drop("__obs")
    for horizon in candidate_horizon_sessions:
        signal = label_scale * (
            pl.col("overnight_ret") * 0.02 - pl.col("intraday_ret") * 0.01
        )
        target = signal.alias(f"net_alpha_{horizon}d_target")
        residual = signal.alias(f"risk_residual_{horizon}d")
        reference_cost = pl.lit(0.001, dtype=pl.Float64).alias(
            f"reference_cost_{horizon}d"
        )
        available = (pl.col("session") + pl.duration(days=horizon)).alias(
            f"label_available_time_{horizon}d"
        )
        frame = frame.with_columns(target, residual, reference_cost, available)
    return frame


def stock_liquidity_model(
    impact_coefficient: float = 0.05,
    stress_multiplier: float = 1.0,
) -> LiquiditySlippageModel:
    """Deterministic point-in-time liquidity slippage model for net-alpha fixtures."""
    tick = TickSizeSchedule(
        rules=(
            TickSizeRule(
                rule_id="fixture-tick",
                effective_from=datetime(2000, 1, 1, tzinfo=UTC),
                lower_inclusive=0.0,
                upper_exclusive=inf,
                tick=0.05,
            ),
        )
    )
    return LiquiditySlippageModel(
        impact_coefficient=impact_coefficient,
        tick_schedule=tick,
        stress_multiplier=stress_multiplier,
        model_id="fixture-sqrt-impact-v1",
    )


def stock_net_alpha_manifest(
    columns: list[str] | None = None,
    horizon: int = 5,
    decision_time: datetime | None = None,
) -> DatasetManifest:
    return stock_manifest(
        columns=columns,
        feature_set="stock_net_alpha_v1",
        horizon=horizon,
        decision_time=decision_time,
    )
