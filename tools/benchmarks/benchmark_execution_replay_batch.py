"""Deterministic synthetic-panel benchmark for execution replay batch optimization.

PARALLEL_COMPLETION_04_BENCHMARK: Compares baseline (single-request) vs batch
(same-cadence shared-prepared) replay evidence, build counts, and timing.
Exits 0 only if:
  1. batch evidence is exact-equal to baseline evidence
  2. batch prepared-market builds are no greater than segment_count
  3. preparation and execution milliseconds are printed separately
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.costs import default_base_schedule, default_stress_schedule
from src.core.datasets import DatasetManifest
from src.core.instruments import AssetKind
from src.core.portfolio import PortfolioSnapshot
from src.stocks.domain.execution_policy import SCHEDULED_OPEN_V1
from src.stocks.ml.execution_replay import (
    ExecutionEquivalentReplayRequest,
    ExecutionReplayContext,
    ExecutionReplayEvidence,
    instruments_from_frame,
    prepare_execution_replay_batch,
    replay_execution_equivalent,
    replay_execution_equivalent_batch,
)
from src.stocks.research.artifacts import ModelArtifactRegistry
from src.stocks.trading.portfolio_constructor import StockRiskPolicy
from tests.fixtures.stocks.helpers import stock_liquidity_model


def _build_synthetic_panel(
    n_segments: int,
    sessions_per_segment: int,
    n_instruments: int,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, tuple[datetime, ...]]]:
    """Build deterministic synthetic market and score panels."""
    rows_market: list[dict[str, object]] = []
    rows_score: list[dict[str, object]] = []
    decision_sessions: dict[int, list[datetime]] = {}

    for segment in range(n_segments):
        for index in range(sessions_per_segment):
            session = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(
                days=segment * (sessions_per_segment + 5) + index
            )
            decision_sessions.setdefault(segment, []).append(session)
            for ticker in range(n_instruments):
                price = 100.0 + ticker + index * 0.1
                rows_market.append(
                    {
                        "instrument_id": f"KRX:{ticker + 1:05d}",
                        "session": session,
                        "observation_time": session.replace(hour=15, minute=30),
                        "available_time": session.replace(hour=15, minute=31),
                        "open": price,
                        "close": price * 1.01,
                        "volume": 1_000_000.0,
                        "trading_value": price * 1_000_000.0,
                        "sector": f"S{ticker % 4}",
                        "adtv": price * 1_000_000.0,
                    }
                )
                rows_score.append(
                    {
                        "instrument_id": f"KRX:{ticker + 1:05d}",
                        "session": session,
                        "oof_segment_id": segment,
                        "predicted_net_alpha": 0.01 + ticker * 0.001,
                        "expected_active_alpha": 0.01 + ticker * 0.001,
                        "alpha_lower_bound": 0.0,
                        "expected_net_alpha": 0.01 + ticker * 0.001,
                        "net_alpha_lower_bound": 0.0,
                        "exit_cost_rate": 0.001,
                    }
                )

    return (
        pl.DataFrame(rows_market),
        pl.DataFrame(rows_score),
        {s: tuple(sessions) for s, sessions in decision_sessions.items()},
    )


def _build_context(
    market: pl.DataFrame,
    *,
    seed: int = 42,
    top_k: int = 20,
) -> ExecutionReplayContext:
    instruments = instruments_from_frame(market)
    sessions = sorted(market["session"].unique().to_list())
    manifest = DatasetManifest(
        asset_kind=AssetKind.STOCK,
        schema_version="v1",
        schema_hash="bench",
        provider_version="bench",
        universe_policy_version="bench",
        universe_policy_hash="bench",
        feature_set="stock_net_alpha_v1",
        feature_set_hash="bench",
        label_definition="net_alpha_o2o",
        label_horizon_sessions=5,
        time_start=sessions[0],
        time_end=sessions[-1],
        generated_time=sessions[-1],
        row_count=market.height,
    )
    return ExecutionReplayContext(
        registry=ModelArtifactRegistry(Path("mem://bench")),
        manifest=manifest,
        instruments=instruments,
        artifact_id="bench_replay",
        strategy_id="bench_replay",
        initial_portfolio=PortfolioSnapshot(
            account_snapshot_id="oof",
            as_of=sessions[0],
            settled_cash=100_000_000.0,
            unsettled_cash=0.0,
            positions=(),
        ),
        risk_policy=StockRiskPolicy(
            top_k=top_k,
            gross_cap=0.9,
            single_name_cap=0.3,
            sector_cap=0.5,
            participation_limit=0.01,
            no_trade_band_bps=0.0,
        ),
        base_cost_schedule=default_base_schedule(),
        stress_cost_schedule=default_stress_schedule(),
        liquidity_model=stock_liquidity_model(),
        stress_liquidity_model=stock_liquidity_model(stress_multiplier=1.5),
        execution_policy=SCHEDULED_OPEN_V1,
        seed=seed,
    )


def _bench_baseline(
    requests: list[ExecutionEquivalentReplayRequest],
) -> tuple[list[ExecutionReplayEvidence], int]:
    """Run baseline single-request replay, return evidence and segment build count."""
    start = time.monotonic()
    evidence = [replay_execution_equivalent(req) for req in requests]
    elapsed_ms = int((time.monotonic() - start) * 1000)
    return evidence, elapsed_ms


def _bench_batch(
    requests: list[ExecutionEquivalentReplayRequest],
    *,
    max_workers: int = 1,
) -> tuple[list[ExecutionReplayEvidence], int, dict[str, int]]:
    """Run batch replay, return evidence, timing, and build metrics."""
    start_prepare = time.monotonic()
    batch = prepare_execution_replay_batch(requests[0])
    prepare_ms = int((time.monotonic() - start_prepare) * 1000)

    start_execute = time.monotonic()
    result_tuple = replay_execution_equivalent_batch(
        requests, prepared_batch=batch, max_workers=max_workers
    )
    execute_ms = int((time.monotonic() - start_execute) * 1000)

    return list(result_tuple), prepare_ms + execute_ms, {
        "segment_builds": len(batch.segment_data),
        "prepare_ms": prepare_ms,
        "execute_ms": execute_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark execution replay batch optimization"
    )
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--sessions", type=int, default=500)
    parser.add_argument("--instruments", type=int, default=400)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    sessions_per_segment = max(1, args.sessions // args.segments)
    market, scores, decision_sessions = _build_synthetic_panel(
        args.segments, sessions_per_segment, args.instruments
    )

    context = _build_context(market)
    requests = [
        ExecutionEquivalentReplayRequest(
            context=ExecutionReplayContext(
                registry=context.registry,
                manifest=context.manifest,
                instruments=context.instruments,
                artifact_id=context.artifact_id,
                strategy_id=context.strategy_id,
                initial_portfolio=context.initial_portfolio,
                risk_policy=context.risk_policy,
                base_cost_schedule=context.base_cost_schedule,
                stress_cost_schedule=context.stress_cost_schedule,
                liquidity_model=context.liquidity_model,
                stress_liquidity_model=context.stress_liquidity_model,
                execution_policy=context.execution_policy,
                seed=context.seed + i,
            ),
            market_frame=market,
            score_frame=scores,
            segment_column="oof_segment_id",
            decision_sessions_by_segment=decision_sessions,
            horizon_sessions=10,
        )
        for i in range(args.requests)
    ]

    baseline_evidence, baseline_ms = _bench_baseline(requests)
    batch_evidence, batch_ms, batch_metrics = _bench_batch(
        requests, max_workers=args.workers
    )

    segment_builds = batch_metrics["segment_builds"]
    prepare_ms = batch_metrics["prepare_ms"]
    execute_ms = batch_metrics["execute_ms"]

    parity_ok = True
    for b_ev, bt_ev in zip(baseline_evidence, batch_evidence, strict=True):
        if b_ev.base_log_growth != bt_ev.base_log_growth:
            parity_ok = False
            break
        if b_ev.stress_log_growth != bt_ev.stress_log_growth:
            parity_ok = False
            break
        if b_ev.segment_ids != bt_ev.segment_ids:
            parity_ok = False
            break
        if b_ev.planned_cycles != bt_ev.planned_cycles:
            parity_ok = False
            break
        if b_ev.filled_orders != bt_ev.filled_orders:
            parity_ok = False
            break

    build_bound_ok = segment_builds <= args.segments
    passed = parity_ok and build_bound_ok

    _out = sys.stdout
    _out.write(
        f"segments={args.segments} sessions={args.sessions} "
        f"instruments={args.instruments} requests={args.requests}\n"
    )
    _out.write(f"baseline_ms={baseline_ms} batch_ms={batch_ms}\n")
    _out.write(f"prepare_ms={prepare_ms} execute_ms={execute_ms}\n")
    _out.write(
        f"baseline_segment_builds={args.segments * args.requests} "
        f"batch_segment_builds={segment_builds}\n"
    )
    _out.write(f"parity={'PASS' if parity_ok else 'FAIL'}\n")
    _out.write(f"build_bound={'PASS' if build_bound_ok else 'FAIL'}\n")
    _out.write(f"RESULT={'PASS' if passed else 'FAIL'}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
