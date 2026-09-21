"""Single-instrument smoke tests on historical 2016 market data.

Validates end-to-end execution, fill model, cost schedule, slippage,
T+2 settlement, accounting identity, and result artifact generation.
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from src.core.costs import (
    LiquiditySlippageModel,
    TickSizeRule,
    TickSizeSchedule,
    default_base_schedule,
)
from src.core.instruments import AssetKind, Instrument
from src.core.time import SessionCalendar
from src.data.backtest_runner import run_managed_backtest, verify_accounting_identity
from src.data.backtest_sessions import build_backtest_sessions
from src.data.schemas import SilverTable
from src.data.snapshot import PITSnapshotRepository
from src.engine.backtest import BacktestConfig, BacktestSession
from src.engine.decision import DecisionContext
from src.engine.fill_model import ExecutionScenario, HistoricalFillModel
from src.execution.domain.intents import TradeIntent

KST = ZoneInfo("Asia/Seoul")
UTC = UTC


@pytest.fixture
def samsung_2016_january_sessions(tmp_path: Path) -> tuple[tuple[BacktestSession, ...], SessionCalendar, Instrument]:
    dm_root = Path("data/silver/stocks/daily_market")
    if not dm_root.exists():
        pytest.skip("Silver daily_market data not present")
    files = list(dm_root.rglob("*.parquet"))
    if not files:
        pytest.skip("No parquet files in daily_market")

    df_market = (
        pl.scan_parquet(
            files,
            cast_options=pl.ScanCastOptions(datetime_cast="convert-timezone"),
        )
        .filter(
            (pl.col("instrument_id") == "KRX:005930")
            & (pl.col("session") >= datetime(2016, 1, 1, tzinfo=KST))
            & (pl.col("session") <= datetime(2016, 2, 5, tzinfo=KST))
        )
        .unique(subset=["session"], keep="first", maintain_order=True)
        .sort("session")
        .collect()
    )
    if df_market.height < 62 or "market_cap" not in df_market.columns:
        pytest.skip("Insufficient Samsung rolling market inputs")

    df_market_pit = df_market.with_columns(
        pl.col("session").dt.replace(hour=15, minute=30, second=0).alias("available_at")
    )
    sessions_list = tuple(sorted(df_market_pit["session"].to_list()))
    calendar = SessionCalendar(sessions_list)
    repo = PITSnapshotRepository.from_frames({SilverTable.DAILY_MARKET: df_market_pit}, root=tmp_path)

    master = pl.DataFrame({
        "instrument_id": ["KRX:005930"],
        "sector": ["Technology"],
        "valid_from": [sessions_list[0]],
        "valid_to": [sessions_list[-1]],
        "available_at": [sessions_list[0]],
    })
    actions = pl.DataFrame(schema={
        "effective_session": pl.Datetime(time_zone="Asia/Seoul"),
        "instrument_id": pl.String,
        "action_type": pl.String,
        "available_at": pl.Datetime(time_zone="Asia/Seoul"),
    })

    start = sessions_list[0]
    end = sessions_list[-2]
    sessions = build_backtest_sessions(
        snapshot_repository=repo,
        calendar=calendar,
        start=start,
        end=end,
        decision_time_of=lambda s: s.replace(hour=15, minute=30, second=0),
        security_master=master,
        corporate_actions=actions,
    )
    instrument = Instrument("KRX:005930", AssetKind.STOCK, "KRX", "005930", "KRW")
    return sessions, calendar, instrument


def test_samsung_2016_smoke_roundtrip_settlement_and_accounting(
    samsung_2016_january_sessions: tuple[tuple[BacktestSession, ...], SessionCalendar, Instrument],
    tmp_path: Path,
) -> None:
    sessions, calendar, instrument = samsung_2016_january_sessions
    costs = default_base_schedule()
    ticks = TickSizeSchedule((
        TickSizeRule("all", datetime(2000, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1000.0),
    ))
    fill_model = HistoricalFillModel(
        costs,
        LiquiditySlippageModel(0.1, ticks),
        ExecutionScenario.BASE,
        target_participation_cap=0.1,
        hard_participation_cap=0.2,
    )

    sessions_ordered = [s.session_open for s in sessions]

    class RoundtripStrategy:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            idx = sessions_ordered.index(context.decision_time.replace(hour=9, minute=0, second=0))
            if idx == 0:
                return (
                    TradeIntent(
                        intent_id="buy-samsung-smoke",
                        asset_kind=AssetKind.STOCK,
                        instrument_id=instrument.instrument_id,
                        target_value=50_000_000.0,
                        decision_time=context.decision_time,
                        execution_time=sessions_ordered[1],
                        strategy_id="champion-v1",
                        reason="smoke_entry",
                        idempotency_key="smoke_buy_key",
                        account_snapshot_id=context.portfolio.account_snapshot_id,
                    ),
                )
            if idx == 8:
                return (
                    TradeIntent(
                        intent_id="sell-samsung-smoke",
                        asset_kind=AssetKind.STOCK,
                        instrument_id=instrument.instrument_id,
                        target_value=0.0,
                        decision_time=context.decision_time,
                        execution_time=sessions_ordered[9],
                        strategy_id="champion-v1",
                        reason="smoke_exit",
                        idempotency_key="smoke_sell_key",
                        account_snapshot_id=context.portfolio.account_snapshot_id,
                    ),
                )
            return ()

    config = BacktestConfig(
        ledger_id="smoke-samsung-roundtrip",
        initial_cash=100_000_000.0,
        instruments={instrument.instrument_id: instrument},
        scenario=ExecutionScenario.BASE,
        cost_schedule=costs,
        calendar=calendar,
        fill_model=fill_model,
    )

    result, manifest = run_managed_backtest(
        sessions=sessions,
        config=config,
        strategy=RoundtripStrategy(),
        artifact_root=tmp_path / "artifacts",
        dataset_hash="test_samsung_jan_2016",
        smoke_symbol="KRX:005930",
    )

    assert len(result.fills) == 2
    buy_fill = result.fills[0]
    sell_fill = result.fills[1]
    assert buy_fill.side.value == "BUY"
    assert sell_fill.side.value == "SELL"
    assert buy_fill.quantity == sell_fill.quantity
    assert buy_fill.quantity > 0

    assert buy_fill.commission > 0
    assert buy_fill.tax == 0.0
    assert buy_fill.slippage_cost > 0
    assert sell_fill.commission > 0
    assert sell_fill.tax > 0.0
    assert sell_fill.slippage_cost > 0

    assert verify_accounting_identity(result.daily_nav) is True
    final_nav = result.daily_nav[-1]
    assert final_nav.unsettled_cash == pytest.approx(0.0, abs=1e-5)
    assert final_nav.marked_value == pytest.approx(0.0, abs=1e-5)
    assert final_nav.nav == pytest.approx(final_nav.settled_cash, abs=1e-5)

    assert manifest["content_hash"] != ""
    assert manifest["fill_count"] == 2
    assert manifest["reject_count"] == 0
    assert manifest["accounting_reconciled"] is True
    assert "performance" in manifest
    perf = manifest["performance"]
    assert perf["initial_nav"] == 100_000_000.0
    assert perf["mdd"] >= 0.0


def test_samsung_smoke_reproducibility_identical_hash(
    samsung_2016_january_sessions: tuple[tuple[BacktestSession, ...], SessionCalendar, Instrument],
    tmp_path: Path,
) -> None:
    sessions, calendar, instrument = samsung_2016_january_sessions
    costs = default_base_schedule()
    ticks = TickSizeSchedule((
        TickSizeRule("all", datetime(2000, 1, 1, tzinfo=UTC), 0.0, float("inf"), 1000.0),
    ))
    fill_model = HistoricalFillModel(
        costs,
        LiquiditySlippageModel(0.1, ticks),
        ExecutionScenario.BASE,
        target_participation_cap=0.1,
        hard_participation_cap=0.2,
    )

    class StaticStrategy:
        def decide(self, context: DecisionContext) -> tuple[TradeIntent, ...]:
            return ()

    config = BacktestConfig(
        ledger_id="smoke-reproducibility",
        initial_cash=100_000_000.0,
        instruments={instrument.instrument_id: instrument},
        scenario=ExecutionScenario.BASE,
        cost_schedule=costs,
        calendar=calendar,
        fill_model=fill_model,
    )

    _, manifest_run1 = run_managed_backtest(
        sessions=sessions,
        config=config,
        strategy=StaticStrategy(),
        artifact_root=tmp_path / "artifacts1",
        dataset_hash="reproducibility_test_v1",
        smoke_symbol="KRX:005930",
    )

    _, manifest_run2 = run_managed_backtest(
        sessions=sessions,
        config=config,
        strategy=StaticStrategy(),
        artifact_root=tmp_path / "artifacts2",
        dataset_hash="reproducibility_test_v1",
        smoke_symbol="KRX:005930",
    )

    assert manifest_run1["content_hash"] == manifest_run2["content_hash"]
    assert manifest_run1["performance"] == manifest_run2["performance"]


def test_compute_backtest_performance_empty() -> None:
    from src.data.backtest_runner import compute_backtest_performance
    perf = compute_backtest_performance(())
    assert perf["initial_nav"] == 0.0
    assert perf["cagr"] == 0.0


def test_verify_accounting_identity_valid() -> None:
    from datetime import UTC, datetime
    from src.core.ledger import LedgerNav
    from src.data.backtest_runner import verify_accounting_identity

    nav = LedgerNav(
        mark_id="mark-1",
        as_of=datetime(2020, 1, 1, 9, 0, tzinfo=UTC),
        nav=100.0,
        settled_cash=50.0,
        unsettled_cash=20.0,
        marked_value=30.0,
    )
    assert verify_accounting_identity((nav,)) is True


def test_core_artifact_provenance() -> None:
    from src.data.backtest_runner import run_managed_backtest

    assert callable(run_managed_backtest)
    required = {'strategy_id', 'score_policy_version', 'selection_policy_version', 'portfolio_policy_version', 'market_input_policy_version', 'warmup_sessions', 'data_action_certified'}
    metadata = {'strategy_id': 'core-v1', 'score_policy_version': 'korean-core-v1-scoring-v1', 'selection_policy_version': 'korean-core-v1-selection-v1', 'portfolio_policy_version': 'champion-v1-portfolio-v1', 'market_input_policy_version': 'korean-equity-market-inputs-v1', 'warmup_sessions': 60, 'data_action_certified': True}
    assert required <= metadata.keys()


def test_run_managed_backtest_emits_research_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from datetime import date
    from types import SimpleNamespace

    from src.core.ledger import LedgerNav

    def _nav(day: date, nav: float) -> LedgerNav:
        return LedgerNav(
            mark_id=f"m-{day.isoformat()}",
            as_of=datetime(day.year, day.month, day.day, 15, 30, tzinfo=KST),
            nav=nav, settled_cash=nav, unsettled_cash=0.0, marked_value=0.0,
        )

    # Given: an engine boundary stub spanning the development/holdout split.
    navs = (_nav(date(2023, 12, 27), 100.0), _nav(date(2023, 12, 28), 105.0), _nav(date(2024, 1, 2), 115.5))
    monkeypatch.setattr(
        "src.engine.runner.run_backtest",
        lambda config, sessions, strategy: SimpleNamespace(daily_nav=navs, fills=(), rejects=()),
    )
    config = SimpleNamespace(ledger_id="segments", scenario=SimpleNamespace(value="base"), initial_cash=100.0)

    # When
    _, payload = run_managed_backtest(
        sessions=(object(),), config=config, strategy=object(),  # type: ignore[arg-type]
        artifact_root=tmp_path / "artifacts", dataset_hash="segments",
    )

    # Then
    segments = payload["research_segments"]
    assert set(segments) == {"development", "holdout"}
    assert segments["development"]["total_return"] == pytest.approx(0.05)
    assert segments["holdout"]["total_return"] == pytest.approx(0.1)
    written = json.loads((tmp_path / "artifacts" / "backtests" / payload["content_hash"] / "result.json").read_text())
    assert written["research_segments"]["holdout"]["sessions"] == 1


SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")
SESSIONS_2023 = [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5), date(2023, 1, 6)]
A = "KRX:005930"
B = "KRX:000660"
C = "KRX:035420"
D = "KRX:051910"
E = "KRX:000270"
F = "KRX:068270"

CLOSES_A = [100.0, 110.0, 120.0, 130.0, 140.0]


def _scope_runtime(tmp_path: Path):
    from src.data.runtime import load_data_runtime

    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def _stamp_silver_bars(runtime, dataset_id: str, rows: list[dict] | None) -> None:
    import json

    target = runtime.workspace.silver_root / "daily_market" / dataset_id
    target.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        pl.DataFrame(rows).write_parquet(target / "bars.parquet")
    (target / "dataset_manifest.json").write_text(
        json.dumps({"scope_hash": runtime.scope.content_hash, "dataset_id": dataset_id}), encoding="utf-8"
    )


def _coverage(runtime):
    from src.data.scope_coverage import CoverageRequirement, ScopeCoverageReport

    fulfilled = (
        CoverageRequirement(source="krx_daily_market", natural_key="2023-01-02", as_of=date(2023, 1, 2), fiscal_period=None, required=True),
        CoverageRequirement(source="financial_facts", natural_key="00126380:2023:11013", as_of=date(2023, 5, 15), fiscal_period="2023Q1", required=True),
    )
    return ScopeCoverageReport(scope_hash=runtime.scope.content_hash, fulfilled=fulfilled, missing=(), unresolved=())


def _stamp_release(runtime, dataset_id: str, universe_rows: list[dict] | None = None):
    from src.data.gold import create_scope_bound_gold_release

    create_scope_bound_gold_release(
        runtime=runtime,
        dataset_id=dataset_id,
        silver_dataset_ids={"daily_market": "bars-v1"},
        universe_policy_hash="u",
        feature_policy_hash="f",
        coverage_report=_coverage(runtime),
    )
    if universe_rows is not None:
        release_dir = runtime.workspace.gold_root / "releases" / dataset_id
        pl.DataFrame(universe_rows).write_parquet(release_dir / "universe.parquet")


def _bar(session: date, instrument: str, close: float) -> dict:
    return {"session": session, "instrument_id": instrument, "close": close}


def _uni(session: date, instrument: str, *, eligible: bool = True, verified: bool = True, common: bool = True) -> dict:
    return {"session": session, "instrument_id": instrument, "eligible": eligible, "verified": verified, "common": common}


def _fixture_rows() -> tuple[list[dict], list[dict]]:
    bars: list[dict] = []
    universe: list[dict] = []
    for index, session in enumerate(SESSIONS_2023):
        bars.append(_bar(session, A, CLOSES_A[index]))
        bars.append(_bar(session, B, 50.0))
        bars.append(_bar(session, C, 50.0))
        bars.append(_bar(session, D, 50.0))
        if index < 4:
            bars.append(_bar(session, F, 50.0))
        universe.append(_uni(session, A))
        universe.append(_uni(session, B, common=False))
        universe.append(_uni(session, C, verified=False))
        universe.append(_uni(session, D, eligible=False))
        universe.append(_uni(session, E))
        universe.append(_uni(session, F))
    bars.append({"session": SESSIONS_2023[0], "instrument_id": "", "close": 1.0})
    bars.append({"session": SESSIONS_2023[0], "instrument_id": "JUNK", "close": "N/A"})
    universe.append({"session": SESSIONS_2023[0], "instrument_id": "", "eligible": True, "verified": True, "common": True})
    return bars, universe


def _build_manifest(runtime, *, dataset_id: str = "gold-v1", silver_ids: dict | None = None):
    from src.data.backtest_run_manifest import build_backtest_run_manifest

    return build_backtest_run_manifest(
        runtime=runtime, segment="validation", silver_dataset_ids=silver_ids or {"daily_market": "bars-v1"},
        gold_dataset_id=dataset_id, strategy_id="test-all",
        strategy_policy_hash="s", execution_policy_hash="e", universe_policy_hash="u",
    )


def _scope_fixture(tmp_path: Path, *, dataset_id: str = "gold-v1"):
    from src.data.backtest_runner import NextSessionExecutionModel

    runtime = _scope_runtime(tmp_path)
    bars, universe = _fixture_rows()
    _stamp_silver_bars(runtime, "bars-v1", bars)
    _stamp_release(runtime, dataset_id, universe)
    manifest = _build_manifest(runtime, dataset_id=dataset_id)
    execution = NextSessionExecutionModel(commission_rate=0.001, tax_rate=0.0025)
    return runtime, manifest, execution


class _SignalAll:
    strategy_id = "test-all"

    def __init__(self) -> None:
        self.fitted: tuple = ()

    def propose_fit_rows(self):  # type: ignore[no-untyped-def]
        return ()

    def apply_fit(self, rows) -> None:  # type: ignore[no-untyped-def]
        self.fitted = tuple(rows)

    def decide(self, *, session, candidates):  # type: ignore[no-untyped-def]
        from src.data.backtest_runner import SwingSignal

        ids = (*sorted({item.instrument_id for item in candidates}), "GHOST")
        return (SwingSignal("NEG", -1.0), *(SwingSignal(iid, 1.0) for iid in ids))


def test_run_scope_bound_backtest_selects_manifest_inputs(tmp_path: Path) -> None:
    from src.data.backtest_runner import run_scope_bound_backtest

    runtime, manifest, execution = _scope_fixture(tmp_path)
    declining = [{**row, "close": 200.0 - CLOSES_A[SESSIONS_2023.index(row["session"])] + 100.0} if row["instrument_id"] == A else row for row in pl.read_parquet(runtime.workspace.silver_root / "daily_market" / "bars-v1" / "bars.parquet").to_dicts()]
    _stamp_silver_bars(runtime, "bars-v2", declining)
    _stamp_release(runtime, "gold-v2", _fixture_rows()[1])
    manifest_new = _build_manifest(runtime, dataset_id="gold-v2", silver_ids={"daily_market": "bars-v2"})

    old = run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)
    new = run_scope_bound_backtest(runtime=runtime, manifest=manifest_new, strategy=_SignalAll(), execution_model=execution)

    assert old.metrics["final_nav"] > old.metrics["initial_nav"]
    assert new.metrics["final_nav"] < new.metrics["initial_nav"]
    assert old.result_path.parent != new.result_path.parent


def test_run_scope_bound_backtest_excludes_unverified_tradability(tmp_path: Path) -> None:
    import json

    from src.data.backtest_runner import run_scope_bound_backtest

    runtime, manifest, execution = _scope_fixture(tmp_path)
    result = run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)
    payload = json.loads(result.result_path.read_text(encoding="utf-8"))

    assert payload["exclusions"] == {
        "unverified": 4, "ineligible": 4, "preferred": 4, "no_bar": 9, "no_execution": 7, "invalid_weight": 5,
    }
    assert payload["metrics"]["trade_count"] == 7.0
    assert payload["eligible_session_count"] == 5
    assert all(fill["instrument_id"] not in {B, C, D, E, "GHOST", "NEG"} for fill in payload["fills"])


def test_run_scope_bound_backtest_rejects_later_segment_fit(tmp_path: Path) -> None:
    from src.data.backtest_runner import StaticTransformRow, run_scope_bound_backtest
    from src.data.schemas import PITDataError

    runtime, manifest, execution = _scope_fixture(tmp_path)

    class _HoldoutFit(_SignalAll):
        def propose_fit_rows(self):  # type: ignore[no-untyped-def]
            return (StaticTransformRow(date(2024, 6, 1), 1.0),)

    with pytest.raises(PITDataError, match="outside the development segment"):
        run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_HoldoutFit(), execution_model=execution)

    class _DevFit(_SignalAll):
        def propose_fit_rows(self):  # type: ignore[no-untyped-def]
            return (StaticTransformRow(date(2021, 6, 1), 1.0),)

    strategy = _DevFit()
    run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=strategy, execution_model=execution)
    assert strategy.fitted == (StaticTransformRow(date(2021, 6, 1), 1.0),)


def test_run_scope_bound_backtest_delays_execution_to_next_session(tmp_path: Path) -> None:
    from src.data.backtest_runner import run_scope_bound_backtest

    runtime, manifest, execution = _scope_fixture(tmp_path)

    class _FirstDayOnly(_SignalAll):
        def decide(self, *, session, candidates):  # type: ignore[no-untyped-def]
            from src.data.backtest_runner import SwingSignal

            if session != SESSIONS_2023[0]:
                return ()
            return (SwingSignal(A, 1.0),)

    import json

    result = run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_FirstDayOnly(), execution_model=execution)
    payload = json.loads(result.result_path.read_text(encoding="utf-8"))

    assert len(payload["fills"]) == 1
    assert payload["fills"][0]["session"] == "2023-01-03"
    assert payload["fills"][0]["price"] == 110.0
    assert payload["metrics"]["trade_count"] == 1.0


def test_run_scope_bound_backtest_result_is_self_describing(tmp_path: Path) -> None:
    import json

    from src.data.backtest_runner import run_scope_bound_backtest

    runtime, manifest, execution = _scope_fixture(tmp_path)
    result = run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)
    payload = json.loads(result.result_path.read_text(encoding="utf-8"))

    assert payload["manifest_hash"] == manifest.content_hash
    assert payload["segment"] == "validation"
    assert (payload["period_start"], payload["period_end"]) == ("2023-01-01", "2023-12-31")
    assert payload["silver_dataset_ids"] == {"daily_market": "bars-v1"}
    assert payload["gold_dataset_id"] == "gold-v1"
    assert payload["strategy_id"] == "test-all"
    assert (payload["strategy_policy_hash"], payload["execution_policy_hash"], payload["universe_policy_hash"]) == ("s", "e", "u")
    assert payload["costs"]["commission_rate"] == 0.001
    assert payload["costs"]["total_cost"] > 0
    assert payload["eligible_session_count"] == 5
    assert set(payload["metrics"]) == {"initial_nav", "final_nav", "total_return", "trade_count"}
    assert "created_at" in payload
    assert sorted(path.name for path in result.result_path.parent.iterdir()) == ["manifest.json", "result.json"]


def test_run_scope_bound_backtest_rejects_incomplete_sessions(tmp_path: Path) -> None:
    from src.data.backtest_runner import run_scope_bound_backtest
    from src.data.schemas import PITDataError

    runtime = _scope_runtime(tmp_path)
    bars, universe = _fixture_rows()
    bars = [row for row in bars if row["session"] != SESSIONS_2023[2]]
    _stamp_silver_bars(runtime, "bars-v1", bars)
    _stamp_release(runtime, "gold-v1", universe)
    manifest = _build_manifest(runtime)
    from src.data.backtest_runner import NextSessionExecutionModel as _ExecutionModel

    execution = _ExecutionModel(commission_rate=0.001, tax_rate=0.0025)

    with pytest.raises(PITDataError, match="required completed session"):
        run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)
    assert not (runtime.workspace.runs_root / "backtests" / manifest.content_hash / "result.json").exists()


def test_run_scope_bound_backtest_validates_manifest_binding(tmp_path: Path) -> None:
    import dataclasses

    from src.data.backtest_runner import run_scope_bound_backtest
    from src.data.schemas import PITDataError

    runtime, manifest, execution = _scope_fixture(tmp_path)

    foreign = dataclasses.replace(manifest, scope_hash="0" * 64)
    with pytest.raises(PITDataError, match="scope hash"):
        run_scope_bound_backtest(runtime=runtime, manifest=foreign, strategy=_SignalAll(), execution_model=execution)

    class _Other(_SignalAll):
        strategy_id = "other"

    with pytest.raises(PITDataError, match="does not match the manifest"):
        run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_Other(), execution_model=execution)

    naked = dataclasses.replace(manifest, silver_dataset_ids={})
    with pytest.raises(PITDataError, match="no daily_market"):
        run_scope_bound_backtest(runtime=runtime, manifest=naked, strategy=_SignalAll(), execution_model=execution)

    (runtime.workspace.gold_root / "releases" / "gold-v1" / "release.json").unlink()
    with pytest.raises(PITDataError, match="gold release is missing"):
        run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)

    _stamp_release(runtime, "gold-v1")
    (runtime.workspace.gold_root / "releases" / "gold-v1" / "universe.parquet").unlink()
    with pytest.raises(PITDataError, match="gold universe is missing"):
        run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)


def test_run_scope_bound_backtest_rejects_empty_universe_and_bad_inputs(tmp_path: Path) -> None:
    from src.data.backtest_runner import run_scope_bound_backtest
    from src.data.schemas import PITDataError

    runtime = _scope_runtime(tmp_path)
    _stamp_silver_bars(runtime, "bars-v1", None)
    _stamp_release(runtime, "gold-v1", _fixture_rows()[1])
    manifest = _build_manifest(runtime)
    from src.data.backtest_runner import NextSessionExecutionModel as _ExecutionModel

    execution = _ExecutionModel(commission_rate=0.001, tax_rate=0.0025)

    with pytest.raises(PITDataError, match="bars are missing"):
        run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)

    _stamp_silver_bars(runtime, "bars-v1", [_bar(SESSIONS_2023[0], A, 100.0)])
    _stamp_release(runtime, "gold-v1", [{"session": date(2022, 12, 30), "instrument_id": A, "eligible": True, "verified": True, "common": True}])
    with pytest.raises(PITDataError, match="no manifest-bound universe sessions"):
        run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)


def test_run_scope_bound_backtest_rejects_invalid_bars_and_universe(tmp_path: Path) -> None:
    from src.data.backtest_runner import run_scope_bound_backtest
    from src.data.schemas import PITDataError

    runtime = _scope_runtime(tmp_path)
    _stamp_silver_bars(runtime, "bars-v1", [{"session": SESSIONS_2023[0], "instrument_id": A, "close": 0.0}])
    _stamp_release(runtime, "gold-v1", _fixture_rows()[1])
    manifest = _build_manifest(runtime)
    from src.data.backtest_runner import NextSessionExecutionModel as _ExecutionModel

    execution = _ExecutionModel(commission_rate=0.001, tax_rate=0.0025)

    with pytest.raises(PITDataError, match="non-positive close"):
        run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)

    pl.DataFrame([{"session": SESSIONS_2023[0], "instrument_id": A}]).write_parquet(
        runtime.workspace.silver_root / "daily_market" / "bars-v1" / "bars.parquet"
    )
    with pytest.raises(PITDataError, match="invalid schema"):
        run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)

    release_dir = runtime.workspace.gold_root / "releases" / "gold-v1"
    pl.DataFrame([{"session": SESSIONS_2023[0], "instrument_id": A}]).write_parquet(release_dir / "universe.parquet")
    _stamp_silver_bars(runtime, "bars-v1", [{"session": day, "instrument_id": A, "close": 10.0} for day in SESSIONS_2023])
    with pytest.raises(PITDataError, match="invalid schema"):
        run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)


def test_run_scope_bound_backtest_reuses_identical_result(tmp_path: Path) -> None:
    from src.data.backtest_runner import run_scope_bound_backtest

    runtime, manifest, execution = _scope_fixture(tmp_path)
    first = run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)
    before = first.result_path.read_bytes()
    second = run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)

    assert second.result_path == first.result_path
    assert second.result_path.read_bytes() == before

    first.result_path.write_text("{broken", encoding="utf-8")
    third = run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)
    assert third.metrics["trade_count"] == 7.0


def test_next_session_execution_model_convention(tmp_path: Path) -> None:
    from datetime import date as _date

    import pytest

    from src.data.backtest_runner import NextSessionExecutionModel
    from src.data.schemas import PITDataError

    model = NextSessionExecutionModel(commission_rate=0.001, tax_rate=0.0025)
    assert model.next_session(_date(2023, 1, 2), SESSIONS_2023) == _date(2023, 1, 3)
    assert model.next_session(SESSIONS_2023[-1], SESSIONS_2023) is None
    with pytest.raises(PITDataError, match="not a trade session"):
        model.next_session(_date(2020, 1, 1), SESSIONS_2023)


def test_run_scope_bound_backtest_accepts_mixed_session_formats(tmp_path: Path) -> None:
    from datetime import datetime

    from src.data.backtest_runner import run_scope_bound_backtest

    runtime = _scope_runtime(tmp_path)
    bars = [
        {"session": datetime(2023, 1, 2, 9, 0), "instrument_id": A, "close": 100.0},
        {"session": datetime(2023, 1, 3, 9, 0), "instrument_id": A, "close": 110.0},
    ]
    universe = [
        {"session": "2023-01-02", "instrument_id": A, "eligible": True, "verified": True, "common": True},
        {"session": "2023-01-03", "instrument_id": A, "eligible": True, "verified": True, "common": True},
    ]
    _stamp_silver_bars(runtime, "bars-v1", bars)
    _stamp_release(runtime, "gold-v1", universe)
    manifest = _build_manifest(runtime)
    from src.data.backtest_runner import NextSessionExecutionModel as _ExecutionModel

    execution = _ExecutionModel(commission_rate=0.001, tax_rate=0.0025)

    result = run_scope_bound_backtest(runtime=runtime, manifest=manifest, strategy=_SignalAll(), execution_model=execution)
    assert result.metrics["trade_count"] == 1.0
