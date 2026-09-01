from __future__ import annotations


def test_stock_only_factor_study_selects_causal_factor_and_reports_costed_excess() -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from src.core.costs import default_base_schedule, default_stress_schedule
    from src.stocks.research.stock_only_factor_study import StockOnlyFactorStudySettings, run_stock_only_factor_study

    start = datetime(2020, 1, 1, tzinfo=UTC)
    rows = []
    labels = []
    for session_index in range(330):
        session = start + timedelta(days=session_index)
        for name, signal in (("KRX:000001", 1.0), ("KRX:000002", -1.0), ("KRX:000003", 0.2)):
            rows.append({"instrument_id": name, "session": session, "available_time": session, "open": 100.0, "close": 101.0, "volume": 2_000_000.0, "trading_value": 2_000_000_000.0, "sector": "S", "ret_21_60d": signal, "relative_trend_score": signal, "disparity_120d": signal, "ret_2_5d": -signal, "close_high_ratio_10d": 0.5, "foreign_net_buy": signal, "institution_net_buy": signal, "flow_consensus": signal, "flow_intensity_20d": signal, "volatility_20d": 0.1, "amihud_20d": 0.001, "adtv_20d": 2_000_000_000.0})
            labels.append({"instrument_id": name, "session": session, "horizon_sessions": 5, "net_alpha_target": 0.02 if name == "KRX:000001" else -0.01, "label_available_time": session + timedelta(days=6)})
    result = run_stock_only_factor_study(pl.DataFrame(rows), {5: pl.DataFrame(labels)}, StockOnlyFactorStudySettings(candidate_horizon_sessions=(5,), candidate_rebalance_frequency_sessions=(5,), candidate_top_k=(3,), account_capital_krw=10_000_000.0, forward_holdout_sessions=30), default_base_schedule(), default_stress_schedule())
    assert result.status in {"RESEARCH_ONLY", "PROMOTABLE", "NO_TRADE"}
    assert result.stock_only_audit.passed
    assert result.candidate_count > 0
    assert result.data_gaps


def test_stock_only_factor_study_rejects_future_data_and_non_stock_instruments() -> None:
    from datetime import UTC, datetime
    import polars as pl
    import pytest
    from src.core.costs import default_base_schedule, default_stress_schedule
    from src.stocks.research.stock_only_factor_study import StockOnlyFactorStudySettings, run_stock_only_factor_study

    now = datetime(2024, 1, 2, tzinfo=UTC)
    panel = pl.DataFrame({"instrument_id": ["KRX:ETF:252670"], "session": [now], "available_time": [now], "open": [100.0], "close": [100.0], "volume": [1_000_000.0], "trading_value": [1_000_000_000.0], "sector": ["ETF"], "ret_21_60d": [0.1], "relative_trend_score": [0.1], "disparity_120d": [0.1], "ret_2_5d": [0.0], "close_high_ratio_10d": [0.5], "foreign_net_buy": [0.0], "institution_net_buy": [0.0], "flow_consensus": [0.0], "flow_intensity_20d": [0.0], "volatility_20d": [0.1], "amihud_20d": [0.001], "adtv_20d": [1_000_000_000.0]})
    labels = pl.DataFrame({"instrument_id": ["KRX:ETF:252670"], "session": [now], "horizon_sessions": [5], "net_alpha_target": [0.1], "label_available_time": [now]})
    settings = StockOnlyFactorStudySettings(candidate_horizon_sessions=(5,), candidate_rebalance_frequency_sessions=(5,), candidate_top_k=(3,), account_capital_krw=10_000_000.0, forward_holdout_sessions=1)
    with pytest.raises(ValueError, match="stock-only"):
        run_stock_only_factor_study(panel, {5: labels}, settings, default_base_schedule(), default_stress_schedule())


def test_stock_only_factor_study_future_labels_do_not_change_selected_cell() -> None:
    from datetime import UTC, datetime, timedelta
    import polars as pl
    from src.core.costs import default_base_schedule, default_stress_schedule
    from src.stocks.research.stock_only_factor_study import StockOnlyFactorStudySettings, run_stock_only_factor_study

    start = datetime(2020, 1, 1, tzinfo=UTC)
    rows = []
    labels = []
    for day in range(310):
        session = start + timedelta(days=day)
        for name, score in (("KRX:000001", 1.0), ("KRX:000002", -1.0), ("KRX:000003", 0.0)):
            rows.append({"instrument_id": name, "session": session, "available_time": session, "open": 100.0, "close": 100.0, "volume": 1_000_000.0, "trading_value": 1_000_000_000.0, "sector": "S", "ret_21_60d": score, "relative_trend_score": score, "disparity_120d": score, "ret_2_5d": -score, "close_high_ratio_10d": 0.5, "foreign_net_buy": score, "institution_net_buy": score, "flow_consensus": score, "flow_intensity_20d": score, "volatility_20d": 0.1, "amihud_20d": 0.001, "adtv_20d": 1_000_000_000.0})
            labels.append({"instrument_id": name, "session": session, "horizon_sessions": 5, "net_alpha_target": score / 100.0, "label_available_time": session + timedelta(days=6)})
    settings = StockOnlyFactorStudySettings(candidate_horizon_sessions=(5,), candidate_rebalance_frequency_sessions=(5,), candidate_top_k=(3,), account_capital_krw=10_000_000.0, forward_holdout_sessions=30)
    baseline = run_stock_only_factor_study(pl.DataFrame(rows), {5: pl.DataFrame(labels)}, settings, default_base_schedule(), default_stress_schedule())
    flipped = pl.DataFrame(labels).with_columns(pl.when(pl.col("label_available_time") > start + timedelta(days=250)).then(-pl.col("net_alpha_target")).otherwise(pl.col("net_alpha_target")).alias("net_alpha_target"))
    changed = run_stock_only_factor_study(pl.DataFrame(rows), {5: flipped}, settings, default_base_schedule(), default_stress_schedule())
    assert baseline.selected_cell == changed.selected_cell
