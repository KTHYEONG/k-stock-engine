"""Daily-market coverage backfill scenarios (contract skeletons)."""
from __future__ import annotations


def test_backfill_daily_market_coverage_normalizes_then_requires_no_gap(monkeypatch, tmp_path) -> None:
    from datetime import UTC, date, datetime
    import src.data.operations as module
    from src.data.gold_loader import DailyMarketBackfillPlan

    missing = (date(2024, 1, 3), date(2024, 1, 4))
    plans = iter((DailyMarketBackfillPlan(date(2024, 1, 1), date(2024, 1, 4), (), missing), DailyMarketBackfillPlan(date(2024, 1, 1), date(2024, 1, 4), missing, ())))
    monkeypatch.setattr(module, 'plan_daily_market_backfill', lambda **_kwargs: next(plans))
    calls = []
    monkeypatch.setattr(module, 'collect_daily_market_sessions', lambda **kwargs: calls.append(kwargs['sessions']) or object())
    monkeypatch.setattr(module, 'stream_normalize_stock_evidence', lambda **_kwargs: object())
    request = module.DailyMarketBackfillRequest(tmp_path / 'bronze', tmp_path / 'silver', tmp_path / 'artifacts', date(2024, 1, 3), date(2024, 1, 4), datetime(2026, 9, 6, tzinfo=UTC))

    result = module.backfill_daily_market_coverage(request, krx=object())

    assert calls == [missing]
    assert result.missing_sessions == ()
    assert result.backfilled_sessions == missing


def test_backfill_daily_market_coverage_reports_already_covered_and_cli_dispatch(monkeypatch, tmp_path, capsys) -> None:
    from datetime import UTC, date, datetime
    import sys
    import src.data.operations as module
    from src.data.gold_loader import DailyMarketBackfillPlan
    from src.data import cli

    plan = DailyMarketBackfillPlan(date(2024, 1, 1), date(2024, 1, 4), (date(2024, 1, 1),), ())
    monkeypatch.setattr(module, "plan_daily_market_backfill", lambda **_kwargs: plan)
    request = module.DailyMarketBackfillRequest(tmp_path / "bronze", tmp_path / "silver", tmp_path / "artifacts", date(2024, 1, 3), date(2024, 1, 4), datetime(2026, 9, 6, tzinfo=UTC))
    result = module.backfill_daily_market_coverage(request, krx=object())
    assert result.backfilled_sessions == ()
    assert (request.artifact_root / "daily_market_backfill.json").exists()

    monkeypatch.setattr(module, "backfill_daily_market_coverage", lambda request, **_kwargs: result)
    monkeypatch.setattr("src.integrations.krx.historical.KrxHistoricalCollector", lambda **_kwargs: object())
    monkeypatch.setattr(sys, "argv", ["stock-data", "backfill-daily-market", "--bronze-root", str(request.bronze_root), "--silver-root", str(request.silver_root), "--artifact-root", str(request.artifact_root), "--validation-start", "2024-01-03", "--validation-end", "2024-01-04", "--decision-time", "2026-09-06T00:00:00+00:00"])
    assert cli.main() == 0
    assert "backfilled_count" in capsys.readouterr().out

    monkeypatch.setattr(module, "backfill_daily_market_coverage", lambda *_args, **_kwargs: (_ for _ in ()).throw(module.PITDataError("blocked")))
    assert cli.main() == 1


def test_backfill_daily_market_coverage_rejects_invalid_or_partial_requests(monkeypatch, tmp_path) -> None:
    from datetime import UTC, date, datetime
    import pytest
    import src.data.operations as module
    from src.data.gold_loader import DailyMarketBackfillPlan
    from src.data.schemas import PITDataError

    base = module.DailyMarketBackfillRequest(tmp_path / "b", tmp_path / "s", tmp_path / "a", date(2024, 1, 1), date(2024, 1, 2), datetime(2026, 9, 6, tzinfo=UTC))
    with pytest.raises(PITDataError, match="timezone"):
        module.backfill_daily_market_coverage(module.DailyMarketBackfillRequest(base.bronze_root, base.silver_root, base.artifact_root, base.validation_start, base.validation_end, datetime(2026, 9, 6)), krx=object())
    with pytest.raises(PITDataError, match="official"):
        module.backfill_daily_market_coverage(base, krx=None)
    with pytest.raises(PITDataError, match="inverted"):
        module.backfill_daily_market_coverage(module.DailyMarketBackfillRequest(base.bronze_root, base.silver_root, base.artifact_root, base.validation_end, base.validation_start, base.decision_time), krx=object())
    missing = (date(2024, 1, 2),)
    monkeypatch.setattr(module, "plan_daily_market_backfill", lambda **_kwargs: DailyMarketBackfillPlan(date(2024, 1, 1), date(2024, 1, 2), (), missing))
    monkeypatch.setattr(module, "collect_daily_market_sessions", lambda **_kwargs: object())
    monkeypatch.setattr(module, "stream_normalize_stock_evidence", lambda **_kwargs: object())
    with pytest.raises(PITDataError, match="incomplete"):
        module.backfill_daily_market_coverage(base, krx=object())
