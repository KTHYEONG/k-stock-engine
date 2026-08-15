"""CostEvidence effective-dated schedule conversion tests."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.fixtures.stocks.helpers import cost_evidence_fixture


def test_base_schedule_resolves_commission_and_sell_tax() -> None:
    evidence = cost_evidence_fixture()
    schedule = evidence.base_schedule(market="KOSPI")
    point = schedule.cost_for(datetime(2024, 2, 1, tzinfo=UTC))
    assert point.commission_rate == 0.000036396
    assert point.tax_rate == 0.0003 + 0.0015
    assert point.slippage_bps == 0.0


def test_stress_schedule_resolves_sell_side_commission_and_tax() -> None:
    evidence = cost_evidence_fixture()
    schedule = evidence.stress_schedule(market="KOSDAQ")
    point = schedule.cost_for(datetime(2024, 2, 1, tzinfo=UTC))
    assert point.commission_rate == 0.000036396
    assert point.tax_rate == 0.0018
    assert schedule.name == "fixture_kis_v1-stress"


def test_schedule_points_fail_closed_before_coverage() -> None:
    evidence = cost_evidence_fixture(effective_from=datetime(2024, 1, 1, tzinfo=UTC))
    schedule = evidence.base_schedule()
    with pytest.raises(ValueError, match="no cost coverage"):
        schedule.cost_for(datetime(2023, 12, 31, tzinfo=UTC))
