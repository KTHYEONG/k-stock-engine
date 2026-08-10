"""Cost-evidence loader and shared fill-cost resolution contract tests."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.core.paths import DATA_ROOT
from src.stocks.data.contracts import CoverageRange
from src.stocks.data.costs import (
    CostEvidence,
    krx_market_for_code,
    load_cost_evidence,
    resolve_fill_cost,
)
from tests.fixtures.stocks.helpers import cost_evidence_fixture

REQUIRED = CoverageRange(start=date(2024, 1, 1), end=date(2024, 3, 31))
STT_URI = "https://www.law.go.kr/lsRvsRsnListP.do?lsId=005028"
STT_HASH = "s" * 64


def artifact_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "coverage": {"start": "2024-01-01", "end": "2024-03-31"},
        "assumption_id": "test_kis_v1",
        "sources": [
            {
                "uri": STT_URI,
                "retrieved_at": "2026-01-01T00:00:00Z",
                "content_hash": STT_HASH,
            }
        ],
        "commission": [
            {"effective_from": "2024-01-01", "buy_rate": 0.000036396, "sell_rate": 0.000036396}
        ],
        "sell_taxes": [
            {
                "effective_from": "2024-01-01",
                "market": "KOSPI",
                "securities_transaction_tax_rate": 0.0003,
                "rural_special_tax_rate": 0.0015,
                "sell_tax_rate": 0.0018,
                "source_uri": STT_URI,
                "source_hash": STT_HASH,
            },
            {
                "effective_from": "2024-01-01",
                "market": "KOSDAQ",
                "securities_transaction_tax_rate": 0.0018,
                "rural_special_tax_rate": 0.0,
                "sell_tax_rate": 0.0018,
                "source_uri": STT_URI,
                "source_hash": STT_HASH,
            },
        ],
        "tick_size_rules": [
            {"rule_id": f"krx_test_{i}", "effective_from": "2024-01-01", "lower_inclusive": lo, "upper_exclusive": hi, "tick": tick}
            for i, (lo, hi, tick) in enumerate(
                (
                    (0.0, 1000.0, 1.0),
                    (1000.0, 5000.0, 5.0),
                    (5000.0, 10000.0, 10.0),
                    (10000.0, 50000.0, 50.0),
                    (50000.0, 100000.0, 100.0),
                    (100000.0, 500000.0, 500.0),
                    (500000.0, None, 1000.0),
                )
            )
        ],
        "liquidity_model": {"model_id": "sqrt_impact_v1", "impact_coefficient": 0.1, "stress_multiplier": 1.5},
        "settlement_days": 2,
    }


def write_artifact(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "costs.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


class TestLoadCostEvidence:
    def test_loads_real_counterfactual_artifact(self) -> None:
        path = (
            DATA_ROOT
            / "evidence"
            / "stocks"
            / "costs"
            / "kis_lifetime_preferential_counterfactual_v1.json"
        )
        evidence = load_cost_evidence(
            path,
            CoverageRange(start=date(2016, 1, 4), end=date(2026, 3, 3)),
        )
        assert isinstance(evidence, CostEvidence)
        assert evidence.assumption_id == "kis_banKIS_lifetime_preferential_counterfactual_v1"
        assert evidence.content_hash
        assert len(evidence.sell_taxes) == 14

    def test_resolves_effective_dated_sell_tax(self, tmp_path) -> None:
        evidence = load_cost_evidence(write_artifact(tmp_path, artifact_payload()), REQUIRED)
        early = evidence.sell_tax_for("KOSPI", datetime(2024, 1, 15, tzinfo=UTC))
        assert early.securities_transaction_tax_rate == 0.0003
        assert early.sell_tax_rate == 0.0018
        kospi = evidence.sell_tax_for("KOSPI", datetime(2024, 2, 1, tzinfo=UTC))
        kosdaq = evidence.sell_tax_for("KOSDAQ", datetime(2024, 2, 1, tzinfo=UTC))
        assert kospi.rural_special_tax_rate == 0.0015
        assert kosdaq.rural_special_tax_rate == 0.0
        assert kospi.securities_transaction_tax_rate != kosdaq.securities_transaction_tax_rate

    def test_malformed_json_rejected(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid cost evidence"):
            load_cost_evidence(path, REQUIRED)

    def test_unknown_schema_version_rejected(self, tmp_path) -> None:
        payload = artifact_payload()
        payload["schema_version"] = 99
        with pytest.raises(ValueError, match="schema_version"):
            load_cost_evidence(write_artifact(tmp_path, payload), REQUIRED)

    def test_range_gap_rejected(self, tmp_path) -> None:
        payload = artifact_payload()
        payload["coverage"] = {"start": "2024-02-01", "end": "2024-03-31"}
        with pytest.raises(ValueError, match="does not contain"):
            load_cost_evidence(write_artifact(tmp_path, payload), REQUIRED)

    def test_unsorted_effective_dates_rejected(self, tmp_path) -> None:
        payload = artifact_payload()
        taxes = payload["sell_taxes"]
        assert isinstance(taxes, list)
        taxes.append(
            {
                "effective_from": "2024-02-01",
                "market": "KOSPI",
                "securities_transaction_tax_rate": 0.0003,
                "rural_special_tax_rate": 0.0015,
                "sell_tax_rate": 0.0018,
                "source_uri": STT_URI,
                "source_hash": STT_HASH,
            }
        )
        taxes[0]["effective_from"] = "2024-03-01"
        with pytest.raises(ValueError, match="sorted ascending"):
            load_cost_evidence(write_artifact(tmp_path, payload), REQUIRED)

    def test_tax_sum_mismatch_rejected(self, tmp_path) -> None:
        payload = artifact_payload()
        taxes = payload["sell_taxes"]
        assert isinstance(taxes, list)
        taxes[0]["sell_tax_rate"] = 0.9999
        with pytest.raises(ValueError, match="sum of its components"):
            load_cost_evidence(write_artifact(tmp_path, payload), REQUIRED)

    def test_unsupported_market_rejected(self, tmp_path) -> None:
        payload = artifact_payload()
        taxes = payload["sell_taxes"]
        assert isinstance(taxes, list)
        taxes[0]["market"] = "KOSDAQ-VENTURE"
        with pytest.raises(ValueError, match="unsupported sell_tax market"):
            load_cost_evidence(write_artifact(tmp_path, payload), REQUIRED)

    def test_negative_rate_rejected(self, tmp_path) -> None:
        payload = artifact_payload()
        taxes = payload["sell_taxes"]
        assert isinstance(taxes, list)
        taxes[0]["rural_special_tax_rate"] = -0.001
        with pytest.raises(ValueError, match="non-negative"):
            load_cost_evidence(write_artifact(tmp_path, payload), REQUIRED)

    def test_missing_source_uri_rejected(self, tmp_path) -> None:
        payload = artifact_payload()
        taxes = payload["sell_taxes"]
        assert isinstance(taxes, list)
        del taxes[0]["source_uri"]
        with pytest.raises(ValueError, match="missing"):
            load_cost_evidence(write_artifact(tmp_path, payload), REQUIRED)

    def test_unknown_source_uri_rejected(self, tmp_path) -> None:
        payload = artifact_payload()
        taxes = payload["sell_taxes"]
        assert isinstance(taxes, list)
        taxes[0]["source_uri"] = "https://unknown.example"
        with pytest.raises(ValueError, match="unknown source"):
            load_cost_evidence(write_artifact(tmp_path, payload), REQUIRED)

    def test_tick_band_gap_rejected(self, tmp_path) -> None:
        payload = artifact_payload()
        rules = payload["tick_size_rules"]
        assert isinstance(rules, list)
        rules[1]["lower_inclusive"] = 1200.0
        with pytest.raises(ValueError, match="gap or overlap"):
            load_cost_evidence(write_artifact(tmp_path, payload), REQUIRED)


class TestKrxMarketForCode:
    def test_resolves_kospi_and_kosdaq(self) -> None:
        assert krx_market_for_code("KRX:000050") == "KOSPI"
        assert krx_market_for_code("KRX:110020") == "KOSDAQ"

    def test_unknown_code_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot resolve"):
            krx_market_for_code("KRX:900999")


class TestResolveFillCost:
    def test_buy_has_no_sell_tax_applied(self) -> None:
        evidence = cost_evidence_fixture()
        when = datetime(2024, 1, 15, tzinfo=UTC)
        breakdown, artifact_hash = resolve_fill_cost(
            evidence,
            side="BUY",
            market="KOSPI",
            price=10_000.0,
            notional=5_000_000.0,
            adtv_20d=1e9,
            daily_volatility=0.02,
            effective_time=when,
        )
        assert breakdown.sell_tax_rate == 0.0018
        assert breakdown.total_rate(side="BUY") == pytest.approx(
            breakdown.commission_rate + breakdown.slippage_bps / 10_000
        )
        assert artifact_hash == "fixture-cost-hash"

    def test_sell_applies_statutory_tax_components(self) -> None:
        evidence = cost_evidence_fixture()
        when = datetime(2024, 1, 15, tzinfo=UTC)
        breakdown, _ = resolve_fill_cost(
            evidence,
            side="SELL",
            market="KOSPI",
            price=10_000.0,
            notional=5_000_000.0,
            adtv_20d=1e9,
            daily_volatility=0.02,
            effective_time=when,
        )
        assert breakdown.total_rate(side="SELL") == pytest.approx(
            breakdown.total_rate(side="BUY") + breakdown.sell_tax_rate
        )
        assert breakdown.sell_tax_rate == pytest.approx(
            breakdown.securities_transaction_tax_rate
            + breakdown.rural_special_tax_rate
        )

    def test_stress_uses_artifact_multiplier(self) -> None:
        evidence = cost_evidence_fixture()
        when = datetime(2024, 1, 15, tzinfo=UTC)
        base, _ = resolve_fill_cost(
            evidence, side="BUY", market="KOSPI", price=10_000.0, notional=5_000_000.0,
            adtv_20d=1e9, daily_volatility=0.02, effective_time=when, stress=False,
        )
        stress, _ = resolve_fill_cost(
            evidence, side="BUY", market="KOSPI", price=10_000.0, notional=5_000_000.0,
            adtv_20d=1e9, daily_volatility=0.02, effective_time=when, stress=True,
        )
        assert stress.slippage_bps > base.slippage_bps
        assert stress.params_hash != base.params_hash

    def test_missing_volatility_fails_closed(self) -> None:
        evidence = cost_evidence_fixture()
        when = datetime(2024, 1, 15, tzinfo=UTC)
        with pytest.raises(ValueError, match="daily_volatility"):
            resolve_fill_cost(
                evidence, side="BUY", market="KOSPI", price=10_000.0, notional=5_000_000.0,
                adtv_20d=1e9, daily_volatility=0.0, effective_time=when,
            )
