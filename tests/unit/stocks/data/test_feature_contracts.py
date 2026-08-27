"""Feature contracts: label-free projections and explicit duplicate lineage."""
from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from src.stocks.data.feature_contracts import (
    DuplicateRule,
    FeatureContractBook,
    feature_contract_book_from_allowlist,
    make_feature_contract,
    resolve_raw_source_names,
    semantic_feature_contract_book,
)

ALLOWLIST = ("total_assets", "total_liabilities", "per", "pbr")
RULES = (
    DuplicateRule(canonical="total_assets", alternatives=("total_assets_right",)),
    DuplicateRule(canonical="total_liabilities", alternatives=("total_liabilities_right",)),
)


def book() -> FeatureContractBook:
    return feature_contract_book_from_allowlist("v1", ALLOWLIST, RULES)


def base_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["KRX:1"] * 2,
            "session": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "raw__total_assets": [1.0, 2.0],
            "raw__total_assets_right": [1.0, 2.0],
            "raw__total_liabilities_right": [0.5, 1.0],
            "raw__per": [10.0, 11.0],
            "raw__pbr": [1.2, 1.3],
        }
    )


class TestFeatureProjection:
    def test_no_target_or_label_columns_in_feature_panel(self) -> None:
        frame = base_frame().with_columns(
            pl.lit(0.5).alias("raw__target_return_5d"),
            pl.lit(0.5).alias("raw__label_should_be_dropped"),
        )
        with pytest.raises(ValueError, match="target_/label_"):
            book().project(frame, source_prefix="raw__")

    def test_duplicate_lineage_resolves_explicitly(self) -> None:
        out = book().project(base_frame(), source_prefix="raw__")
        columns = set(out.columns)
        # Canonical source wins; the alternative is never projected.
        assert "feature__total_assets" in columns
        assert "feature__total_liabilities" in columns
        assert "feature__per" in columns
        assert "feature__pbr" in columns
        assert out["feature__total_assets"].to_list() == [1.0, 2.0]
        assert out["feature__total_liabilities"].to_list() == [0.5, 1.0]

    def test_ambiguous_duplicate_without_canonical_is_rejected(self) -> None:
        # Canonical absent with two alternatives is ambiguous.
        restricted = feature_contract_book_from_allowlist(
            "v1",
            ("total_equity",),
            (
                DuplicateRule(
                    canonical="total_equity",
                    alternatives=("total_equity_right", "total_equity_alt"),
                ),
            ),
        )
        frame = base_frame().select("instrument_id", "session").with_columns(
            pl.lit(1.0).alias("raw__total_equity_right"),
            pl.lit(1.0).alias("raw__total_equity_alt"),
        )
        with pytest.raises(ValueError, match="ambiguous duplicate lineage"):
            restricted.project(frame, source_prefix="raw__")

    def test_resolve_raw_source_names_picks_canonical_and_keeps_single_alternative(
        self,
    ) -> None:
        assert resolve_raw_source_names(
            ("total_assets", "total_assets_right"), RULES
        ) == ("total_assets",)
        assert resolve_raw_source_names(("total_assets_right",), RULES) == (
            "total_assets_right",
        )


class TestContractDeterminism:
    def test_identical_contracts_have_identical_dependency_hashes(self) -> None:
        first = make_feature_contract(name="per", source_field="per")
        second = make_feature_contract(name="per", source_field="per")
        assert first.dependency_hash == second.dependency_hash

    def test_contract_change_changes_dependency_hash(self) -> None:
        base = make_feature_contract(name="per", source_field="per")
        altered = make_feature_contract(name="per", source_field="pbr")
        assert base.dependency_hash != altered.dependency_hash


class TestSemanticContracts:
    def test_semantic_book_requires_role_and_lineage(self) -> None:
        book = semantic_feature_contract_book(
            "stock_alpha_v3",
            (
                {
                    "name": "adtv_20d",
                    "role": "LIQUIDITY",
                    "source_field": "adtv_20d",
                    "source_dataset_ids": ("base_panel",),
                    "source_columns": ("adtv_20d",),
                    "formula_id": "stock_alpha_v3:adtv_20d:v1",
                    "lookback_sessions": 20,
                    "adjustment_basis": "split_adjusted",
                    "null_policy": "retain_null",
                    "stale_after_sessions": 0,
                    "expected_frequency": "session",
                },
            ),
        )
        contract = book.contracts[0]
        assert contract.role == "LIQUIDITY"
        assert contract.lookback_sessions == 20
        assert contract.formula_id == "stock_alpha_v3:adtv_20d:v1"
        assert contract.adjustment_basis == "split_adjusted"
        assert contract.source_dataset_ids == ("base_panel",)

    def test_semantic_book_rejects_invalid_role(self) -> None:
        with pytest.raises(ValueError, match="role must be one of"):
            semantic_feature_contract_book(
                "v1",
                (
                    {
                        "name": "x",
                        "role": "SIGNAL",
                        "source_field": "x",
                        "source_dataset_ids": ("base_panel",),
                        "source_columns": ("x",),
                        "formula_id": "v1:x:v1",
                        "lookback_sessions": 1,
                        "adjustment_basis": "split_adjusted",
                        "null_policy": "retain_null",
                        "stale_after_sessions": 0,
                        "expected_frequency": "session",
                    },
                ),
            )

    def test_semantic_book_rejects_generic_fallback(self) -> None:
        with pytest.raises(ValueError, match="generic fallback"):
            semantic_feature_contract_book(
                "v1",
                (
                    {
                        "name": "x",
                        "role": "ALPHA",
                        "source_field": "x",
                        "source_dataset_ids": ("base_panel",),
                        "source_columns": ("x",),
                        "formula_id": "",
                        "lookback_sessions": 0,
                        "adjustment_basis": "split_adjusted",
                        "null_policy": "retain_null",
                        "stale_after_sessions": 0,
                        "expected_frequency": "session",
                    },
                ),
            )

    def test_semantic_hash_changes_with_lineage(self) -> None:
        base = {
            "name": "ep_ratio",
            "role": "ALPHA",
            "source_field": "ep_ratio",
            "source_dataset_ids": ("base_panel",),
            "source_columns": ("ep_ratio",),
            "formula_id": "stock_alpha_v3:ep_ratio:v1",
            "lookback_sessions": 0,
            "adjustment_basis": "split_adjusted",
            "null_policy": "retain_null",
            "stale_after_sessions": 0,
            "expected_frequency": "session",
        }
        first = semantic_feature_contract_book("v1", (base,)).contracts[0]
        changed = dict(base)
        changed["formula_id"] = "stock_alpha_v3:ep_ratio:v2"
        second = semantic_feature_contract_book("v1", (changed,)).contracts[0]
        assert first.dependency_hash != second.dependency_hash


def test_ALPHA_ARCH_03_PIT_LINEAGE() -> None:
    """ALPHA_ARCH_03_PIT_LINEAGE.

    Explicit 5d, 20d, and 120d lookbacks and DERIVED lineage are hashed;
    bp/ep without disclosure availability are rejected for production.
    """
    from src.stocks.data.ml_integrity import validate_ml_snapshot
    from src.stocks.data.quality import KRXSessionCalendar
    from src.stocks.ml.features import stock_net_alpha_v2_contract_book, stock_net_alpha_v2_semantic_contracts
    import polars as pl
    from datetime import datetime, UTC

    contracts = stock_net_alpha_v2_semantic_contracts()
    by_name = {c["name"]: c for c in contracts}
    assert by_name["disparity_120d"]["lookback_sessions"] == 120
    assert by_name["flow_intensity_20d"]["lookback_sessions"] == 20
    assert by_name["ret_2_5d"]["lookback_sessions"] == 5
    assert by_name["disparity_120d"]["source_kind"] == "derived"
    # hash includes source_available_time_field
    from src.stocks.data.feature_contracts import make_feature_contract

    c1 = make_feature_contract(name="x", source_field="x", source_available_time_field="available_time", formula_id="a")
    c2 = make_feature_contract(name="x", source_field="x", source_available_time_field="disclosure_date", formula_id="a")
    assert c1.dependency_hash != c2.dependency_hash
    # bp/ep without disclosure are rejected
    book = stock_net_alpha_v2_contract_book()
    bp = book.contract_for("bp_ratio")
    assert bp.source_available_time_field == "disclosure_date"
    calendar = KRXSessionCalendar(
        version="test",
        sessions=tuple(date(2024, 1, d) for d in range(1, 12)),
        generated_time=datetime(2024, 1, 1, tzinfo=UTC),
    )
    frame = pl.DataFrame(
        {
            "instrument_id": ["KRX:1"] * 6 + ["KRX:2"] * 6,
            "session": [date(2024, 1, d) for d in range(1, 7)] * 2,
            "observation_time": [datetime(2024, 1, d, 15, 0, tzinfo=UTC) for d in range(1, 7)] * 2,
            "available_time": [datetime(2024, 1, d, 15, 30, tzinfo=UTC) for d in range(1, 7)] * 2,
            "open": [100.0 + i for i in range(6)] * 2,
            "high": [101.0 + i for i in range(6)] * 2,
            "low": [99.0 + i for i in range(6)] * 2,
            "close": [100.5 + i for i in range(6)] * 2,
            "volume": [1_000_000.0] * 12,
        }
    )
    # Add v2 features: need at least bp_ratio/ep_ratio columns, others null
    for name in [c["name"] for c in stock_net_alpha_v2_semantic_contracts()]:
        if name not in frame.columns:
            if name in ("bp_ratio", "ep_ratio"):
                frame = frame.with_columns(pl.lit(1.0).alias(name))
            else:
                frame = frame.with_columns(pl.lit(0.1).alias(name))
    audit = validate_ml_snapshot(frame, book, datetime(2024, 1, 10, tzinfo=UTC), calendar)
    assert audit.passed is False
    assert any(c.name == "pit_availability" and not c.passed for c in audit.checks)
