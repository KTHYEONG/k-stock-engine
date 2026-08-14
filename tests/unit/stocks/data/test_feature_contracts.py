"""Feature contracts: label-free projections and explicit duplicate lineage."""
from __future__ import annotations

from datetime import datetime

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
