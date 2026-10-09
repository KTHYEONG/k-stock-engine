"""Frozen revision vocabulary and single-source duplicate guards."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import Mock

from src.core import market_rules
from src.core.digest import dataset_digest
from src.data import (
    cash_series_silver,
    daily_market_silver,
    datasets,
    dividend_events,
    earnings_releases,
    fact_page_meta,
    hedge_series_silver,
    incremental_normalization,
    industry_silver,
    investor_flow_kis_supplement,
    investor_flow_silver,
    investor_flow_union,
    market_actions,
    market_panel,
    normalization,
    ordinary_universe,
    ordinary_universe_price_audit,
    pipeline_graph,
    receipt_catalog,
    reference_benchmarks,
)
from src.data.datasets import DatasetIdentity, DatasetLayer, dataset_id_for
from src.execution.domain import intents
from src.integrations.dart import accounts, document_statements
from src.research import cube, registry

_FROZEN: tuple[tuple[object, str, object], ...] = (
    (investor_flow_union, "REVISION", "investor-flow-ls-kis-union-v1"),
    (daily_market_silver, "REVISION", "krx-daily-market-v1"),
    (earnings_releases, "REVISION", "earnings-releases-v2"),
    (dividend_events, "REVISION", "dividend-events-v3"),
    (hedge_series_silver, "REVISION", "krx-hedge-series-v1"),
    (investor_flow_silver, "REVISION", "ls-t1702-net-shares-v2"),
    (investor_flow_kis_supplement, "REVISION", "kis-investor-trade-net-shares-supplement-v2"),
    (industry_silver, "REVISION", "kis-industry-classification-v2"),
    (market_panel, "REVISION", "krx-market-panel-v4"),
    (market_actions, "REVISION", "market-actions-v10"),
    (cash_series_silver, "REVISION", "krx-cash-series-v1"),
    (ordinary_universe, "REVISION", "krx-ordinary-equity-v1"),
    (reference_benchmarks, "REVISION", "reference-benchmarks-v1"),
    (accounts, "REVISION", "dart-fact-map-v1"),
    (document_statements, "REVISION", "dart-document-statements-v3"),
    (incremental_normalization, "REVISION", "dart-incremental-v1"),
    (cube, "REVISION", "research-cube-v2"),
    (market_rules, "REVISION", "krx-market-rules-v1"),
    (fact_page_meta, "REVISION", 1),
    (datasets, "SCHEMA_VERSION", "dataset-manifest-v2"),
    (receipt_catalog, "_SCHEMA_VERSION", 2),
    (intents, "SCHEMA_VERSION", "v2"),
    (registry, "_SCHEMA_VERSION", 2),
)

_RETIRED = (
    "POLICY_VERSION",
    "PARSER_VERSION",
    "MAPPING_VERSION",
    "DEFINITIONS_VERSION",
    "CUBE_POLICY_VERSION",
    "DERIVATION_VERSION",
    "MANIFEST_SCHEMA",
    "INTENT_SCHEMA_VERSION",
    "_INDEX_SCHEMA",
    "_CATALOG_SCHEMA_VERSION",
)


def test_revision_values_are_frozen() -> None:
    for module, name, expected in _FROZEN:
        actual = getattr(module, name)
        assert actual == expected
        assert type(actual) is type(expected)


def test_duplicates_resolve_to_single_source() -> None:
    assert not hasattr(normalization, "_DART_MAPPING_VERSION")
    assert normalization.accounts.REVISION is accounts.REVISION
    records = normalization._flatten_dart_fact_pages([{"records": [{"fact": "assets"}]}])
    assert records[0]["mapping_version"] is accounts.REVISION
    assert ordinary_universe_price_audit.DATASETS_SCHEMA_VERSION is datasets.SCHEMA_VERSION
    root = Path(__file__).resolve().parents[2]
    normalization_src = (root / "src/data/normalization.py").read_text(encoding="utf-8")
    pipeline_src = (root / "src/data/pipeline_graph.py").read_text(encoding="utf-8")
    audit_src = (root / "src/data/ordinary_universe_price_audit.py").read_text(encoding="utf-8")
    assert "dart-fact-map-v1" not in normalization_src
    assert "accounts.REVISION" in normalization_src
    assert "dart-incremental-v1" not in pipeline_src
    assert "INCREMENTAL_NORMALIZATION_REVISION" in pipeline_src
    assert "dataset-manifest-v2" not in audit_src


def test_no_retired_identifier_survives() -> None:
    pattern = re.compile(r"\b(?:" + "|".join(_RETIRED) + r")\b")
    root = Path(__file__).resolve().parents[2]
    offenders = []
    for base in (root / "src", root / "tests"):
        for path in sorted(base.rglob("*.py")):
            if path == Path(__file__).resolve():
                continue
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_dataset_identities_are_unchanged() -> None:
    bronze = dataset_digest(["a", "b"])
    universe = DatasetIdentity(
        kind="ordinary_universe",
        layer=DatasetLayer.SILVER,
        policy_version=ordinary_universe.REVISION,
        inputs={"bronze_master": bronze},
        params={"calendar_digest": "cafef00d"},
    )
    panel = DatasetIdentity(
        kind="market_panel",
        layer=DatasetLayer.GOLD,
        policy_version=market_panel.REVISION,
        inputs={"daily_market": "daily_market_0123456789abcdef"},
        params={"rules_version": market_rules.REVISION},
    )
    assert dataset_id_for(universe) == "ordinary_universe_a5b34433e8542c70"
    assert dataset_id_for(panel) == "market_panel_056b25ac22c2b31d"


def test_pipeline_preview_identities_match_pre_rename_goldens() -> None:
    ctx = Mock(spec=pipeline_graph.RefreshContext)
    ctx.registry = Mock()
    ctx.registry.require.side_effect = {
        "investor_flow_ls": "investor_flow_ls_0123456789abcdef",
        "investor_flow_kis_supplement": "investor_flow_kis_supplement_fedcba9876543210",
    }.__getitem__
    ctx.catalog = Mock()
    ctx.catalog.blob_digest.return_value = dataset_digest(["a", "b"])

    union = pipeline_graph._preview_investor_flow(ctx)
    industry = pipeline_graph._preview_industry(ctx)

    assert dataset_id_for(union) == "investor_flow_4564d9ab653f3370"
    assert dataset_id_for(industry) == "industry_a30a33c05533fdc5"
