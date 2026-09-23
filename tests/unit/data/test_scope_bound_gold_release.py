from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.data.gold import create_scope_bound_gold_release
from src.data.runtime import load_data_runtime
from src.data.schemas import PITDataError
from src.data.scope_coverage import CoverageRequirement, ScopeCoverageReport

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")


def _runtime(tmp_path: Path, *, scope_file: str | None = None, flow_enabled: bool = False):
    if scope_file is None:
        # 정본 스코프의 플래그 값과 무관하게 비활성 소스 기록 의미론을 검증한다.
        disabled = tmp_path / "canonical_flags_off.toml"
        text = SCOPE_CONFIG.read_text(encoding="utf-8")
        text = text.replace("investor_flow_enabled = true", "investor_flow_enabled = false").replace("industry_enabled = true", "industry_enabled = false")
        disabled.write_text(text, encoding="utf-8")
        return load_data_runtime(scope_config=disabled, data_root=tmp_path / "data")
    config_path = tmp_path / scope_file
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join([
            'scope_id = "flow_scope"',
            'evidence_start = "2019-01-01"',
            'development_start = "2020-01-01"',
            'development_end = "2022-12-31"',
            'validation_start = "2023-01-01"',
            'validation_end = "2023-12-31"',
            'holdout_start = "2024-01-01"',
            'holdout_end = "2025-12-31"',
            'forward_start = "2026-01-01"',
            "[features]",
            "price_lookback_sessions = 252",
            "fundamental_lookback_quarters = 5",
            'fundamental_fiscal_start = "2019Q1"',
            f"investor_flow_enabled = {'true' if flow_enabled else 'false'}",
            "industry_enabled = false",
            "[collection]",
            "dart_daily_budget = 16000",
            "dart_batch_identities = 500",
        ]),
        encoding="utf-8",
    )
    return load_data_runtime(scope_config=config_path, data_root=tmp_path / "data")


def _req(source: str, key: str) -> CoverageRequirement:
    return CoverageRequirement(source=source, natural_key=key, as_of=date(2023, 1, 2), fiscal_period=None, required=True)


def _report(runtime, *, missing: tuple = (), unresolved: tuple = ()):
    fulfilled = (
        _req("krx_daily_market", "2023-01-02"),
        _req("krx_security_master", "universe"),
        CoverageRequirement(source="financial_facts", natural_key="00126380:2023:11013", as_of=date(2023, 5, 15), fiscal_period="2023Q1", required=True),
    )
    return ScopeCoverageReport(scope_hash=runtime.scope.content_hash, fulfilled=fulfilled, missing=tuple(missing), unresolved=tuple(unresolved))


def _create(runtime, **overrides):
    params = {
        "dataset_id": "gold-v1",
        "silver_dataset_ids": {"daily_market": "bars-v1"},
        "universe_policy_hash": "u",
        "feature_policy_hash": "f",
    }
    params.update(overrides)
    return create_scope_bound_gold_release(runtime=runtime, coverage_report=_report(runtime), **params)


def test_create_scope_bound_gold_release_stamps_scope_metadata(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    metadata = _create(runtime)

    assert metadata.scope_id == "kr_swing_2019_v1"
    assert metadata.scope_hash == runtime.scope.content_hash
    assert metadata.silver_dataset_ids == {"daily_market": "bars-v1"}
    assert metadata.disabled_sources == ("industry", "investor_flow")
    assert len(metadata.content_hash) == 64
    release_path = runtime.workspace.gold_root / "releases" / "gold-v1" / "release.json"
    stored = json.loads(release_path.read_text(encoding="utf-8"))
    assert stored["content_hash"] == metadata.content_hash
    assert stored["coverage_report_hash"] == metadata.coverage_report_hash

    again = _create(runtime)
    assert again.content_hash == metadata.content_hash


def test_create_scope_bound_gold_release_blocks_unresolved_coverage(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    with pytest.raises(PITDataError, match="unresolved required"):
        create_scope_bound_gold_release(
            runtime=runtime, dataset_id="gold-v1", silver_dataset_ids={"daily_market": "bars-v1"},
            universe_policy_hash="u", feature_policy_hash="f",
            coverage_report=_report(runtime, missing=(_req("krx_daily_market", "2023-01-03"),)),
        )
    with pytest.raises(PITDataError, match="unresolved required"):
        create_scope_bound_gold_release(
            runtime=runtime, dataset_id="gold-v1", silver_dataset_ids={"daily_market": "bars-v1"},
            universe_policy_hash="u", feature_policy_hash="f",
            coverage_report=_report(
                runtime,
                unresolved=(CoverageRequirement(source="financial_facts", natural_key="x", as_of=None, fiscal_period="2023Q1", required=True),),
            ),
        )


def test_create_scope_bound_gold_release_records_disabled_sources(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    metadata = create_scope_bound_gold_release(
        runtime=runtime, dataset_id="gold-v1", silver_dataset_ids={"daily_market": "bars-v1"},
        universe_policy_hash="u", feature_policy_hash="f",
        coverage_report=_report(runtime, missing=(_req("investor_flow", "005930:2023-01-02"), _req("industry", "ind-1"))),
    )

    assert metadata.disabled_sources == ("industry", "investor_flow")

    flow_runtime = _runtime(tmp_path / "flow", scope_file="scope.toml", flow_enabled=True)
    with pytest.raises(PITDataError, match="unresolved required"):
        create_scope_bound_gold_release(
            runtime=flow_runtime, dataset_id="gold-v1", silver_dataset_ids={"daily_market": "bars-v1"},
            universe_policy_hash="u", feature_policy_hash="f",
            coverage_report=_report(flow_runtime, missing=(_req("investor_flow", "005930:2023-01-02"),)),
        )


def test_create_scope_bound_gold_release_rejects_malformed_requests(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    with pytest.raises(PITDataError, match="dataset_id"):
        _create(runtime, dataset_id="a/b")
    with pytest.raises(PITDataError, match="dataset_id"):
        _create(runtime, dataset_id="  ")
    with pytest.raises(PITDataError, match="non-empty mapping"):
        _create(runtime, silver_dataset_ids={})
    with pytest.raises(PITDataError, match="universe_policy_hash"):
        _create(runtime, universe_policy_hash="")
    with pytest.raises(PITDataError, match="feature_policy_hash"):
        _create(runtime, feature_policy_hash="  ")
