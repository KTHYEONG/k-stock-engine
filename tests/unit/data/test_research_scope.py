from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

CANONICAL = Path("config/research/kr_swing_2019_v1.toml")


def _load_canonical():  # type: ignore[no-untyped-def]
    from src.data.research_scope import load_research_scope

    return load_research_scope(CANONICAL)


def _scope_payload(**overrides):  # type: ignore[no-untyped-def]
    from src.data.research_scope import load_research_scope

    base = load_research_scope(CANONICAL).model_dump(mode="json")
    for key, value in overrides.items():
        if "." in key:
            top, sub = key.split(".", 1)
            base[top][sub] = value
        else:
            base[key] = value
    return base


def test_canonical_scope_loads_exact_segments() -> None:
    scope = _load_canonical()
    assert scope.scope_id == "kr_swing_2019_v1"
    assert scope.evidence_start == date(2016, 1, 1)
    assert (scope.development_start, scope.development_end) == (date(2020, 1, 1), date(2022, 12, 31))
    assert (scope.validation_start, scope.validation_end) == (date(2023, 1, 1), date(2023, 12, 31))
    assert (scope.holdout_start, scope.holdout_end) == (date(2024, 1, 1), date(2025, 12, 31))
    assert scope.forward_start == date(2026, 1, 1)
    assert scope.features.price_lookback_sessions == 252
    assert scope.features.fundamental_lookback_quarters == 5
    assert scope.features.fundamental_fiscal_start == "2016Q1"
    assert scope.completed_start == date(2020, 1, 1)
    assert scope.completed_end == date(2025, 12, 31)
    assert scope.classify_completed_date(date(2021, 6, 1)) == "development"
    assert scope.classify_completed_date(date(2023, 6, 1)) == "validation"
    assert scope.classify_completed_date(date(2024, 6, 1)) == "holdout"
    assert scope.contains_evidence_date(date(2016, 1, 1))
    assert scope.require_fiscal_period("2016Q1")
    assert scope.require_fiscal_period("2015Q4") is False


def test_scope_hash_is_path_independent(tmp_path: Path) -> None:
    from src.data.research_scope import load_research_scope

    raw = CANONICAL.read_bytes()
    first = tmp_path / "a.toml"
    second = tmp_path / "sub" / "b.toml"
    second.parent.mkdir(parents=True)
    first.write_bytes(raw)
    second.write_bytes(raw)
    assert load_research_scope(first).content_hash == load_research_scope(second).content_hash


def test_scope_hash_changes_with_policy() -> None:
    from src.data.research_scope import ResearchScope

    base = ResearchScope.model_validate(_scope_payload())
    changed = ResearchScope.model_validate(_scope_payload(**{"features.price_lookback_sessions": 100}))
    assert base.content_hash != changed.content_hash


def test_noncontiguous_segments_fail_closed() -> None:
    from pydantic import ValidationError

    from src.data.research_scope import ResearchScope

    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(validation_start="2023-01-05"))
    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(validation_start="2022-12-31"))
    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(holdout_start="2024-01-05"))
    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(forward_start="2026-01-05"))
    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(evidence_start="2020-01-02"))
    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(development_start="2022-12-31", development_end="2020-01-01"))
    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(validation_start="2023-06-01", validation_end="2023-01-01"))
    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(holdout_start="2024-06-01", holdout_end="2024-01-01"))


def test_forward_period_is_not_historical() -> None:
    scope = _load_canonical()
    with pytest.raises(ValueError, match="outside completed"):
        scope.classify_completed_date(date(2026, 1, 1))
    with pytest.raises(ValueError, match="outside completed"):
        scope.classify_completed_date(date(2026, 6, 15))


def test_pre_2016_evidence_is_rejected() -> None:
    scope = _load_canonical()
    assert not scope.contains_evidence_date(date(2015, 12, 31))
    assert scope.contains_evidence_date(date(2016, 1, 1))
    assert scope.contains_evidence_date(date(2025, 12, 31))


def test_invalid_fiscal_floor_fails() -> None:
    from pydantic import ValidationError

    from src.data.research_scope import ResearchScope

    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(**{"features.fundamental_fiscal_start": "2015Q4"}))
    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(**{"features.fundamental_fiscal_start": "2019Q5"}))
    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(**{"features.fundamental_fiscal_start": "FY2019"}))


def test_fiscal_start_at_2016_floor_validates() -> None:
    from src.data.research_scope import ResearchScope

    scope = ResearchScope.model_validate(_scope_payload(**{"features.fundamental_fiscal_start": "2016Q1"}))

    assert scope.features.fundamental_fiscal_start == "2016Q1"


def test_preferred_share_excluded_within_extended_window() -> None:
    from src.data.ordinary_universe import classify_krx_master_row

    eligible, reason = classify_krx_master_row({
        "ISU_SRT_CD": "000001",
        "ISU_CD": "KR0000000001",
        "KIND_STKCERT_TP_NM": "우선주",
        "SECUGRP_NM": "주권",
        "MKT_TP_NM": "KOSPI",
        "LIST_DD": "20100101",
    })

    assert eligible is False
    assert reason == "non_ordinary_share"


def test_canonical_scope_enables_flow_and_industry() -> None:
    scope = _load_canonical()
    assert scope.features.investor_flow_enabled is True
    assert scope.features.industry_enabled is True


def test_budget_headroom_is_validated() -> None:
    from pydantic import ValidationError

    from src.data.research_scope import ResearchScope

    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(**{"collection.dart_batch_identities": 0}))
    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(**{"collection.dart_daily_budget": 20000}))
    with pytest.raises(ValidationError):
        ResearchScope.model_validate(_scope_payload(**{"collection.dart_daily_budget": 25000}))
    with pytest.raises(ValidationError):
        ResearchScope.model_validate(
            _scope_payload(**{"collection.dart_daily_budget": 400, "collection.dart_daily_reserve": 400})
        )


def test_invalid_scope_id_fails() -> None:
    from pydantic import ValidationError

    from src.data.research_scope import ResearchScope

    for bad in ("kr/swing", "kr\\swing", "kr swing", "KR_SWING", "has..dots", ""):
        with pytest.raises(ValidationError):
            ResearchScope.model_validate(_scope_payload(scope_id=bad))


def test_require_fiscal_period_bounds() -> None:
    scope = _load_canonical()
    assert scope.require_fiscal_period("2019Q1") is True
    assert scope.require_fiscal_period("2024Q3") is True
    with pytest.raises(ValueError, match="fiscal period"):
        scope.require_fiscal_period("2019Q0")
    with pytest.raises(ValueError, match="fiscal period"):
        scope.require_fiscal_period("not-a-period")


def test_workspace_layout_is_scope_namespaced(tmp_path: Path) -> None:
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=CANONICAL, data_root=tmp_path / "data")
    runtime.workspace.initialize()
    expected = {
        runtime.workspace.bronze_root,
        runtime.workspace.silver_root,
        runtime.workspace.gold_root,
        runtime.workspace.state_root,
        runtime.workspace.runs_root,
    }
    for root in expected:
        assert root.is_dir()
        assert root.name == "kr_swing_2019_v1"
    created = {p for p in (tmp_path / "data").rglob("*") if p.is_dir()}
    assert expected <= created
    assert len([p for p in created if p.parent == tmp_path / "data"]) == 5


def test_workspace_has_no_legacy_output_root(tmp_path: Path) -> None:
    from src.data.runtime import load_data_runtime

    runtime = load_data_runtime(scope_config=CANONICAL, data_root=tmp_path / "data")
    runtime.workspace.initialize()
    root = tmp_path / "data"
    assert not (root / "archive").exists()
    assert not (root / "artifacts").exists()
    assert not (tmp_path / "data" / "bronze" / "stocks").exists()


def test_relative_root_fails() -> None:
    from src.data.workspace import build_workspace

    scope = _load_canonical()
    with pytest.raises(ValueError, match="absolute"):
        build_workspace(data_root=Path("data"), scope=scope)


def test_workspace_rejects_path_component_scope(tmp_path: Path) -> None:
    from src.data.workspace import build_workspace

    scope = _load_canonical()
    for bad in ("a/b", "a\\b", "..", "/abs"):
        forged = scope.model_copy(update={"scope_id": bad})
        assert forged.scope_id == bad
        with pytest.raises(ValueError, match=r"scope_id|escapes"):
            build_workspace(data_root=tmp_path / "data", scope=forged)
