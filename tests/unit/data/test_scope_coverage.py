from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from src.data.runtime import load_data_runtime
from src.data.schemas import PITDataError
from src.data.scope_coverage import CoverageRequirement, build_scope_coverage_report

SCOPE_CONFIG = Path("config/research/kr_swing_2019_v1.toml")


def _publish(
    catalog: ReceiptCatalog,
    tmp_path: Path,
    *,
    source: str,
    natural_key: str,
    status: EvidenceStatus,
    as_of: date | None = date(2024, 1, 2),
    fiscal_period: str | None = None,
) -> None:
    import hashlib

    body = f"{source}:{natural_key}:{status.value}".encode()
    payload_path = tmp_path / "blobs" / f"{natural_key}.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(body)
    catalog.publish(
        (
            ReceiptIndexEntry(
                source=source,
                natural_key=natural_key,
                as_of=as_of,
                fiscal_period=fiscal_period,
                status=status,
                content_hash=hashlib.sha256(body).hexdigest(),
                retrieved_at=datetime(2024, 6, 1, tzinfo=UTC),
                payload_path=payload_path,
            ),
        )
    )


def _requirement(
    source: str, natural_key: str, *, required: bool = True, as_of: date | None = date(2024, 1, 2)
) -> CoverageRequirement:
    return CoverageRequirement(
        source=source, natural_key=natural_key, as_of=as_of, fiscal_period=None, required=required
    )


def _runtime(tmp_path: Path):  # type: ignore[no-untyped-def]
    return load_data_runtime(scope_config=SCOPE_CONFIG, data_root=tmp_path / "data")


def test_successful_evidence_fulfills_requirement(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    _publish(catalog, tmp_path, source="krx_daily_market", natural_key="2024-01-02", status=EvidenceStatus.SUCCESS)

    report = build_scope_coverage_report(
        scope=runtime.scope,
        requirements=(_requirement("other", "b"), _requirement("krx_daily_market", "2024-01-02")),
        catalog=catalog,
    )

    assert report.scope_hash == runtime.scope.content_hash
    assert [item.natural_key for item in report.fulfilled] == ["2024-01-02"]
    assert [item.natural_key for item in report.missing] == ["b"]


def test_absent_evidence_is_missing(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")

    report = build_scope_coverage_report(
        scope=runtime.scope, requirements=(_requirement("krx_daily_market", "2024-01-02"),), catalog=catalog
    )

    assert report.fulfilled == ()
    assert report.unresolved == ()
    assert [item.natural_key for item in report.missing] == ["2024-01-02"]


def test_provider_failure_is_unresolved(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    _publish(catalog, tmp_path, source="krx_daily_market", natural_key="2024-01-02", status=EvidenceStatus.PROVIDER_ERROR)

    report = build_scope_coverage_report(
        scope=runtime.scope, requirements=(_requirement("krx_daily_market", "2024-01-02"),), catalog=catalog
    )

    assert report.fulfilled == ()
    assert report.missing == ()
    assert [item.natural_key for item in report.unresolved] == ["2024-01-02"]


def test_optional_source_does_not_block(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")

    report = build_scope_coverage_report(
        scope=runtime.scope,
        requirements=(_requirement("industry", "ind-1", required=False),),
        catalog=catalog,
    )
    assert report.fulfilled == ()
    assert report.missing == ()
    assert report.unresolved == ()

    _publish(catalog, tmp_path, source="industry", natural_key="ind-1", status=EvidenceStatus.SUCCESS)
    observed = build_scope_coverage_report(
        scope=runtime.scope,
        requirements=(_requirement("industry", "ind-1", required=False),),
        catalog=catalog,
    )
    assert [item.natural_key for item in observed.fulfilled] == ["ind-1"]


def test_outside_scope_requirement_fails(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")

    with pytest.raises(PITDataError, match="evidence start"):
        build_scope_coverage_report(
            scope=runtime.scope,
            requirements=(_requirement("krx_daily_market", "2018-12-28", as_of=date(2018, 12, 28)),),
            catalog=catalog,
        )
    with pytest.raises(PITDataError, match="fiscal floor"):
        build_scope_coverage_report(
            scope=runtime.scope,
            requirements=(
                CoverageRequirement(
                    source="financial_facts",
                    natural_key="00126380:2018:11011",
                    as_of=date(2019, 5, 16),
                    fiscal_period="2018Q4",
                    required=True,
                ),
            ),
            catalog=catalog,
        )
    with pytest.raises(PITDataError, match="fiscal period"):
        build_scope_coverage_report(
            scope=runtime.scope,
            requirements=(
                CoverageRequirement(
                    source="financial_facts",
                    natural_key="bad",
                    as_of=None,
                    fiscal_period="20XXQ1",
                    required=True,
                ),
            ),
            catalog=catalog,
        )


def test_scoped_flow_plan_chunks_missing_sessions(tmp_path: Path) -> None:
    from src.data.collection_plan import build_scoped_flow_plan, scoped_checkpoint_dir, scoped_plan_dir
    from src.data.scoped_ingestion import flow_natural_key

    runtime = _runtime(tmp_path)
    catalog = ReceiptCatalog(runtime.workspace.bronze_root / "catalog")
    report = build_scope_coverage_report(
        scope=runtime.scope,
        requirements=(
            _requirement("investor_flow", flow_natural_key(symbol="005930", session=date(2024, 1, 2))),
            _requirement("investor_flow", flow_natural_key(symbol="005930", session=date(2024, 1, 3))),
            _requirement("investor_flow", flow_natural_key(symbol="005930", session=date(2024, 1, 4))),
            _requirement("investor_flow", flow_natural_key(symbol="000001", session=date(2024, 1, 2))),
            _requirement("krx_daily_market", "2024-01-02"),
        ),
        catalog=catalog,
    )

    plan = build_scoped_flow_plan(runtime=runtime, report=report, max_sessions_per_request=2)

    assert plan.coverage_start == runtime.scope.completed_start
    assert plan.coverage_end == runtime.scope.completed_end
    assert [(chunk.symbol, len(chunk.sessions)) for chunk in plan.chunks] == [
        ("000001", 1),
        ("005930", 2),
        ("005930", 1),
    ]
    assert (scoped_plan_dir(runtime=runtime) / f"{plan.plan_id}.json").is_file()
    assert scoped_checkpoint_dir(runtime=runtime).parent == runtime.workspace.state_root
    again = build_scoped_flow_plan(runtime=runtime, report=report, max_sessions_per_request=2)
    assert again.plan_id == plan.plan_id

    with pytest.raises(PITDataError, match="positive integer"):
        build_scoped_flow_plan(runtime=runtime, report=report, max_sessions_per_request=0)
    with pytest.raises(PITDataError, match="positive integer"):
        build_scoped_flow_plan(runtime=runtime, report=report, max_sessions_per_request=True)


def test_scoped_flow_plan_rejects_malformed_keys(tmp_path: Path) -> None:
    from src.data.collection_plan import build_scoped_flow_plan
    from src.data.scope_coverage import ScopeCoverageReport

    runtime = _runtime(tmp_path)
    bad = ScopeCoverageReport(
        scope_hash=runtime.scope.content_hash,
        fulfilled=(),
        missing=(_requirement("investor_flow", "no-separator"),),
        unresolved=(),
    )
    with pytest.raises(PITDataError, match="natural key"):
        build_scoped_flow_plan(runtime=runtime, report=bad, max_sessions_per_request=700)

    bad_date = ScopeCoverageReport(
        scope_hash=runtime.scope.content_hash,
        fulfilled=(),
        missing=(),
        unresolved=(_requirement("investor_flow", "005930:not-a-date"),),
    )
    with pytest.raises(PITDataError, match="natural key"):
        build_scoped_flow_plan(runtime=runtime, report=bad_date, max_sessions_per_request=700)


def test_plan_and_resume_scoped_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from src.data.cli import _parse_args, main

    requirements = [
        {"source": "investor_flow", "natural_key": "005930:2024-01-02", "as_of": "2024-01-02", "required": True},
        {"source": "investor_flow", "natural_key": "005930:2024-01-03", "as_of": "2024-01-03", "required": True},
    ]
    requirements_path = tmp_path / "requirements.json"
    requirements_path.write_text(json.dumps(requirements), encoding="utf-8")

    args = _parse_args(
        ["plan-scoped", "--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data"),
         "--requirements", str(requirements_path), "--max-sessions", "700"]
    )
    assert not hasattr(args, "bronze_root")
    assert not hasattr(args, "artifact_root")
    assert main(["plan-scoped", "--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data"),
                 "--requirements", str(requirements_path), "--max-sessions", "700"]) == 0
    plan_id = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["plan_id"]

    assert main(["resume-scoped", "--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data"),
                 "--plan-id", plan_id]) == 0
    pending = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["pending"]
    assert pending == ["005930-0000"]

    with pytest.raises(SystemExit):
        _parse_args(["plan-scoped", "--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data"),
                     "--requirements", str(requirements_path), "--coverage-start", "2020-01-01"])
    assert main(["plan-scoped", "--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data"),
                 "--requirements", str(tmp_path / "missing.json")]) == 1

    not_list = tmp_path / "not-list.json"
    not_list.write_text(json.dumps({"source": "x"}), encoding="utf-8")
    assert main(["plan-scoped", "--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data"),
                 "--requirements", str(not_list)]) == 1

    not_rows = tmp_path / "not-rows.json"
    not_rows.write_text(json.dumps([42]), encoding="utf-8")
    assert main(["plan-scoped", "--scope-config", str(SCOPE_CONFIG), "--data-root", str(tmp_path / "data"),
                 "--requirements", str(not_rows)]) == 1
