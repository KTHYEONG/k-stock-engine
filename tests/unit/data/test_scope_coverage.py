from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.data.receipt_catalog import EvidenceStatus, ReceiptCatalog, ReceiptIndexEntry
from src.data.runtime import load_data_runtime
from src.core.pit import PITDataError
from src.data.scope_coverage import CoverageRequirement, build_scope_coverage_report
from tests.fixtures import seed_receipts

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
    payload_path = tmp_path / "data" / "bronze" / "kr_swing_2019_v1" / "blobs" / f"{natural_key}.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(body)
    seed_receipts(
        catalog,
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
        ),
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
            requirements=(_requirement("krx_daily_market", "2015-12-30", as_of=date(2015, 12, 30)),),
            catalog=catalog,
        )
    with pytest.raises(PITDataError, match="fiscal floor"):
        build_scope_coverage_report(
            scope=runtime.scope,
            requirements=(
                CoverageRequirement(
                    source="financial_facts",
                    natural_key="00126380:2015:11011",
                    as_of=date(2016, 5, 16),
                    fiscal_period="2015Q4",
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
