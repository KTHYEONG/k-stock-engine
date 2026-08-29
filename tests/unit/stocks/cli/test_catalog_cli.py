"""Catalog CLI parser and behavior tests for validate-readiness and inventory."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime

import polars as pl
import pytest

from src.core.datasets import DatasetCertification
from src.stocks.cli import catalog
from src.stocks.data.catalog import (
    CatalogEntry,
    CatalogKind,
    CatalogStore,
    EvidenceCompleteness,
    build_snapshot_manifest,
)
from src.stocks.data.contracts import CoverageRange, ResearchWindows, TimingConvention
from src.storage.parquet_datasets import file_sha256
from tests.fixtures.stocks.helpers import feature_readiness_dataset

REGISTERED = datetime(2026, 1, 1, tzinfo=UTC)
WINDOWS = ResearchWindows(
    train=CoverageRange(start=date(2024, 1, 1), end=date(2024, 1, 31)),
    validation=CoverageRange(start=date(2024, 2, 1), end=date(2024, 2, 15)),
    test=CoverageRange(start=date(2024, 2, 16), end=date(2024, 3, 31)),
)


def _write_snapshot(catalog_root, entries: list[CatalogEntry], snapshot_id: str) -> None:
    manifest = build_snapshot_manifest(
        snapshot_id=snapshot_id,
        certification=DatasetCertification.RESEARCH,
        timing_convention=TimingConvention.DECISION_AFTER_CLOSE_EXECUTE_NEXT_OPEN,
        windows=WINDOWS,
        references=tuple(entries),
    )
    path = catalog_root / "snapshots" / snapshot_id / "snapshot_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_json(), sort_keys=True, indent=2), encoding="utf-8")


def _tree_state(root) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_catalog_validate_readiness_requires_dataset_dir_and_feature() -> None:
    with pytest.raises(SystemExit):
        catalog.main(["validate-readiness"])
    with pytest.raises(SystemExit):
        catalog.main(["validate-readiness", "--dataset-dir", "features"])


def test_catalog_audit_reports_missing_paths(tmp_path, capsys) -> None:
    catalog_root = tmp_path / "catalog"
    store = CatalogStore(catalog_root)
    store.register(
        CatalogEntry(
            kind=CatalogKind.FEATURES,
            name="missing_features",
            content_hash="hash",
            schema_hash="schema",
            registered_at=REGISTERED,
            path=str(tmp_path / "gone"),
        )
    )
    code = catalog.main(["--catalog-root", str(catalog_root), "audit"])
    assert code == 1
    assert "missing\tfeatures\tmissing_features" in capsys.readouterr().out


def test_catalog_validate_readiness_fails_on_unusable_feature(tmp_path, capsys) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    code = catalog.main(
        [
            "--catalog-root",
            str(tmp_path / "catalog"),
            "validate-readiness",
            "--dataset-dir",
            str(dataset_dir),
            "--feature",
            "inactive",
        ]
    )
    assert code == 1
    assert "readiness failed" in capsys.readouterr().out


def test_catalog_validate_readiness_reports_usable_selection(tmp_path, capsys) -> None:
    dataset_dir = feature_readiness_dataset(tmp_path)
    code = catalog.main(
        [
            "--catalog-root",
            str(tmp_path / "catalog"),
            "validate-readiness",
            "--dataset-dir",
            str(dataset_dir),
            "--feature",
            "overnight_ret",
            "--feature",
            "ret_21_60d",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert "feature__inactive" in out


class TestInventory:
    def _fixture(self, tmp_path) -> None:
        self.data_root = tmp_path / "data"
        self.catalog_root = tmp_path / "catalog"
        evidence_dir = self.data_root / "evidence" / "stocks"
        evidence_dir.mkdir(parents=True)
        content = '{"records": [1, 2, 3]}'
        self.evidence_file = evidence_dir / "dart_disclosures_20240101_v1.json"
        self.evidence_file.write_text(content, encoding="utf-8")
        duplicate = evidence_dir / "dart_disclosures_unregistered_duplicate.json"
        duplicate.write_text(content, encoding="utf-8")

        canonical_dir = self.data_root / "canonical" / "stocks" / "base_panel"
        canonical_dir.mkdir(parents=True)
        self.canonical_file = canonical_dir / "base_panel.parquet"
        pl.DataFrame(
            {"instrument_id": [1, 2], "session": ["2024-01-01", "2024-01-02"]}
        ).write_parquet(self.canonical_file)

        store = CatalogStore(self.catalog_root)
        evidence_entry = CatalogEntry(
            kind=CatalogKind.DISCLOSURES,
            name="dart_disclosures_20240101_v1",
            content_hash=file_sha256(self.evidence_file),
            schema_hash="",
            registered_at=REGISTERED,
            completeness=EvidenceCompleteness.COMPLETE,
            path=str(self.evidence_file),
        )
        store.register(evidence_entry)
        _write_snapshot(self.catalog_root, [evidence_entry], "research_v1")

    def test_inventory_emits_stable_json_without_mutation(self, tmp_path, capsys) -> None:
        self._fixture(tmp_path)
        before = _tree_state(self.data_root)

        code = catalog.main(
            [
                "--catalog-root",
                str(self.catalog_root),
                "inventory",
                "--data-root",
                str(self.data_root),
                "--format",
                "json",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        report = json.loads(out)
        assert report["inventory_version"] == 1
        paths = [record["path"] for record in report["records"]]
        assert paths == sorted(paths)
        assert report["summary"]["files"] == 3

        records = {record["path"]: record for record in report["records"]}
        evidence_record = records["evidence/stocks/dart_disclosures_20240101_v1.json"]
        assert evidence_record["lifecycle"] == "evidence"
        assert evidence_record["catalog_reference"] == "disclosures:dart_disclosures_20240101_v1"
        assert evidence_record["snapshot_reference"] == "research_v1"
        assert evidence_record["recommendation"] == "retain"

        duplicate_record = records["evidence/stocks/dart_disclosures_unregistered_duplicate.json"]
        assert duplicate_record["lifecycle"] == "archive_candidate"
        assert duplicate_record["recommendation"] == "candidate"

        canonical_record = records["canonical/stocks/base_panel/base_panel.parquet"]
        assert canonical_record["lifecycle"] == "canonical"
        assert canonical_record["recommendation"] == "retain"

        assert _tree_state(self.data_root) == before

        code = catalog.main(
            [
                "--catalog-root",
                str(self.catalog_root),
                "inventory",
                "--data-root",
                str(self.data_root),
                "--format",
                "json",
            ]
        )
        assert code == 0
        assert capsys.readouterr().out == out

    def test_inventory_text_summary(self, tmp_path, capsys) -> None:
        self._fixture(tmp_path)
        code = catalog.main(
            [
                "--catalog-root",
                str(self.catalog_root),
                "inventory",
                "--data-root",
                str(self.data_root),
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert out.startswith(f"# inventory {self.data_root.resolve()}")
        assert "evidence\t1\t" in out
        assert "canonical\t1\t" in out
        assert "archive_candidate\t1\t" in out
        assert "candidates\t1" in out

    def test_inventory_fails_closed_on_unreadable_file(self, tmp_path, capsys) -> None:
        self._fixture(tmp_path)
        target = self.data_root / "evidence" / "stocks" / "dart_disclosures_20240101_v1.json"
        target.chmod(0o000)
        try:
            code = catalog.main(
                [
                    "--catalog-root",
                    str(self.catalog_root),
                    "inventory",
                    "--data-root",
                    str(self.data_root),
                    "--format",
                    "json",
                ]
            )
            assert code == 1
            assert "inventory failed" in capsys.readouterr().out
        finally:
            target.chmod(0o644)
