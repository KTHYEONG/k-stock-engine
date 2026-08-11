"""Read-only data inventory: deterministic classification, hashing, fail-closed."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import polars as pl
import pytest

from src.stocks.data.catalog import CatalogStore
from src.stocks.data.inventory import (
    InventoryLifecycle,
    Recommendation,
    classify_lifecycle,
    scan_inventory,
)


class TestLifecycleClassification:
    def test_source_of_truth_and_legacy_paths(self) -> None:
        mapping = {
            "canonical/stocks/base_panel/x/dataset_manifest.json": InventoryLifecycle.CANONICAL,
            "derived/stocks/features/v/part.parquet": InventoryLifecycle.DERIVED,
            "evidence/stocks/dart_parts_single_worker/manifest.json": InventoryLifecycle.EVIDENCE,
            "processed/features/00126800_feat": InventoryLifecycle.ACTIVE_LEGACY,
            "etf_daily/year=2016/2016-01-04_data.parquet": InventoryLifecycle.ACTIVE_LEGACY,
            "market_index/2024/2024-01-04_data.parquet": InventoryLifecycle.ACTIVE_LEGACY,
            "financials.parquet": InventoryLifecycle.ACTIVE_LEGACY,
            "model_features/feature_map.json": InventoryLifecycle.ACTIVE_LEGACY,
            "runtime/trading_state.db": InventoryLifecycle.RUNTIME_STATE,
            "trading_state.db": InventoryLifecycle.RUNTIME_STATE,
            "catalog/stocks/catalog.jsonl": InventoryLifecycle.UNKNOWN,
            "snapshots/stocks/research_1/snapshot_manifest.json": InventoryLifecycle.UNKNOWN,
            "scratch/notes.txt": InventoryLifecycle.UNKNOWN,
        }
        for path, expected in mapping.items():
            assert classify_lifecycle(path) is expected, path


class TestInventoryReport:
    def test_ordering_and_byte_totals(self, tmp_path) -> None:
        root = tmp_path / "data"
        (root / "canonical" / "stocks").mkdir(parents=True)
        (root / "evidence" / "stocks").mkdir(parents=True)
        (root / "canonical" / "stocks" / "b.json").write_text("bb", encoding="utf-8")
        (root / "evidence" / "stocks" / "a.json").write_text("a", encoding="utf-8")

        report = scan_inventory(root, CatalogStore(tmp_path / "catalog"))
        assert [record.path for record in report.records] == [
            "canonical/stocks/b.json",
            "evidence/stocks/a.json",
        ]
        summary = report.summary()
        assert summary["files"] == 2
        assert summary["bytes"] == 3
        by_lifecycle = summary["by_lifecycle"]
        assert by_lifecycle["canonical"] == {"files": 1, "bytes": 2}
        assert by_lifecycle["evidence"] == {"files": 1, "bytes": 1}

    def test_records_carry_streamed_sha256_and_extension(self, tmp_path) -> None:
        root = tmp_path / "data"
        (root / "evidence" / "stocks").mkdir(parents=True)
        (root / "evidence" / "stocks" / "x.JSON").write_text("abc", encoding="utf-8")

        report = scan_inventory(root, CatalogStore(tmp_path / "catalog"))
        record = report.records[0]
        assert record.sha256 == hashlib.sha256(b"abc").hexdigest()
        assert record.byte_count == 3
        assert record.extension == "json"
        assert record.owner == "evidence"

    def test_unknown_and_runtime_paths_are_never_candidates(self, tmp_path) -> None:
        root = tmp_path / "data"
        (root / "mystery").mkdir(parents=True)
        (root / "mystery" / "b_notes.txt").write_text("b", encoding="utf-8")
        (root / "mystery" / "a_notes.txt").write_text("a", encoding="utf-8")
        (root / "trading_state.db").write_text("state", encoding="utf-8")
        pl.DataFrame({"instrument_id": [1]}).write_parquet(root / "financials.parquet")

        report = scan_inventory(root, CatalogStore(tmp_path / "catalog"))
        assert [record.path for record in report.records] == sorted(
            record.path for record in report.records
        )
        records = {record.path: record for record in report.records}
        assert records["mystery/a_notes.txt"].lifecycle is InventoryLifecycle.UNKNOWN
        assert records["mystery/a_notes.txt"].recommendation is Recommendation.RETAIN
        assert records["trading_state.db"].lifecycle is InventoryLifecycle.RUNTIME_STATE
        assert records["financials.parquet"].lifecycle is InventoryLifecycle.ACTIVE_LEGACY
        assert not any(
            record.lifecycle is InventoryLifecycle.ARCHIVE_CANDIDATE
            for record in report.records
        )

    def test_changed_file_fails_closed(self, tmp_path, monkeypatch) -> None:
        root = tmp_path / "data"
        (root / "evidence" / "stocks").mkdir(parents=True)
        target = root / "evidence" / "stocks" / "x.json"
        target.write_text("{}", encoding="utf-8")

        real_stat = Path.stat
        counts: dict[Path, int] = {}

        def flaky_stat(self, *args, **kwargs):
            counts[self] = counts.get(self, 0) + 1
            result = real_stat(self, *args, **kwargs)
            # Per-path stat order: is_symlink=1, is_file=2, then _stream_sha256
            # before=3, after=4. Tampering call 3 makes the pre-hash size
            # differ from the post-hash size, simulating an appended file.
            if self == target and counts[self] == 3:
                result = os.stat_result(
                    (
                        result.st_mode,
                        result.st_ino,
                        result.st_dev,
                        result.st_nlink,
                        result.st_uid,
                        result.st_gid,
                        result.st_size + 1,
                        result.st_atime,
                        result.st_mtime,
                        result.st_ctime,
                    )
                )
            return result

        monkeypatch.setattr(Path, "stat", flaky_stat)
        with pytest.raises(ValueError, match="changed during inventory"):
            scan_inventory(root, CatalogStore(tmp_path / "catalog"))

    def test_unreadable_file_fails_closed(self, tmp_path) -> None:
        root = tmp_path / "data"
        (root / "evidence" / "stocks").mkdir(parents=True)
        target = root / "evidence" / "stocks" / "x.json"
        target.write_text("{}", encoding="utf-8")
        target.chmod(0o000)
        try:
            with pytest.raises(ValueError, match="unreadable file"):
                scan_inventory(root, CatalogStore(tmp_path / "catalog"))
        finally:
            target.chmod(0o644)
